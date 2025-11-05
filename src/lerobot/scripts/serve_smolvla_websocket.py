"""
WebSocket server adapter for LeRobot SmolVLA policy to work with OmniGibson/Isaac Sim.

This script loads a LeRobot SmolVLA model and serves it via WebSocket protocol,
compatible with OmniGibson's WebsocketClientPolicy interface.

Usage:
    python src/lerobot/scripts/serve_smolvla_websocket.py \
        --pretrained_name_or_path=~/workspace/smolvla-task0000/checkpoints/step_025000 \
        --device=cuda \
        --host=0.0.0.0 \
        --port=8000 \
        --task="Complete the task"

Then in eval.py config, use:
    policy=websocket
"""

import asyncio
import functools
import http
import logging
import msgpack
import numpy as np
import time
import torch
import traceback
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import websockets
try:
    import websockets.asyncio.server as _server
except ImportError:
    import websockets.server as _server

from lerobot.policies.factory import get_policy_class

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for SmolVLA WebSocket Server"""
    pretrained_name_or_path: str
    policy_type: str = "smolvla"
    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = 8000
    task: str = "Turn on the radio receiver that's on the table in the living room."  # Task description for SmolVLA
    # Image features expected by policy (should match training config)
    image_features: Optional[list] = None


class SmolVLAWebSocketServer:
    """Serves LeRobot SmolVLA policy via WebSocket protocol compatible with OmniGibson"""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Expand path if it contains ~
        model_path = Path(config.pretrained_name_or_path).expanduser()
        logger.info(f"Loading {config.policy_type} policy from {model_path}")

        policy_class = get_policy_class(config.policy_type)

        # Load dataset stats from preprocessor file if it exists
        dataset_stats = self._load_dataset_stats(str(model_path))
        if dataset_stats is not None:
            logger.info("Loaded dataset stats from preprocessor file")

        self.policy = policy_class.from_pretrained(str(model_path), dataset_stats=dataset_stats)
        self.policy.to(self.device)
        self.policy.eval()

        logger.info(f"Policy loaded successfully on {self.device}")
        logger.info(f"Policy expects image features: {self.policy.config.input_features}")
        logger.info(f"Using task description: '{config.task}'")

        # Metadata to send to client
        # Convert image_features to serializable format (list of feature names)
        image_feature_names = list(self.policy.config.input_features.keys()) if hasattr(self.policy.config, 'input_features') else []

        self._metadata = {
            "policy_type": config.policy_type,
            "pretrained_name_or_path": str(model_path),
            "device": str(self.device),
            "image_features": image_feature_names,
            "task": config.task,
        }

        # Episode state
        self.reset()

        # Track whether we've warned about missing modalities
        self._warned_missing_depth_seg = False

    def _load_dataset_stats(self, model_path: str) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """Load dataset stats from preprocessor safetensors file"""
        from safetensors.torch import load_file
        from pathlib import Path

        model_path = Path(model_path)
        # SmolVLA uses step_5 instead of step_3
        stats_file = model_path / "policy_preprocessor_step_5_normalizer_processor.safetensors"

        if not stats_file.exists():
            logger.warning(f"No stats file found at {stats_file}")
            return None

        try:
            # Load all stats from the file
            all_stats = load_file(str(stats_file))

            # Reorganize stats into the format expected by the policy:
            # {feature_name: {'mean': tensor, 'std': tensor, 'min': tensor, 'max': tensor, ...}}
            dataset_stats = {}

            # Extract unique feature names
            feature_names = set()
            for key in all_stats.keys():
                # Keys are like "observation.state.mean", "observation.state.std", etc.
                feature_name = '.'.join(key.split('.')[:-1])  # Remove the stat type suffix
                feature_names.add(feature_name)

            # Organize stats by feature
            for feature_name in feature_names:
                dataset_stats[feature_name] = {}
                for stat_type in ['mean', 'std', 'min', 'max', 'q01', 'q99', 'count']:
                    stat_key = f"{feature_name}.{stat_type}"
                    if stat_key in all_stats:
                        dataset_stats[feature_name][stat_type] = all_stats[stat_key]

            # Add dummy stats for observation.task_info if missing (common issue with task embeddings)
            if "observation.task_info" not in dataset_stats:
                logger.warning("observation.task_info stats not found, creating dummy stats")
                # Create dummy stats (identity normalization: mean=0, std=1)
                task_info_dim = 46  # From config
                dataset_stats["observation.task_info"] = {
                    "mean": torch.zeros(task_info_dim),
                    "std": torch.ones(task_info_dim),
                }

            logger.info(f"Loaded stats for features: {list(dataset_stats.keys())}")
            return dataset_stats

        except Exception as e:
            logger.error(f"Error loading dataset stats: {e}")
            return None

    def reset(self):
        """Reset policy state for new episode"""
        # SmolVLA policies may have internal state for action chunking. Reset here if needed.
        self.action_chunk = None
        self.chunk_idx = 0
        self._warned_missing_depth_seg = False
        logger.info("Policy state reset")

    def preprocess_observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Convert OmniGibson observation format to LeRobot format for b1k-task0000 model.

        Expected model inputs:
        - observation.images.rgb.{left_wrist,right_wrist,head}
        - observation.images.depth.{left_wrist,right_wrist,head}
        - observation.images.seg_instance_id.{left_wrist,right_wrist,head}
        - observation.cam_rel_poses: [21]
        - observation.state: [256] (proprioception)
        - observation.task_info: [46]
        - task: string (task description for SmolVLA)
        """
        lerobot_obs = {}

        # Camera mapping: OmniGibson -> LeRobot format
        # Using the actual OmniGibson R1Pro camera names (note: double robot_r1:: in the full key)
        camera_mapping = {
            "robot_r1::robot_r1:left_realsense_link:Camera:0": ("left_wrist", (480, 480)),
            "robot_r1::robot_r1:right_realsense_link:Camera:0": ("right_wrist", (480, 480)),
            "robot_r1::robot_r1:zed_link:Camera:0": ("head", (720, 720)),
        }

        # Modality mapping: OmniGibson suffix -> LeRobot modality name
        modality_mapping = {
            "rgb": "rgb",
            "depth": "depth",
            "seg_instance_id": "seg_instance_id",
        }

        # Process all image modalities (RGB, depth, segmentation)
        # Note: OmniGibson only sends RGB by default, depth and seg are not available
        for og_camera, (lr_camera, target_size) in camera_mapping.items():
            for og_modality, lr_modality in modality_mapping.items():
                og_key = f"{og_camera}::{og_modality}"  # Keys already include robot_r1::
                lr_key = f"observation.images.{lr_modality}.{lr_camera}"

                if og_key in obs:
                    img = obs[og_key]  # [H, W, 3] or [H, W, C]

                    # Resize to target resolution
                    if img.shape[:2] != target_size:
                        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

                    # Normalize to [0, 1] if uint8
                    if img.dtype == np.uint8:
                        img = img.astype(np.float32) / 255.0
                    else:
                        img = img.astype(np.float32)

                    # Handle different channel counts
                    if img.ndim == 2:
                        # Single channel (e.g., depth) -> replicate to 3 channels
                        img = np.stack([img, img, img], axis=-1)  # [H, W, 3]
                    elif img.shape[-1] == 1:
                        # Single channel -> replicate to 3 channels
                        img = np.repeat(img, 3, axis=-1)
                    elif img.shape[-1] == 4:
                        # RGBA -> drop alpha channel, keep only RGB
                        img = img[:, :, :3]

                    # Convert to tensor: [H, W, 3] -> [1, 3, H, W]
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                    lerobot_obs[lr_key] = img_tensor.to(self.device)
                else:
                    # If modality not available, create zero tensor with correct shape
                    # Only warn once for depth/seg since OmniGibson doesn't send them by default
                    if og_modality in ["depth", "seg_instance_id"] and not self._warned_missing_depth_seg:
                        logger.warning(f"Depth and segmentation images not available from OmniGibson, using zeros")
                        self._warned_missing_depth_seg = True
                    elif og_modality == "rgb":
                        logger.warning(f"Missing RGB observation: {og_key}, using zeros")

                    img_tensor = torch.zeros(1, 3, target_size[0], target_size[1])
                    lerobot_obs[lr_key] = img_tensor.to(self.device)

        # Camera relative poses [21] (7 per camera: xyz position + xyzw quaternion)
        if "robot_r1::cam_rel_poses" in obs:
            cam_rel_poses = torch.from_numpy(obs["robot_r1::cam_rel_poses"]).unsqueeze(0)  # [1, 21]
            lerobot_obs["observation.cam_rel_poses"] = cam_rel_poses.to(self.device).float()
        else:
            logger.warning("Missing observation.cam_rel_poses, using zeros")
            lerobot_obs["observation.cam_rel_poses"] = torch.zeros(1, 21).to(self.device)

        # Proprioception state [256]
        if "robot_r1::proprio" in obs:
            state = torch.from_numpy(obs["robot_r1::proprio"]).unsqueeze(0)  # [1, state_dim]
            lerobot_obs["observation.state"] = state.to(self.device).float()
        else:
            logger.warning("Missing observation.state (robot_r1::proprio), using zeros")
            lerobot_obs["observation.state"] = torch.zeros(1, 256).to(self.device)

        # Task info [46] - this may need special handling
        # Option 1: If task_id is available, create one-hot or embedding
        if "task_id" in obs:
            task_id = obs["task_id"]
            # For now, create a simple one-hot encoding (assuming 46 tasks)
            # You may need to adjust this based on actual task encoding
            task_info = np.zeros(46, dtype=np.float32)
            if isinstance(task_id, np.ndarray):
                task_id = int(task_id.item())
            if task_id < 46:
                task_info[task_id] = 1.0
            lerobot_obs["observation.task_info"] = torch.from_numpy(task_info).unsqueeze(0).to(self.device)
        else:
            # Option 2: Use zeros if no task info available
            logger.warning("Missing observation.task_info, using zeros")
            lerobot_obs["observation.task_info"] = torch.zeros(1, 46).to(self.device)

        # Add task description for SmolVLA (this will be tokenized by the preprocessor)
        lerobot_obs["task"] = self.config.task

        return lerobot_obs

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get action from policy given observation.

        Returns:
            action: np.ndarray of shape [action_dim]
        """
        try:
            # Preprocess observation
            lerobot_obs = self.preprocess_observation(obs)

            # Get action from policy
            with torch.no_grad():
                # SmolVLA may return action chunks
                action = self.policy.select_action(lerobot_obs)

                # If policy returns batched action, unbatch
                if action.ndim > 1:
                    action = action.squeeze(0)  # [action_dim]

                # Convert to numpy
                action_np = action.cpu().numpy()

            return action_np

        except Exception as e:
            logger.error(f"Error in act(): {e}")
            logger.error(traceback.format_exc())
            raise

    async def run(self):
        """Start the WebSocket server"""
        logger.info(f"Starting SmolVLA WebSocket server on {self.config.host}:{self.config.port}...")
        async with _server.serve(
            self._handler,
            self.config.host,
            self.config.port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket):
        """Handle WebSocket connection"""
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        # Send metadata to client
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                result = unpackb(await websocket.recv())

                # Handle reset command
                if "reset" in result:
                    self.reset()
                    continue

                obs = result

                # Run inference
                infer_time = time.monotonic()
                action_np = self.act(obs)
                infer_time = time.monotonic() - infer_time

                # Prepare response
                response = {
                    "action": action_np,
                    "server_timing": {
                        "infer_ms": infer_time * 1000,
                    }
                }
                if prev_total_time is not None:
                    response["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(response))
                prev_total_time = time.monotonic() - start_time

                logger.info(f"Inference: {infer_time*1000:.2f}ms | Action shape: {action_np.shape}")

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                logger.error(f"Error in connection from {websocket.remote_address}:\n{traceback.format_exc()}")
                await websocket.send(traceback.format_exc())
                try:
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error",
                    )
                except:
                    pass
                raise

    def _health_check(self, connection, request) -> Optional[Any]:
        """Health check endpoint for client to verify server is ready"""
        if hasattr(request, "path") and request.path == "/healthz":
            if hasattr(connection, "respond"):
                return connection.respond(http.HTTPStatus.OK, "OK\n")
            else:
                return http.HTTPStatus.OK, {"Content-Type": "text/plain"}, b"OK\n"
        return None

    def serve_forever(self):
        """Start serving (blocking)"""
        asyncio.run(self.run())


# msgpack NumPy array serialization helpers (from OmniGibson)
def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Serve LeRobot SmolVLA via WebSocket")
    parser.add_argument("--pretrained_name_or_path", type=str, required=True,
                       help="Path to pretrained SmolVLA model")
    parser.add_argument("--policy_type", type=str, default="smolvla",
                       help="Policy type (default: smolvla)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run inference on (default: cuda)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind server to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind server to (default: 8000)")
    parser.add_argument("--task", type=str, default="Turn on the radio receiver that's on the table in the living room.",
                       help="Task description for SmolVLA (default: 'Turn on the radio receiver that's on the table in the living room.')")

    args = parser.parse_args()

    config = ServerConfig(
        pretrained_name_or_path=args.pretrained_name_or_path,
        policy_type=args.policy_type,
        device=args.device,
        host=args.host,
        port=args.port,
        task=args.task,
    )

    server = SmolVLAWebSocketServer(config)
    server.serve_forever()


if __name__ == "__main__":
    main()
