"""
WebSocket server adapter for LeRobot ACT policy to work with OmniGibson/Isaac Sim.

This script loads a LeRobot ACT model and serves it via WebSocket protocol,
compatible with OmniGibson's WebsocketClientPolicy interface.

Usage:
    python src/lerobot/scripts/serve_act_websocket.py \
        --pretrained_name_or_path=lerobot/act_aloha_sim_transfer_cube \
        --device=cuda \
        --host=0.0.0.0 \
        --port=8000

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
    """Configuration for ACT WebSocket Server"""
    pretrained_name_or_path: str
    policy_type: str = "act"
    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = 8000
    # Image features expected by policy (should match training config)
    # For ACT trained on ALOHA: ["observation.images.top", "observation.images.wrist"]
    image_features: Optional[list] = None


class ACTWebSocketServer:
    """Serves LeRobot ACT policy via WebSocket protocol compatible with OmniGibson"""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Load policy
        logger.info(f"Loading {config.policy_type} policy from {config.pretrained_name_or_path}")
        policy_class = get_policy_class(config.policy_type)
        self.policy = policy_class.from_pretrained(config.pretrained_name_or_path)
        self.policy.to(self.device)
        self.policy.eval()

        logger.info(f"Policy loaded successfully on {self.device}")
        logger.info(f"Policy expects image features: {self.policy.config.image_features}")

        # Metadata to send to client
        self._metadata = {
            "policy_type": config.policy_type,
            "pretrained_name_or_path": config.pretrained_name_or_path,
            "device": str(self.device),
            "image_features": self.policy.config.image_features,
        }

        # Episode state
        self.reset()

    def reset(self):
        """Reset policy state for new episode"""
        # ACT policies are typically stateless per-step, but may have internal state
        # for action chunking. Reset here if needed.
        self.action_chunk = None
        self.chunk_idx = 0
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
        """
        lerobot_obs = {}

        # Camera mapping: OmniGibson -> LeRobot format
        camera_mapping = {
            "left_wrist_camera": ("left_wrist", (480, 480)),
            "right_wrist_camera": ("right_wrist", (480, 480)),
            "head_camera": ("head", (720, 720)),
        }

        # Modality mapping: OmniGibson suffix -> LeRobot modality name
        modality_mapping = {
            "rgb": "rgb",
            "depth": "depth",
            "seg_instance_id": "seg_instance_id",
        }

        # Process all image modalities (RGB, depth, segmentation)
        for og_camera, (lr_camera, target_size) in camera_mapping.items():
            for og_modality, lr_modality in modality_mapping.items():
                og_key = f"robot_r1::{og_camera}::{og_modality}"
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

                    # Ensure 3 channels (depth might be single channel)
                    if img.ndim == 2:
                        img = np.stack([img, img, img], axis=-1)  # [H, W, 3]
                    elif img.shape[-1] == 1:
                        img = np.repeat(img, 3, axis=-1)

                    # Convert to tensor: [H, W, 3] -> [1, 3, H, W]
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                    lerobot_obs[lr_key] = img_tensor.to(self.device)
                else:
                    # If modality not available, create zero tensor with correct shape
                    logger.warning(f"Missing observation: {og_key}, using zeros")
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
        if "robot_r1::proprioception" in obs:
            state = torch.from_numpy(obs["robot_r1::proprioception"]).unsqueeze(0)  # [1, state_dim]
            lerobot_obs["observation.state"] = state.to(self.device).float()
        else:
            logger.warning("Missing observation.state, using zeros")
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
                # ACT may return action chunks
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
        logger.info(f"Starting ACT WebSocket server on {self.config.host}:{self.config.port}...")
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

    parser = argparse.ArgumentParser(description="Serve LeRobot ACT via WebSocket")
    parser.add_argument("--pretrained_name_or_path", type=str, required=True,
                       help="Path to pretrained ACT model")
    parser.add_argument("--policy_type", type=str, default="act",
                       help="Policy type (default: act)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run inference on (default: cuda)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind server to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind server to (default: 8000)")

    args = parser.parse_args()

    config = ServerConfig(
        pretrained_name_or_path=args.pretrained_name_or_path,
        policy_type=args.policy_type,
        device=args.device,
        host=args.host,
        port=args.port,
    )

    server = ACTWebSocketServer(config)
    server.serve_forever()


if __name__ == "__main__":
    main()
