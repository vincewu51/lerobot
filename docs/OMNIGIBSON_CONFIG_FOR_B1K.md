# OmniGibson Configuration for b1k-task0000 Model

The b1k-task0000 model expects **9 image modalities** (RGB, depth, segmentation for 3 cameras).

## Required Changes to OmniGibson eval.py

By default, OmniGibson only provides RGB and proprioception. You need to enable depth and segmentation.

### Option 1: Modify eval.py directly

In `/path/to/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py`, line 138:

**Current:**
```python
cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
```

**Change to:**
```python
cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb", "depth", "seg_instance_id"]
```

### Option 2: Override via command line

```bash
python omnigibson/learning/eval.py \
    policy=websocket \
    task.name=turning_on_radio \
    robot.obs_modalities="['proprio','rgb','depth','seg_instance_id']" \
    log_path=~/eval_results
```

## Verify Observations

Add this debug code in `serve_act_websocket.py` to verify observations:

```python
def preprocess_observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    # Add at the beginning
    logger.info(f"Received observation keys: {list(obs.keys())}")
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    # ... rest of function
```

## Expected Observation Keys

When properly configured, you should see:

```
robot_r1::head_camera::rgb              [H, W, 3] uint8
robot_r1::head_camera::depth            [H, W, 1] or [H, W] float32
robot_r1::head_camera::seg_instance_id  [H, W, 1] or [H, W, 3] int32

robot_r1::left_wrist_camera::rgb              [H, W, 3] uint8
robot_r1::left_wrist_camera::depth            [H, W, 1] or [H, W] float32
robot_r1::left_wrist_camera::seg_instance_id  [H, W, 1] or [H, W, 3] int32

robot_r1::right_wrist_camera::rgb              [H, W, 3] uint8
robot_r1::right_wrist_camera::depth            [H, W, 1] or [H, W] float32
robot_r1::right_wrist_camera::seg_instance_id  [H, W, 1] or [H, W, 3] int32

robot_r1::proprioception  [256] float32
robot_r1::cam_rel_poses   [21] float32
task_id                   [1] int64
```

## Camera Resolution

The model expects:
- **Wrist cameras**: 480x480
- **Head camera**: 720x720

OmniGibson may provide different resolutions. The server will automatically resize, but for best performance, configure OmniGibson to match these resolutions.

## Task Info Encoding

The model expects `observation.task_info` with shape [46], representing 46 different BEHAVIOR-1K tasks.

Currently, the server converts `task_id` to one-hot encoding. If your training used a different encoding (e.g., learned embedding, language features), you'll need to update the preprocessing accordingly.
