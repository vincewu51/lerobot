# Integrating LeRobot ACT with Isaac Sim / OmniGibson

This guide explains how to use LeRobot's ACT policy with OmniGibson/Isaac Sim for the BEHAVIOR-1K challenge.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  Isaac Sim / OmniGibson (Client)                                 │
│                                                                   │
│  eval.py                                                          │
│  ├─ Creates simulation environment                               │
│  ├─ Loads WebsocketPolicy (connects to ws://host:port)           │
│  └─ Main loop:                                                    │
│      1. obs = env.get_observation()                              │
│      2. action = policy.forward(obs)   ◄───── WebSocket ─────┐  │
│      3. env.step(action)                                      │  │
└───────────────────────────────────────────────────────────────┼──┘
                                                                 │
                                                                 │
┌─────────────────────────────────────────────────────────────┼───┐
│  LeRobot ACT Server                                          │   │
│                                                              │   │
│  serve_act_websocket.py                                      │   │
│  ├─ Loads ACT policy checkpoint                              │   │
│  ├─ Starts WebSocket server                                  │   │
│  └─ On each request:                                         │   │
│      1. Receives obs from client                            ─┘   │
│      2. Preprocesses to LeRobot format                           │
│      3. action = policy.select_action(obs)                       │
│      4. Returns action to client                                 │
└──────────────────────────────────────────────────────────────────┘
```

## Setup Instructions

### 1. Start the LeRobot ACT WebSocket Server

```bash
# Using a pretrained model from HuggingFace
python src/lerobot/scripts/serve_act_websocket.py \
    --pretrained_name_or_path=lerobot/act_aloha_sim_transfer_cube \
    --device=cuda \
    --host=0.0.0.0 \
    --port=8000

# Or using your own trained model
python src/lerobot/scripts/serve_act_websocket.py \
    --pretrained_name_or_path=/path/to/your/checkpoint \
    --device=cuda \
    --host=0.0.0.0 \
    --port=8000
```

**Important**: Make sure port 8000 is available. If not, change the port with `--port=8001` and update the eval config accordingly.

### 2. Run OmniGibson Evaluation

```bash
cd /path/to/BEHAVIOR-1K/OmniGibson

# Run evaluation with WebSocket policy
python omnigibson/learning/eval.py \
    policy=websocket \
    task.name=turning_on_radio \
    log_path=~/eval_results \
    headless=true \
    write_video=true
```

The `policy=websocket` config will automatically connect to `ws://0.0.0.0:8000`.

If you changed the port, update the config:
```bash
python omnigibson/learning/eval.py \
    policy=websocket \
    model.port=8001 \
    task.name=turning_on_radio \
    ...
```

## Key Files

### LeRobot Side:
- **`src/lerobot/scripts/serve_act_websocket.py`**: WebSocket server that serves ACT policy
  - Loads ACT checkpoint
  - Converts OmniGibson observations to LeRobot format
  - Returns actions via WebSocket

### OmniGibson Side:
- **`omnigibson/learning/eval.py`**: Main evaluation script
- **`omnigibson/learning/policies.py`**: Contains `WebsocketPolicy` wrapper
- **`omnigibson/learning/utils/network_utils.py`**: `WebsocketClientPolicy` implementation
- **`omnigibson/learning/configs/policy/websocket.yaml`**: Config for WebSocket policy

## Important Customization Points

### 1. Observation Mapping

The **most critical part** is mapping OmniGibson observations to LeRobot format in `serve_act_websocket.py`:

```python
def preprocess_observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """
    OmniGibson provides:
    - "robot_r1::head_camera::rgb": [H, W, 3] uint8
    - "robot_r1::left_wrist_camera::rgb": [H, W, 3] uint8
    - "robot_r1::proprioception": [proprio_dim] float32

    LeRobot ACT expects (example):
    - "observation.images.top": [1, 3, H, W] float32 [0,1]
    - "observation.state": [1, state_dim] float32
    """
    # TODO: Customize based on your model's training data!
```

**You MUST modify this function to match:**
- Camera names in your OmniGibson setup
- Image feature keys your ACT model was trained with
- State/proprioception dimensions

### 2. Action Space

Ensure the action space matches:
- **OmniGibson R1Pro robot**: Typically 14-20 DoF (depends on controller)
- **Your ACT model**: Check `policy.config.output_features`

If dimensions don't match, you may need to:
- Add action mapping/padding in `serve_act_websocket.py`
- Retrain ACT with correct action dimension

### 3. Image Resolution

ACT models are sensitive to image resolution. Options:
1. **Resize in server**: Modify `preprocess_observation()` to resize images
2. **Configure OmniGibson**: Set camera resolution in robot config
3. **Retrain**: Fine-tune ACT on OmniGibson resolution

## Debugging

### Check Server is Running
```bash
curl http://localhost:8000/healthz
# Should return: OK
```

### Enable Debug Logging
In `serve_act_websocket.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Common Issues

**1. Port already in use**
```
RuntimeError: Failed to bind to address 0.0.0.0:8000
```
Solution: Use different port with `--port=8001`

**2. Connection refused**
```
Connection from client failed
```
Solution:
- Check firewall settings
- Use `--host=127.0.0.1` for local testing
- Verify server started successfully

**3. Shape mismatch errors**
```
RuntimeError: Expected tensor of shape [1, 3, 224, 224], got [1, 3, 480, 640]
```
Solution: Add image resizing in `preprocess_observation()`

**4. Action dimension mismatch**
```
RuntimeError: Action dim 14 doesn't match expected 7
```
Solution: Check robot DoF and policy output dimension, add mapping layer

## Comparison with LeRobot's gRPC Server

| Feature | WebSocket (this adapter) | gRPC (lerobot async_inference) |
|---------|-------------------------|--------------------------------|
| **Protocol** | WebSocket + msgpack | gRPC + protobuf |
| **Compatibility** | ✅ Works with OmniGibson | ❌ Requires custom client |
| **Setup Complexity** | Simple (1 server) | Complex (server + robot client) |
| **Action Chunking** | Manual implementation | Built-in |
| **FPS Tracking** | Manual | Built-in |
| **Use Case** | Sim evaluation | Real robot control |

For **Isaac Sim/OmniGibson evaluation**, use the WebSocket adapter.

For **real robot deployment**, use LeRobot's gRPC async_inference system.

## Alternative: Adapt gRPC to OmniGibson

If you prefer using LeRobot's full gRPC infrastructure, you would need to:
1. Create an `OmniGibsonRobotClient` that implements the `RobotClient` interface
2. Wrap OmniGibson environment as a robot
3. Map observations/actions between formats

This is more complex but gives you action chunking and FPS tracking.

## Next Steps

1. **Test the connection**: Start server and run eval.py
2. **Customize observation mapping**: Edit `preprocess_observation()`
3. **Validate action space**: Verify actions execute correctly in sim
4. **Tune performance**: Adjust image resolution, batch size, etc.
5. **Fine-tune policy**: If needed, train ACT on OmniGibson data

## Questions?

Check:
- LeRobot docs: https://github.com/huggingface/lerobot
- OmniGibson docs: https://behavior.stanford.edu/omnigibson/
- BEHAVIOR-1K challenge: https://behavior.stanford.edu/behavior-1k/

