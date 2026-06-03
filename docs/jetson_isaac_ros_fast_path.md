# Jetson Isaac ROS Grounding DINO Fast Path

Date: 2026-06-02

This checklist is for the next Jetson Orin Nano deployment attempt after the local Isaac ROS Grounding DINO proof-of-life succeeded on the x86 RTX 4090 workstation.

The goal for the first Jetson session is not full ADAONE integration. The goal is to quickly answer:

```text
Can the official Isaac ROS Grounding DINO model/engine load and publish detections on this Jetson?
```

## The Most Important Shortcut

Do not start by replaying a full rosbag or integrating with ADAONE. Start with a one-frame proof-of-life:

1. Confirm Jetson/JetPack/Ubuntu/ROS/Docker compatibility.
2. Build or install the Isaac ROS environment.
3. Generate the TensorRT plan on the Jetson.
4. Launch the official graph with no camera load.
5. Set a prompt.
6. Publish one 640x480 image and check `/detections_output`.
7. Only then replay a short rosbag segment.

This keeps each failure mode small.

## Step 0: Run The Preflight Script

From this repo on the Jetson:

```bash
bash tools/isaac_ros/jetson_preflight.sh | tee jetson_preflight_$(date +%Y%m%d_%H%M%S).log
```

Read the "Isaac ROS Compatibility Hint" section first.

### Likely Path A: Ubuntu 24.04 / Jazzy / Newer JetPack

Try the official Isaac ROS `release-4.4` binary/container path first.

The current Isaac ROS 4.4 documentation publishes `noble` and `noble-jetpack` apt repositories and tests against CUDA 13.0 and TensorRT 10.13.3. This is close to the environment that worked locally in the Isaac ROS container.

### Likely Path B: Ubuntu 22.04 / Humble / JetPack 6.x

This is the likely Jetson Orin Nano state.

Do not spend the first hours forcing Isaac ROS 4.4 Jazzy bare-metal packages into the Humble system. The older Isaac ROS 3.2 line supports JetPack 6.1/6.2 and Humble, but Grounding DINO was not the path we validated locally. In this case, try one of these in order:

1. Isaac ROS container path, if NVIDIA provides a compatible arm64 JetPack image for the device.
2. Reuse the official ONNX/model preparation ideas, but keep our local Humble wrapper as the ROS boundary.
3. Fall back to our PyTorch ROS2 node for functional tests while treating Isaac ROS as the reference implementation.

## What To Transfer From The Local Workstation

Useful:

- This repo.
- The official Grounding DINO ONNX model.
- This checklist and helper scripts.
- A few sample frames or a very short rosbag.

Not useful as a primary deployment artifact:

- The x86 RTX 4090 TensorRT `.plan`.

TensorRT plans are tied to GPU architecture, TensorRT version, CUDA version, plugins, and builder settings. Build the Jetson `.plan` on the Jetson.

To package local helper assets:

```bash
bash tools/isaac_ros/package_local_assets_for_jetson.sh
```

This writes a tarball under:

```text
/home/boyang/safeai/GroundingDINO/outputs/isaac_ros/jetson_transfer/
```

## Jetson Setup Before Model Build

Use max performance mode before building or loading the engine:

```bash
sudo nvpmodel -m 0 || true
sudo jetson_clocks || true
```

Open a second terminal:

```bash
sudo tegrastats
```

Close unrelated nodes, browser tabs, camera visualizers, and previous Docker containers. The local TensorRT plan was about 696 MB, and runtime buffers are much larger than the plan file alone.

## Official Model Install Command

Inside the Isaac ROS environment:

```bash
source /opt/ros/jazzy/setup.bash
export ISAAC_ROS_WS=/workspaces/isaac_ros-dev
export ISAAC_ROS_ACCEPT_EULA=1
ros2 run isaac_ros_grounding_dino_models_install install_grounding_dino_models.sh
```

Expected output artifacts:

```text
$ISAAC_ROS_WS/isaac_ros_assets/models/grounding_dino/grounding_dino_model.onnx
$ISAAC_ROS_WS/isaac_ros_assets/models/grounding_dino/grounding_dino_model.plan
```

If the engine build fails from OOM, capture:

```bash
free -h
df -h /
tegrastats log around the failure
full TensorRT error
```

Then try again after stopping all nonessential processes. Only after that should we decide whether to rebuild at a smaller network shape.

## Launch The Reference Graph

Inside the Isaac ROS environment:

```bash
bash tools/isaac_ros/launch_grounding_dino_reference.sh
```

Equivalent direct command:

```bash
ros2 launch isaac_ros_grounding_dino isaac_ros_grounding_dino.launch.py \
  input_image_width:=640 input_image_height:=480 \
  network_image_width:=960 network_image_height:=544 \
  model_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/grounding_dino/grounding_dino_model.onnx \
  engine_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/grounding_dino/grounding_dino_model.plan \
  confidence_threshold:=0.3
```

Do not change `network_image_width` / `network_image_height` unless the engine was rebuilt for the new shape.

## Set Prompt

Use period-separated labels:

```bash
bash tools/isaac_ros/set_grounding_dino_prompt.sh \
  "box.monitor.toolbox.small robot car.chair.stop sign."
```

Outdoor prompt:

```bash
bash tools/isaac_ros/set_grounding_dino_prompt.sh \
  "fire hydrant.vehicle.car.truck.garbage bin.construction barrel.stop sign."
```

## One-Frame Test Before Rosbag Replay

Check interfaces:

```bash
ros2 topic list | sort
ros2 service list | grep prompt
ros2 topic info /detections_output
```

Expected:

```text
/image
/camera_info
/set_prompt
/detections_output
```

Publish one image repeatedly for a few seconds and listen for:

```bash
ros2 topic echo /detections_output --once
```

If the first message after a prompt switch is empty, do not conclude failure immediately. In local testing, reading a stale or transitional output right after switching prompts caused a false zero-detection result. Publish several frames and use the latest message.

## Full Rosbag Test Order

Once the one-frame test works:

1. Replay 5 to 10 seconds from indoor bag.
2. Save detection JSON only.
3. Then save annotated video.
4. Then try outdoor bag.
5. Only after that bridge into ADAONE topic names.

Suggested first prompts:

Indoor:

```text
box.monitor.toolbox.small robot car.chair.stop sign.
```

Outdoor:

```text
fire hydrant.vehicle.car.truck.garbage bin.construction barrel.stop sign.
```

## Success Criteria For The First Jetson Session

Minimum success:

- Engine builds or loads.
- ROS graph starts.
- `/set_prompt` returns `success=True`.
- `/detections_output` publishes a `vision_msgs/msg/Detection2DArray`.

Better success:

- Indoor sample produces `chair`, `monitor`, `box`, or `small robot car`.
- Outdoor sample produces `fire hydrant`, `garbage bin`, `vehicle/car`, `construction barrel`, or `stop sign`.
- We record memory and latency with `tegrastats`.

Do not require realtime FPS on day one. First prove the official path can survive on the Orin Nano.

## Decision Tree

```text
preflight says Ubuntu 24.04/Jazzy-like
  -> try Isaac ROS 4.4 binary/container directly

preflight says JetPack 6.x/Ubuntu 22.04/Humble
  -> try compatible Isaac ROS container first
  -> if blocked by version/NITROS/Jazzy mismatch, keep our Humble wrapper and reuse official ONNX/TensorRT lessons

engine build OOM
  -> max power, close processes, add swap, retry once
  -> then consider smaller shape/rebuild

engine loads but detections empty
  -> verify prompt has periods
  -> publish multiple frames
  -> lower confidence threshold to 0.25
  -> test known sample frame before rosbag

detections work but too slow
  -> scheduled inference at 1 to 5 Hz
  -> use YOLO for continuous safety perception
  -> use GroundingDINO for open-vocabulary semantic query bursts
```
