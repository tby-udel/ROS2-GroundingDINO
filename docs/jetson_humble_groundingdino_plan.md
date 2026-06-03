# JetPack 6.2 / ROS Humble GroundingDINO Plan

Target environment:

```text
Jetson Orin Nano 8GB
JetPack 6.2
Ubuntu 22.04
ROS 2 Humble
CUDA 12.6
TensorRT 10.3
```

## What This Means

This is the right environment for ADAONE/Humble, but it is not the environment where NVIDIA Isaac ROS Grounding DINO worked locally.

The local Isaac ROS success used:

```text
Isaac ROS release-4.4
ROS 2 Jazzy
TensorRT 10.13
CUDA 13
```

Isaac ROS `release-3.2` is the JetPack 6.1/6.2 and Humble compatibility line, but it does not include `isaac_ros_grounding_dino`. Therefore the first Jetson test should not try to install a nonexistent official 3.2 GroundingDINO package.

## Recommended Test Order

### Stage 1: Prove GroundingDINO Can Run Under Humble

Use our existing ROS2 wrapper first. It is already Humble-compatible.

On the Jetson:

```bash
git clone git@github.com:tby-udel/ROS2-GroundingDINO.git
cd ROS2-GroundingDINO
git pull

bash tools/jetson_humble/jetson_humble_preflight.sh | tee jetson_humble_preflight.log
```

If preflight shows `torch.cuda.is_available: True`, build the package:

```bash
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select ros2_groundingdino
source install/setup.bash
```

Then run the most conservative smoke test:

```bash
cd ~/ros2_ws/src/ROS2-GroundingDINO
bash tools/jetson_humble/run_pytorch_ultra_smoke.sh
```

This launches:

```text
precision=fp32
image_size=224
max_size=320
frame_stride=3
max_detections=20
publish_output_image=false
publish_legacy_image=false
```

The goal is survival first, not accuracy.

### Stage 2: Verify ROS I/O

In another terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 topic list | sort
ros2 topic echo /output_detections --once
ros2 topic echo /yolo/detections --once
```

Change query at runtime:

```bash
ros2 topic pub --once /input_query std_msgs/msg/String \
  "{data: 'fire hydrant, vehicle, garbage bin, construction barrel, stop sign'}"
```

If detections publish, GroundingDINO is running on the Jetson/Humble stack.

### Stage 3: Replay A Short Rosbag Segment

Do not start with the full bag. First use a short segment or a low-rate replay.

Indoor prompt:

```text
box, monitor, toolbox, small robot car, chair, stop sign
```

Outdoor prompt:

```text
fire hydrant, vehicle, garbage bin, construction barrel, stop sign
```

If the node survives:

```bash
tegrastats
```

Record:

- startup time
- GPU memory peak
- RAM peak
- effective processed FPS
- whether detections are usable

## If It Fails

### PyTorch CUDA Missing

Fix PyTorch first. Do not debug GroundingDINO until this works:

```bash
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

### Model OOM During Load

Try:

```bash
sudo nvpmodel -m 0 || true
sudo jetson_clocks || true
sudo tegrastats
```

Close camera viewers, browsers, old Docker containers, and previous ROS nodes. Keep `precision=fp32`; PyTorch FP16 already failed in earlier Jetson experiments.

### Model Loads But Inference Is Too Slow

This is expected. GroundingDINO is heavy for Orin Nano.

Use it as scheduled open-vocabulary perception:

```text
YOLO: continuous closed-set perception at high rate
NanoOWL: lightweight open-vocabulary fallback
GroundingDINO: low-rate semantic query bursts, about 1-5 Hz if possible
```

### Detection Quality Is Bad

Do not reduce below `224x320` immediately. Try:

```text
image_size=320
max_size=480
frame_stride=3 or 4
thresholds=0.30
text_threshold=0.20
```

If memory allows, `320x480` is a much better quality test than `128x192`.

## TensorRT Path For JetPack 6.2

Because Isaac ROS 3.2 does not include GroundingDINO, the TensorRT route on JetPack 6.2 should be independent of the Isaac ROS graph:

```text
GroundingDINO / official ONNX
        -> TensorRT engine built on Jetson TensorRT 10.3
        -> lightweight Humble node or Python runner
        -> publish vision_msgs/Detection2DArray and /yolo/detections
```

Do not reuse the x86 RTX 4090 TensorRT `.plan`. Build on the Jetson.

First TensorRT test should be engine-only:

```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

If full official ONNX does not build or OOMs, use the repo's reduced ONNX exporter as an experiment, but remember that the previous `128x192` / `2+2 transformer layer` compression destroyed detection quality.

## Success Definition

Minimum success:

```text
PyTorch CUDA available
ROS Humble package builds
GroundingDINO model loads
node subscribes to images
node publishes output_detections or /yolo/detections
```

Useful success:

```text
short rosbag segment runs without OOM
latency/memory measured with tegrastats
detections appear for at least one indoor and one outdoor target class
```

Realtime is not required for the first Jetson session.
