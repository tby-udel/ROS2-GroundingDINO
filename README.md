# ROS2-GroundingDINO

This repository is a ROS 2 package for running a local GroundingDINO model as an open-vocabulary object detection node.

It is designed to match the topic style used by our local NanoOWL/ADAONE experiments, so GroundingDINO can be swapped into the same perception pipeline for rosbag replay, live camera input, and runtime query switching.

## What It Provides

- ROS 2 node: `ros2_groundingdino/groundingdino_node.py`
- ROS 2 executable: `groundingdino_py`
- Subscribes to runtime text queries on `input_query`
- Subscribes to camera frames on `input_image`
- Publishes structured detections on `output_detections`
- Optionally publishes annotated images on `output_image`
- Optionally publishes ADAONE/NanoOWL-compatible legacy outputs:
  - `/yolo/detections`
  - `/yolo/inference_image`
- Includes rosbag replay, output recording, query switching, and runtime profiling scripts under `tools/rosbag_replay/`

## Repository Layout

```text
.
├── launch/
│   ├── ada_reactive_perception.launch.py
│   ├── camera_input_example.launch.py
│   └── groundingdino_example.launch.py
├── ros2_groundingdino/
│   └── groundingdino_node.py
├── tools/
│   └── rosbag_replay/
└── docs/
    └── runtime_experiments_2026-05-27.md
```

## Requirements

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10 environment with:
  - `torch`
  - `torchvision`
  - `opencv-python`
  - `transformers`
  - `timm`
  - `addict`
  - `yapf`
  - `pycocotools`
- A local GroundingDINO checkout with model weights

Example local GroundingDINO paths used during development:

```bash
/home/boyang/safeai/GroundingDINO
/home/boyang/safeai/GroundingDINO/weights/groundingdino_swint_ogc.pth
```

## Build

Clone this package into a ROS 2 workspace:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone git@github.com:tby-udel/ROS2-GroundingDINO.git
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select ros2_groundingdino
source install/setup.bash
```

If you use a Conda environment for PyTorch/GroundingDINO, activate it before sourcing and building:

```bash
conda activate safeai
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select ros2_groundingdino
source install/setup.bash
```

## Run With A Camera Topic

```bash
ros2 launch ros2_groundingdino ada_reactive_perception.launch.py \
  groundingdino_dir:=/home/boyang/safeai/GroundingDINO \
  config:=/home/boyang/safeai/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  checkpoint:=/home/boyang/safeai/GroundingDINO/weights/groundingdino_swint_ogc.pth \
  input_image_topic:=/camera/camera/color/image_raw \
  initial_query:="box, monitor, chair, stop sign" \
  image_size:=480 \
  max_size:=640 \
  thresholds:=0.25 \
  text_threshold:=0.20 \
  publish_output_image:=true \
  publish_legacy_outputs:=true
```

For our 640x480 rosbag experiments, `image_size:=480 max_size:=640` was much faster than the default GroundingDINO resize while keeping the stream near 15 Hz on the local RTX 4090 workstation.

## Runtime Query Switching

The node supports open-vocabulary query changes while it is running:

```bash
ros2 topic pub --once /input_query std_msgs/msg/String \
  "{data: 'fire hydrant, vehicle, garbage bin, construction barrel, stop sign'}"
```

The node will update its prompt and class mapping without restarting.

## Topics

Subscriptions:

- `input_image` (`sensor_msgs/msg/Image`)
- `input_query` (`std_msgs/msg/String`)

Publishers:

- `output_detections` (`vision_msgs/msg/Detection2DArray`)
- `output_image` (`sensor_msgs/msg/Image`, optional)
- `/yolo/detections` (`std_msgs/msg/String`, optional compatibility output)
- `/yolo/inference_image` (`sensor_msgs/msg/Image`, optional compatibility output)

## Rosbag Replay And Profiling

The helper scripts live in:

```bash
tools/rosbag_replay/
```

Before using them, source ROS and the workspace that contains this package:

```bash
conda activate safeai
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

Example runtime query switching experiment:

```bash
ROS_DOMAIN_ID=81 bash tools/rosbag_replay/run_groundingdino_rosbag_experiment.sh \
  outdoor_query_switch \
  /home/boyang/safeai/converted_rosbags/outdoor_640x480_15hz \
  "stop sign, garbage bin" \
  "stop sign,garbage bin,fire hydrant,construction barrel,vehicle" \
  /home/boyang/safeai/GroundingDINO/outputs/runtime_query_switch/outdoor_query_switch \
  1.0 \
  0.25 \
  0.20 \
  cuda \
  true \
  true \
  "0|stop sign, garbage bin;14|fire hydrant, construction barrel;28|vehicle, stop sign" \
  0 \
  480 \
  640
```

Example profiling-only run:

```bash
ROS_DOMAIN_ID=82 bash tools/rosbag_replay/run_groundingdino_rosbag_experiment.sh \
  outdoor_profile_img480_1x \
  /home/boyang/safeai/converted_rosbags/outdoor_640x480_15hz \
  "fire hydrant, vehicle, garbage bin, construction barrel, stop sign" \
  "fire hydrant,vehicle,garbage bin,construction barrel,stop sign" \
  /home/boyang/safeai/GroundingDINO/outputs/runtime_profile/outdoor_profile_img480_1x \
  1.0 \
  0.25 \
  0.20 \
  cuda \
  false \
  false \
  "" \
  0 \
  480 \
  640
```

See [docs/runtime_experiments_2026-05-27.md](docs/runtime_experiments_2026-05-27.md) for the local query switching and runtime profiling results.

## Notes

This package wraps the PyTorch GroundingDINO implementation. It does not include model weights, TensorRT engines, build directories, rosbag files, or generated videos.
