# Isaac ROS Grounding DINO Local Test

Date: 2026-06-02

This note records the local proof-of-life test for NVIDIA Isaac ROS Grounding DINO. The goal was to check whether the official ROS 2 and TensorRT implementation can serve as a higher-fidelity reference path after our custom ultra-compressed Jetson TensorRT experiment ran but produced poor detections.

This test was run on the local workstation, not on the Jetson Orin Nano.

## Host Environment

- Host OS: Ubuntu 22.04
- Architecture: x86_64
- GPU: NVIDIA RTX 4090
- NVIDIA driver: `580.95.05`
- Docker GPU runtime: NVIDIA Container Toolkit
- Isaac ROS source branch: `release-4.4`
- ROS distro inside Isaac ROS container: Jazzy
- Persistent test container: `isaac_ros_gdino_test`
- Isaac ROS workspace: `/home/boyang/workspaces/isaac_ros-dev`

## Reproduction Steps

### 1. Prepare The Isaac ROS Workspace

The official Isaac ROS workspace was kept outside this repository:

```bash
mkdir -p /home/boyang/workspaces/isaac_ros-dev/src
cd /home/boyang/workspaces/isaac_ros-dev/src

git clone -b release-4.4 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone -b release-4.4 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection.git
git clone -b release-4.4 https://github.com/NVIDIA-ISAAC-ROS/isaac-ros-cli.git
```

The relevant source packages were:

```text
isaac_ros_object_detection/isaac_ros_grounding_dino
isaac_ros_object_detection/isaac_ros_grounding_dino_interfaces
isaac_ros_object_detection/isaac_ros_grounding_dino_models_install
```

### 2. Install And Initialize Isaac ROS CLI

The Isaac ROS CLI was built from source:

```bash
cd /home/boyang/workspaces/isaac_ros-dev/src/isaac-ros-cli
make build
sudo dpkg -i ../isaac-ros-cli_2.3.0-1_all.deb
sudo isaac-ros init docker --yes
```

Then the cached Isaac ROS Docker image was pulled with:

```bash
export ISAAC_ROS_WS=/home/boyang/workspaces/isaac_ros-dev
isaac-ros activate --verbose
```

The image used for the persistent test container was:

```text
cached_isaac_run_dev_image_local:latest
```

### 3. Start A Persistent Test Container

A persistent container was used so package installs and generated engines survived across commands:

```bash
docker run -dit --privileged --network host --ipc=host \
  --workdir /workspaces/isaac_ros-dev \
  -e ISAAC_ROS_PLATFORM=amd64 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ISAAC_ROS_WS=/workspaces/isaac_ros-dev \
  -v /home/boyang/workspaces/isaac_ros-dev:/workspaces/isaac_ros-dev \
  -v /etc/localtime:/etc/localtime:ro \
  --name isaac_ros_gdino_test --gpus all \
  --entrypoint /usr/local/bin/scripts/workspace-entrypoint.sh \
  cached_isaac_run_dev_image_local:latest /bin/bash
```

### 4. Install Isaac ROS Grounding DINO

Inside the container:

```bash
docker exec --user root isaac_ros_gdino_test bash -lc '
  set -e
  source /opt/ros/jazzy/setup.bash
  apt-get update
  apt-get install -y --no-install-recommends ros-jazzy-isaac-ros-grounding-dino
'
```

The binary package installed the Grounding DINO nodes plus the required Isaac ROS image processing, tensor processing, TensorRT, NITROS, CUDA, and VPI dependencies.

### 5. Download The Official Model And Build The TensorRT Plan

The model installer uses an interactive EULA prompt by default. For a non-interactive terminal run, set `ISAAC_ROS_ACCEPT_EULA=1`:

```bash
docker exec --user root isaac_ros_gdino_test bash -lc '
  set -e
  source /opt/ros/jazzy/setup.bash
  export ISAAC_ROS_WS=/workspaces/isaac_ros-dev
  export ISAAC_ROS_ACCEPT_EULA=1
  ros2 run isaac_ros_grounding_dino_models_install install_grounding_dino_models.sh
'
```

Generated artifacts:

```text
/home/boyang/workspaces/isaac_ros-dev/isaac_ros_assets/models/grounding_dino/grounding_dino_model.onnx
/home/boyang/workspaces/isaac_ros-dev/isaac_ros_assets/models/grounding_dino/grounding_dino_model.plan
```

Observed artifact sizes:

- ONNX: about 689 MB
- TensorRT plan: about 696 MB

The generated TensorRT engine used fixed shapes:

```text
inputs:          1x3x544x960
input_ids:       1x256
attention_mask:  1x256
position_ids:    1x256
token_type_ids:  1x256
text_token_mask: 1x256x256
pred_logits:     1x900x256
pred_boxes:      1x900x4
```

The `trtexec` engine self-test passed on the RTX 4090 host. The self-test reported roughly `53 qps` with random inputs on the local workstation. This is only an engine smoke test, not an end-to-end ROS latency result and not a Jetson result.

### 6. Launch The Official ROS Graph

The official standalone launch succeeded with:

```bash
docker exec --user root isaac_ros_gdino_test bash -lc '
  source /opt/ros/jazzy/setup.bash
  export ISAAC_ROS_WS=/workspaces/isaac_ros-dev
  ros2 launch isaac_ros_grounding_dino isaac_ros_grounding_dino.launch.py \
    input_image_width:=640 input_image_height:=480 \
    network_image_width:=960 network_image_height:=544 \
    model_file_path:=/workspaces/isaac_ros-dev/isaac_ros_assets/models/grounding_dino/grounding_dino_model.onnx \
    engine_file_path:=/workspaces/isaac_ros-dev/isaac_ros_assets/models/grounding_dino/grounding_dino_model.plan \
    confidence_threshold:=0.3
'
```

Observed ROS interfaces:

```text
image input:      /image
camera info:      /camera_info
prompt service:   /set_prompt
detection output: /detections_output
```

The detection output type is:

```text
vision_msgs/msg/Detection2DArray
```

### 7. Set A Runtime Prompt

The prompt service worked:

```bash
ros2 service call /set_prompt isaac_ros_grounding_dino_interfaces/srv/SetPrompt \
  "{prompt: \"box.monitor.toolbox.small robot car.chair.stop sign.\"}"
```

Important prompt format detail: Isaac ROS Grounding DINO expects classes to be separated by periods, not commas. For example:

```text
chair.box.stop sign.
```

The preprocessor logs confirmed that the prompt was tokenized and synced with the decoder.

## Test Results On Local Sample Frames

Sample frame outputs were saved outside this repository:

```text
/home/boyang/safeai/GroundingDINO/outputs/isaac_ros/results/
```

These generated overlays and JSON summaries are intentionally not committed to this repo.

### Indoor Sample Scan

Prompt:

```text
box.monitor.toolbox.small robot car.chair.stop sign.
```

Summary:

- Frames scanned: 11
- Total detections: 67
- `chair`: 12 detections, max score `0.869`
- `small robot car`: 11 detections, max score `0.843`
- `monitor`: 16 detections, max score `0.765`
- `box`: 16 detections, max score `0.559`
- `toolbox`: 9 detections, max score `0.404`
- `stop sign`: 3 detections, max score `0.322`

Qualitative note: indoor detections were much more usable than the custom `128x192` TensorRT compression experiment. `chair`, `monitor`, `box`, and `small robot car` were repeatedly detected. `box` and `toolbox` competed with each other, so prompt wording and thresholds matter.

### Outdoor Sample Scan

Prompt:

```text
fire hydrant.vehicle.car.truck.garbage bin.construction barrel.stop sign.
```

Summary:

- Frames scanned: 14
- Total detections: 68
- `fire hydrant`: 3 detections, max score `0.927`
- `garbage bin`: 18 detections, max score `0.699`
- `car`: 12 detections, max score `0.672`
- `vehicle`: 13 detections, max score `0.582`
- `construction barrel`: 11 detections, max score `0.549`
- `truck`: 8 detections, max score `0.454`
- `stop sign`: 3 detections, max score `0.443`

An early one-shot probe saw zero outdoor detections because it read a stale or transitional output immediately after a prompt switch. Repeating publication and taking the last detection message per frame produced stable results.

## Important Integration Notes

### Coordinate Space

The official standalone launch sets the decoder `image_width` and `image_height` to the network dimensions, not the original input image dimensions:

```text
network_image_width=960
network_image_height=544
```

For a 640x480 input image, `keep_aspect_ratio=True` resizes the image to fit inside the network canvas and pads it. The output boxes are therefore in the network/padded image coordinate system. A downstream ADAONE or NanoOWL-compatible wrapper must map detections back to the original camera frame before visualization or control logic.

### Open-Vocabulary Behavior

This official Isaac ROS path preserves runtime open-vocabulary prompt switching through `/set_prompt`. It is a better reference architecture than the custom ultra-compressed `128x192` TensorRT experiment because it keeps the official model structure and official decoder path.

### Jetson Status

This local test does not prove that the same pipeline will run on Jetson Orin Nano 8GB. It does prove that:

- the official model installer can generate the TensorRT plan;
- the official ROS graph can run end to end;
- dynamic prompts work;
- `vision_msgs/Detection2DArray` is already the right output type for our ROS integration.

The next Jetson experiment should try this official engine path first, then measure memory, load time, FPS, and whether the 696 MB engine plus runtime buffers fit on the Orin Nano.

## Relation To This Repository

This repository still contains the Humble-compatible custom ROS2 wrapper. Isaac ROS currently runs in a separate Jazzy-based container and should be treated as:

- an official reference implementation;
- a TensorRT/model preparation template;
- a candidate source for a future lightweight Humble bridge;
- a quality baseline against our own PyTorch and custom TensorRT paths.

Recommended next work:

1. Run the Isaac ROS graph against the full indoor/outdoor rosbags and save videos.
2. Measure end-to-end latency and GPU memory on the RTX 4090 host.
3. Attempt the same engine loading path on Jetson Orin Nano.
4. If full Isaac ROS Jazzy/NITROS is too heavy for ADAONE/Humble, reuse the official engine and prompt/decoder behavior in a lightweight Humble wrapper.
