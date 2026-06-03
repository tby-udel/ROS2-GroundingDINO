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
- Includes a Jetson Orin Nano ultra-compression launch preset after PyTorch FP16 proved unstable on-device
- Documents an NVIDIA Isaac ROS Grounding DINO TensorRT proof-of-life baseline for comparison with our custom wrapper and Jetson compression experiments
- Includes Isaac ROS Jetson preflight and launch helpers under `tools/isaac_ros/`
- Includes JetPack 6.2 / ROS Humble GroundingDINO smoke-test helpers under `tools/jetson_humble/`

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
│   ├── rosbag_replay/
│   ├── isaac_ros/
│   ├── jetson_humble/
│   └── tensorrt/
└── docs/
    ├── isaac_ros_grounding_dino_local_test_2026-06-02.md
    ├── jetson_humble_groundingdino_plan.md
    ├── jetson_isaac_ros_fast_path.md
    ├── jetson_orin_nano_ultra_config.md
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

## Jetson Orin Nano Ultra Mode

After PyTorch FP16 failed on Jetson Orin Nano, the strongest survival preset keeps FP32 and compresses the workload instead:

```bash
ros2 launch ros2_groundingdino jetson_orin_nano_ultra.launch.py \
  groundingdino_dir:=/home/ada2/GroundingDINO \
  config:=/home/ada2/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  checkpoint:=/home/ada2/GroundingDINO/weights/groundingdino_swint_ogc.pth \
  input_image_topic:=/camera/camera/color/image_raw \
  initial_query:="stop sign, garbage bin" \
  precision:=fp32
```

Ultra defaults:

- `image_size=224`
- `max_size=320`
- `frame_stride=3`
- `max_detections=20`
- `publish_output_image=false`
- `publish_legacy_outputs=true`
- `publish_legacy_image=false`
- `empty_cache_every_n_frames=8`
- `torch_num_threads=2`

See [docs/jetson_orin_nano_ultra_config.md](docs/jetson_orin_nano_ultra_config.md) for the full Jetson rescue path and TensorRT export target.

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

## Isaac ROS Grounding DINO Baseline

We also tested NVIDIA Isaac ROS Grounding DINO as the official ROS 2 and TensorRT reference path. On the local x86 workstation with an RTX 4090, the official installer successfully downloaded NVIDIA's Grounding DINO Swin-Tiny ONNX model and generated a TensorRT plan. The official ROS graph launched, accepted runtime prompts through `/set_prompt`, and published `vision_msgs/msg/Detection2DArray` on `/detections_output`.

Key local results:

- TensorRT plan size: about `696 MB`
- Engine image shape: `1x3x544x960`
- Text length: `256`
- Indoor sample scan: 11 frames, 67 detections
- Outdoor sample scan: 14 frames, 68 detections
- Dynamic prompt format: period-separated classes, for example `chair.box.stop sign.`

This is not a Jetson Orin Nano result yet. It is a quality-preserving baseline and integration template to compare against our custom PyTorch wrapper and the much smaller, lower-quality Jetson TensorRT compression experiment below.

See [docs/isaac_ros_grounding_dino_local_test_2026-06-02.md](docs/isaac_ros_grounding_dino_local_test_2026-06-02.md) for the full local build, model install, launch, prompt, and sample-frame test notes.

For the next Jetson Orin Nano deployment attempt, start with:

```bash
bash tools/isaac_ros/jetson_preflight.sh
```

Then follow [docs/jetson_isaac_ros_fast_path.md](docs/jetson_isaac_ros_fast_path.md). The short version is: build the TensorRT plan on the Jetson itself, prove `/set_prompt` and `/detections_output` on a single frame first, then move to rosbag replay.

Important compatibility note: Isaac ROS `release-3.2` matches JetPack 6.1/6.2 and ROS 2 Humble, but it does not include `isaac_ros_grounding_dino`. GroundingDINO appears in the Isaac ROS 4.x object detection stack, while Isaac ROS 4.0 moved to the JetPack 7 / Ubuntu 24.04 / CUDA 13 line. On a JetPack 6.2 Orin Nano, the official 3.2 package route is therefore not a GroundingDINO route.

For a confirmed JetPack 6.2 / Ubuntu 22.04 / ROS Humble Jetson, start with the native Humble wrapper path instead:

```bash
bash tools/jetson_humble/jetson_humble_preflight.sh
bash tools/jetson_humble/run_pytorch_ultra_smoke.sh
```

See [docs/jetson_humble_groundingdino_plan.md](docs/jetson_humble_groundingdino_plan.md).

## Jetson TensorRT Compression Findings

We tested an aggressively compressed open-vocabulary TensorRT path on a Jetson Orin Nano using the local NanoOWL Docker environment. The goal was to see whether GroundingDINO could be compressed enough to run directly on the small Jetson while preserving open-vocabulary prompts.

The tested engine kept the text encoder outside TensorRT, so the prompt remained open-vocabulary at runtime. The TensorRT engine received image tensors plus encoded text tensors:

```text
image: 1x3x128x192 FP16
encoded_text: 1x32x256 FP16
text_token_mask: 1x32 BOOL
position_ids: 1x32 INT64
text_self_attention_masks: 1x32x32 BOOL
outputs: pred_logits 1x100x32, pred_boxes 1x100x4
```

The compression settings were:

- Input resolution: `128x192`
- Max text length: `32`
- Object queries: `100`
- Transformer depth: `2` encoder layers and `2` decoder layers, down from the original `6/6`
- Precision: FP16 ONNX converted to a TensorRT engine

These settings are captured in the exporter as `--preset jetson-tiny-e2d2`. This preset is useful for reproducing the failed experiment, not as a recommended deployment target.

The engine built and executed successfully on the Jetson Orin Nano. A `trtexec` smoke test reported about `66 ms` GPU latency, roughly `15 FPS` engine-only throughput.

### Rosbag Test Result

We rendered the local rosbag:

```text
testing_rosbags/indoor_640x480_15hz
```

with the query:

```text
Box, monitor, toolbox, small robot car, chair, stop sign
```

using:

```bash
source /opt/ros/humble/setup.bash
python3 tools/tensorrt/run_trt_rosbag_video.py \
  --query "Box, monitor, toolbox, small robot car, chair, stop sign" \
  --bag /workspaces/isaac_ros-dev/testing_rosbags/indoor_640x480_15hz \
  --topic /camera/camera/color/image_raw \
  --box-threshold 0.7 \
  --text-threshold 0.5 \
  --max-detections 15 \
  --output artifacts/rosbag_tests/indoor_640x480_15hz_groundingdino_trt_e2d2_fp16_box070_text050.mp4
```

Observed runtime for the full 466-frame bag:

- Video size: `640x480`, `15 FPS`
- Mean TensorRT inference time: `65.7 ms/frame`
- End-to-end script throughput: about `9.1 FPS`, including rosbag decode, postprocessing, annotation, and MP4 writing
- Total displayed detections: `3500`

The model output was not usable. The detections were mostly incorrect, with many false positives and repeated `monitor` / `chair` predictions. Lower thresholds flooded every frame with detections, while stricter thresholds reduced clutter but still did not recover correct object grounding.

### Conclusion

This compressed engine proves that a small open-vocabulary GroundingDINO-style TensorRT graph can run on the Jetson Orin Nano, but this specific compression recipe destroys detection quality. It should not be used as a deployable detector.

The likely causes are:

- The `128x192` input resolution removes too much visual detail for small indoor objects.
- Truncating the transformer from `6/6` layers to `2/2` layers without retraining or distillation loses the grounding behavior.
- The output score distribution becomes poorly calibrated, so thresholds no longer behave like the original model.

### Recommended Next Steps

Do not compress further from this engine. The next work should preserve accuracy first, then optimize.

1. Run a golden baseline on sampled frames from the same rosbag using the original PyTorch GroundingDINO model.
2. Export a higher-fidelity FP16 TensorRT engine with the full `6/6` transformer depth.
3. Reduce image size gradually, for example `224x320` or `256x384`, and compare against the PyTorch baseline frame by frame.
4. Only after FP16 parity is acceptable, try INT8 post-training quantization with calibration images sampled from representative indoor rosbags.
5. Avoid FP8 as a primary target on Jetson Orin Nano. The Nano does not provide the same FP8 acceleration path as newer data-center GPUs.
6. If full-depth GroundingDINO remains too heavy, use GroundingDINO as an offline teacher for distillation or switch the runtime detector back toward NanoOWL/OWL-ViT-style models.

The practical next artifact should be an accuracy audit: side-by-side outputs from PyTorch GroundingDINO, ONNX, and TensorRT on identical sampled rosbag frames.

## Notes

This package wraps the PyTorch GroundingDINO implementation. It does not include model weights, TensorRT engines, build directories, rosbag files, or generated videos.
