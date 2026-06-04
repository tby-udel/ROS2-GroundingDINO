# GroundingDINO Jetson Nano to AGX Handoff

Date captured: 2026-06-04

This file records the current state of the GroundingDINO Jetson work before moving experiments to the new Jetson AGX. The goal is to reproduce the Nano-side settings as closely as possible, build the NVIDIA GroundingDINO TensorRT engine on the larger AGX, and then decide whether to run on AGX directly or copy the engine back to the smaller Jetson.

## Current Goal

Make an open-vocabulary GroundingDINO ROS 2 pipeline run with usable quality on Jetson hardware.

The most promising path is NVIDIA Isaac ROS GroundingDINO with the official TAO ONNX model and a TensorRT engine. The custom ultra-compressed model builds and runs on the small Jetson, but its detection quality was not usable.

## Nano Host Snapshot

This is the current small Jetson host, referred to during the project as the Jetson Orin Nano.

System-reported hardware:

- Hostname: `ubuntu`
- Hardware vendor: `NVIDIA`
- Hardware model from `hostnamectl`: `NVIDIA Jetson Orin NX Engineering Reference Developer Kit`
- Device tree model: `NVIDIA Jetson Orin NX Engineering Reference Developer Kit`
- Architecture: `arm64`
- CPU: 6 x `Cortex-A78AE`
- RAM: about `7.4 GiB` total, `3.7 GiB` swap
- Root storage: `/dev/nvme0n1p1`, `233G` total, `149G` used, `72G` available at capture time

OS and JetPack/L4T:

- Ubuntu: `22.04.5 LTS`, Jammy
- Kernel: `5.15.148-tegra`
- L4T: `R36.4.7`
- `/etc/nv_tegra_release`: `# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025`
- `nvidia-l4t-core`: `36.4.7-20250918154033`

CUDA/TensorRT/Python:

- CUDA compiler: `12.6`, `V12.6.68`
- TensorRT CLI header: `TensorRT v100300`
- TensorRT Debian packages:
  - `tensorrt 10.3.0.30-1+cuda12.5`
  - `libnvinfer10 10.3.0.30-1+cuda12.5`
  - `libnvinfer-plugin10 10.3.0.30-1+cuda12.5`
  - `python3-libnvinfer 10.3.0.30-1+cuda12.5`
- Python TensorRT: `10.3.0`
- Python torch: `2.8.0`, CUDA available: `True`

Thermal/load sample from `tegrastats` during capture:

```text
RAM 5791/7620MB, SWAP 14/3810MB, GR3D_FREQ 65%, GPU about 46 C, VDD_IN about 7 W
```

Important memory observation:

- The small Jetson has only about 8 GB unified memory.
- TensorRT engine building for full NVIDIA GroundingDINO fails locally because the builder cannot allocate enough GPU memory during tactic/build allocation.

## Docker State

Images available on the Nano:

```text
isaac_ros_dev-aarch64:nanoowl-ready        image id 80640436aefd   size 42.5GB
bytian/nanoowl-adaone:latest               image id 80640436aefd   size 42.5GB
isaac_ros_dev-aarch64:latest               image id dd032d9aa0a4   size 40.3GB
nvcr.io/nvidia/isaac/ros:aarch64-ros2_humble_4c0c55dddd2bbcc3e8d5f9753bee634c
nvidia/cuda:12.6.0-base-ubuntu22.04
ros:humble-ros-core-jammy
ubuntu:22.04
```

Containers at capture time:

```text
groundingdino_backport_humble     Exited (137)    isaac_ros_dev-aarch64:nanoowl-ready
nanoowl-eval-container            Exited (143)    isaac_ros_dev-aarch64:nanoowl-ready
isaac_ros_dev-aarch64-container   Exited (143)    isaac_ros_dev-aarch64:latest
```

Useful start command:

```bash
docker start groundingdino_backport_humble nanoowl-eval-container isaac_ros_dev-aarch64-container
```

The main container used for the Humble backport work was:

```text
groundingdino_backport_humble
image: isaac_ros_dev-aarch64:nanoowl-ready
```

## Repository State

Custom ROS 2 GroundingDINO repo:

```text
path: /home/ada2/boyang_ws/src/ROS2-GroundingDINO
remote: https://github.com/tby-udel/ROS2-GroundingDINO.git
branch: main
status: clean relative to origin/main at capture time
latest commit: 9a5caef Add JetPack 6.2 Humble GroundingDINO plan
```

NVIDIA Isaac ROS object detection clone used for backport:

```text
path: /home/ada2/boyang_ws/backport_groundingdino_humble_ws/src/isaac_ros_object_detection
remote: https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection.git
branch: release-4.4
latest upstream commit: 4c4a187 Isaac ROS 4.4
status: local Humble backport edits are present and uncommitted
```

Backport files modified:

```text
isaac_ros_grounding_dino/CMakeLists.txt
isaac_ros_grounding_dino/package.xml
isaac_ros_grounding_dino/src/grounding_dino_decoder_node.cpp
isaac_ros_grounding_dino/src/grounding_dino_preprocessor_node.cpp
```

Backport changes made:

- Removed the 4.x model installer package dependency from `package.xml`.
- Added RPATH handling for the decoder target to match the preprocessor target.
- Replaced 4.x-only `isaac_ros_common/cuda_stream.hpp` usage with direct CUDA stream creation.
- Replaced Jazzy `rclcpp::ServicesQoS()` usage with Humble-compatible `rmw_qos_profile_services_default`.
- Installed missing Humble dependencies in the container, including `transformers`.

Backport result:

- `isaac_ros_grounding_dino_interfaces` builds on Humble.
- `isaac_ros_grounding_dino` builds on Humble.
- Components are visible and instantiate:
  - `GroundingDinoPreprocessorNode`
  - `GroundingDinoDecoderNode`
- Tokenizer smoke test worked for prompt `box. monitor. chair.`

## Official NVIDIA GroundingDINO Model

Official TAO ONNX downloaded after user-authorized EULA acceptance:

```text
path: /home/ada2/boyang_ws/backport_groundingdino_humble_ws/isaac_ros_assets/models/grounding_dino/grounding_dino_model.onnx
size: 721823941 bytes, about 689 MiB
sha256: 6895acdc6b588e923f753e37b3bd18869e064256e5ecc1b2b9853e8c51125f94
```

NGC source URL used:

```text
https://api.ngc.nvidia.com/v2/models/nvidia/tao/grounding_dino/versions/grounding_dino_swin_tiny_commercial_deployable_v1.0/files/grounding_dino_swin_tiny_commercial_deployable.onnx
```

Official ONNX input/output signature:

```text
inputs:
  inputs: FLOAT [batch_size, 3, 544, 960]
  input_ids: INT64 [batch_size, 256]
  attention_mask: BOOL [batch_size, 256]
  position_ids: INT64 [batch_size, 256]
  token_type_ids: INT64 [batch_size, 256]
  text_token_mask: BOOL [batch_size, 256, 256]

outputs:
  pred_logits: FLOAT [batch_size, Gatherpred_logits_dim_1, Gatherpred_logits_dim_2]
  pred_boxes: FLOAT [batch_size, Gatherpred_boxes_dim_1, Gatherpred_boxes_dim_2]
```

This matches the NVIDIA Isaac ROS 4.x GroundingDINO launch graph bindings. This is the best current candidate for a correct open-vocabulary ROS pipeline.

## TensorRT Attempts On The Small Jetson

### Official NVIDIA ONNX, FP16

Command shape profile:

```text
inputs:1x3x544x960
input_ids:1x256
attention_mask:1x256
position_ids:1x256
token_type_ids:1x256
text_token_mask:1x256x256
```

Representative command:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/ada2/boyang_ws/backport_groundingdino_humble_ws/isaac_ros_assets/models/grounding_dino/grounding_dino_model.onnx \
  --saveEngine=/home/ada2/boyang_ws/src/ROS2-GroundingDINO/artifacts/tensorrt/nvidia_tao_grounding_dino_544x960_text256_fp16_trt103.engine \
  --fp16 \
  --memPoolSize=workspace:512 \
  --builderOptimizationLevel=0 \
  --avgTiming=1 \
  --maxAuxStreams=0 \
  --minShapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
  --optShapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
  --maxShapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
  --skipInference
```

Result:

- ONNX parsing succeeded.
- TensorRT found/created `MultiscaleDeformableAttnPlugin_TRT`.
- Engine build failed during tactic/build allocation due to GPU memory pressure.
- No engine was produced.

### Official NVIDIA ONNX, Weight Streaming

Command:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/ada2/boyang_ws/backport_groundingdino_humble_ws/isaac_ros_assets/models/grounding_dino/grounding_dino_model.onnx \
  --saveEngine=/home/ada2/boyang_ws/src/ROS2-GroundingDINO/artifacts/tensorrt/nvidia_tao_grounding_dino_544x960_text256_weightstream_trt103.engine \
  --stronglyTyped \
  --allowWeightStreaming \
  --weightStreamingBudget=0% \
  --memPoolSize=workspace:512 \
  --builderOptimizationLevel=0 \
  --avgTiming=1 \
  --maxAuxStreams=0 \
  --minShapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
  --optShapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
  --maxShapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
  --skipInference
```

Result:

- ONNX parsing succeeded.
- TensorRT plugin creation succeeded.
- Build failed with CUDA out-of-memory.
- Final observed failing request: about `702996352` bytes.
- No official NVIDIA TensorRT engine was produced on the small Jetson.

Conclusion:

- The official NVIDIA model is compatible with the installed TensorRT parser/plugins.
- The blocker is local memory capacity during engine build, not model download, EULA, bindings, or missing plugins.

## Custom Ultra-Compressed Engine

The custom repo also contains a smaller open-vocab export path. This is not the same input contract as the NVIDIA Isaac ROS graph.

Local custom ONNX inputs:

```text
image
encoded_text
text_token_mask
position_ids
text_self_attention_masks
```

Local custom outputs:

```text
pred_logits
pred_boxes
```

Successful TensorRT engine:

```text
path: /home/ada2/boyang_ws/src/ROS2-GroundingDINO/artifacts/tensorrt/groundingdino_swint_open_vocab_128x192_text32_q100_e2d2_fp16_trt103_fast_avg1.engine
size: 87007108 bytes, about 83 MiB
sha256: 20b8ba1dc1f3588c885b855dc1027cb61d3252c78a076a79213886b44371d842
```

Successful build command:

```bash
trtexec \
  --onnx=groundingdino_swint_open_vocab_128x192_text32_q100_e2d2_fp16.onnx \
  --saveEngine=groundingdino_swint_open_vocab_128x192_text32_q100_e2d2_fp16_trt103_fast_avg1.engine \
  --fp16 \
  --memPoolSize=workspace:512 \
  --builderOptimizationLevel=0 \
  --avgTiming=1 \
  --skipInference
```

Runtime smoke result:

- Engine loaded.
- Random-input inference passed.
- Approximate throughput: `15.1 qps`
- Mean latency: about `65.5 ms`
- p95 latency: about `72.2 ms`

Quality result:

- Tested with rosbag `testing_rosbags/indoor_640x480_15hz`.
- Prompt/query list:
  - `Box`
  - `monitor`
  - `toolbox`
  - `small robot car`
  - `chair`
  - `stop sign`
- Output was not usable: detections were mostly wrong.
- This path proves TensorRT viability for a tiny model, but not acceptable detection quality.

Other custom ONNX artifacts present:

```text
groundingdino_swint_open_vocab_128x192_text32_q100_e2d2_fp16.engine
groundingdino_swint_open_vocab_128x192_text32_q100_e2d2_fp16.onnx
groundingdino_swint_open_vocab_128x192_text32_q100_e2d2_fp16_trt103_fast_avg1.engine
groundingdino_swint_open_vocab_128x192_text32_q100_e2d2.onnx
groundingdino_swint_open_vocab_128x192_text32_q100_fp16.onnx
groundingdino_swint_open_vocab_128x192_text32_q100.onnx
groundingdino_swint_open_vocab_160x256_text64_q300.onnx
groundingdino_swint_open_vocab_224x320_text64.onnx
```

Larger custom TensorRT attempts:

- `224x320_text64_q900`: parsed, then hit repeated GPU OOM during tactic selection.
- `160x256_text64_q300`: parsed, but was memory-tight/slow and was stopped intentionally.

## What To Do On The Jetson AGX

The AGX has more memory, so the first experiment should be to build the official NVIDIA TensorRT engine without changing the graph.

### 1. Match The Nano Software Stack As Closely As Possible

Preferred AGX target:

```text
Ubuntu 22.04 / Jammy
JetPack/L4T R36.4.7 if available
CUDA 12.6
TensorRT 10.3.0
ROS 2 Humble for our backport container path
```

TensorRT engines are target-stack-sensitive. Build on the exact machine that will run the engine whenever possible. If building on AGX and copying back to the smaller Jetson, keep JetPack, TensorRT, CUDA, OS architecture, and plugin availability as close as possible.

### 2. Recreate Or Copy The Workspace

Fastest approach:

```bash
rsync -a /home/ada2/boyang_ws/src/ROS2-GroundingDINO <agx_user>@<agx_host>:~/boyang_ws/src/
rsync -a /home/ada2/boyang_ws/backport_groundingdino_humble_ws <agx_user>@<agx_host>:~/boyang_ws/
```

Alternative:

```bash
mkdir -p ~/boyang_ws/src
cd ~/boyang_ws/src
git clone https://github.com/tby-udel/ROS2-GroundingDINO.git
```

Then copy or re-apply the Humble backport workspace from:

```text
/home/ada2/boyang_ws/backport_groundingdino_humble_ws
```

Do not lose the local edits in NVIDIA's `isaac_ros_object_detection` clone unless we first save them as a patch.

### 3. Put The Official ONNX In Place

Use the already-downloaded ONNX:

```bash
mkdir -p ~/boyang_ws/backport_groundingdino_humble_ws/isaac_ros_assets/models/grounding_dino
scp ada2@<nano_host>:/home/ada2/boyang_ws/backport_groundingdino_humble_ws/isaac_ros_assets/models/grounding_dino/grounding_dino_model.onnx \
  ~/boyang_ws/backport_groundingdino_humble_ws/isaac_ros_assets/models/grounding_dino/
sha256sum ~/boyang_ws/backport_groundingdino_humble_ws/isaac_ros_assets/models/grounding_dino/grounding_dino_model.onnx
```

Expected SHA256:

```text
6895acdc6b588e923f753e37b3bd18869e064256e5ecc1b2b9853e8c51125f94
```

### 4. Build The Official TensorRT Engine On AGX

Start with the same conservative command that failed on Nano:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=$HOME/boyang_ws/backport_groundingdino_humble_ws/isaac_ros_assets/models/grounding_dino/grounding_dino_model.onnx \
  --saveEngine=$HOME/boyang_ws/backport_groundingdino_humble_ws/isaac_ros_assets/models/grounding_dino/grounding_dino_model.plan \
  --fp16 \
  --memPoolSize=workspace:512 \
  --builderOptimizationLevel=0 \
  --avgTiming=1 \
  --maxAuxStreams=0 \
  --minShapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
  --optShapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
  --maxShapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
  --skipInference
```

If AGX has enough memory, try a stronger build after the conservative engine succeeds:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=$HOME/boyang_ws/backport_groundingdino_humble_ws/isaac_ros_assets/models/grounding_dino/grounding_dino_model.onnx \
  --saveEngine=$HOME/boyang_ws/backport_groundingdino_humble_ws/isaac_ros_assets/models/grounding_dino/grounding_dino_model_opt.plan \
  --fp16 \
  --memPoolSize=workspace:2048 \
  --builderOptimizationLevel=3 \
  --avgTiming=4 \
  --minShapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
  --optShapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
  --maxShapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
  --skipInference
```

Do not start with FP8. Jetson Orin-class hardware is not the right FP8 target. The practical first target is FP16. The practical second target is INT8, but only after the FP16 official graph works and we have a calibration/evaluation plan.

### 5. Smoke Test Engine Loading

After engine creation:

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=$HOME/boyang_ws/backport_groundingdino_humble_ws/isaac_ros_assets/models/grounding_dino/grounding_dino_model.plan \
  --shapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
  --warmUp=500 \
  --duration=10
```

Record:

- Engine file size
- SHA256
- Mean latency
- p90/p95 latency
- GPU memory use from `tegrastats`
- Whether deserialization succeeds inside the ROS container

### 6. Build Or Run The Humble Backport On AGX

Inside the Humble/Isaac ROS container, rebuild the backport:

```bash
cd /home/ada2/boyang_ws/backport_groundingdino_humble_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select \
  isaac_ros_grounding_dino_interfaces \
  isaac_ros_grounding_dino
source install/setup.bash
```

Then launch with the generated plan file. The helper in this repo is currently written for NVIDIA's Jazzy 4.x path, so for the Humble backport use the same launch arguments but source Humble:

```bash
source /opt/ros/humble/setup.bash
source /home/ada2/boyang_ws/backport_groundingdino_humble_ws/install/setup.bash

ros2 launch isaac_ros_grounding_dino isaac_ros_grounding_dino.launch.py \
  input_image_width:=640 \
  input_image_height:=480 \
  network_image_width:=960 \
  network_image_height:=544 \
  model_file_path:=/home/ada2/boyang_ws/backport_groundingdino_humble_ws/isaac_ros_assets/models/grounding_dino/grounding_dino_model.onnx \
  engine_file_path:=/home/ada2/boyang_ws/backport_groundingdino_humble_ws/isaac_ros_assets/models/grounding_dino/grounding_dino_model.plan \
  confidence_threshold:=0.3 \
  force_engine_update:=False
```

### 7. Run The Same Rosbag Test

Use the same open-vocab test query as NanoOWL/GroundingDINO comparison:

```text
Box, monitor, toolbox, small robot car, chair, stop sign
```

Rosbag:

```text
testing_rosbags/indoor_640x480_15hz
```

Save a quick-check video for direct visual comparison. Use the custom repo's rosbag/video tools as reference:

```text
/home/ada2/boyang_ws/src/ROS2-GroundingDINO/tools/tensorrt/run_trt_rosbag_video.py
/home/ada2/boyang_ws/src/ROS2-GroundingDINO/tools/rosbag_replay/
```

### 8. Copy Back To The Small Jetson Only After AGX Success

If the AGX-built engine runs correctly on AGX, try copying the engine back to the small Jetson only if the JetPack/TensorRT stack is closely matched:

```bash
scp <agx_user>@<agx_host>:~/boyang_ws/backport_groundingdino_humble_ws/isaac_ros_assets/models/grounding_dino/grounding_dino_model.plan \
  /home/ada2/boyang_ws/backport_groundingdino_humble_ws/isaac_ros_assets/models/grounding_dino/
```

Then smoke test deserialization on the small Jetson before launching the full ROS graph:

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=/home/ada2/boyang_ws/backport_groundingdino_humble_ws/isaac_ros_assets/models/grounding_dino/grounding_dino_model.plan \
  --shapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
  --duration=10
```

If deserialization fails, rebuild the engine on the actual small Jetson target is still required. If runtime memory is too high, the official full model is not deployable on the small Jetson without a smaller ONNX export or INT8/calibration route.

## Decision Points

If official FP16 engine builds and runs on AGX:

- Use it as the correctness baseline.
- Run the rosbag video test.
- Compare quality against NanoOWL and the custom compressed model.
- Try copying the engine to the small Jetson only after confirming stack compatibility.

If official FP16 engine builds on AGX but is too slow:

- Keep it as the quality baseline.
- Try TensorRT builder optimization variants.
- Then consider INT8 calibration.

If official FP16 engine does not build on AGX:

- Stop trying to build the full official ONNX on Jetson-class hardware.
- Look for or create a smaller export that preserves NVIDIA's raw text-token inputs:
  - smaller image size than `544x960`
  - shorter text length than `256`
  - fewer queries/proposals
- Keep open-vocab behavior by preserving runtime text token inputs. Do not use frozen prompt export if dynamic vocabulary switching is a requirement.

If the custom ultra-compressed engine is revisited:

- Treat it as a speed experiment, not a quality baseline.
- The previous q100/e2d2/text32/128x192 result was too degraded for the indoor rosbag.

## Short Summary

Current best evidence:

- Backporting NVIDIA's ROS graph to Humble is feasible and mostly done.
- NVIDIA's official ONNX is present and has the correct bindings.
- TensorRT 10.3 on the small Jetson can parse the model and find the required plugin.
- The small Jetson cannot build the full official engine due to memory limits.
- The AGX should first be used to build the official FP16 TensorRT plan with the exact `544x960`, text-256 profile.
- After AGX success, use the official engine as the quality baseline, then decide whether Nano deployment needs engine transfer, INT8, or a smaller export.
