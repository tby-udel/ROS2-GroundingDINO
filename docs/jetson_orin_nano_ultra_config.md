# Jetson Orin Nano Ultra Compression Config

This config is the "make it survive first" path for Jetson Orin Nano 8GB after PyTorch FP16 failed.

The key decision is: do not rely on PyTorch FP16 as the first rescue path. Keep the model in FP32, then reduce runtime cost aggressively through spatial compression, temporal downsampling, disabled annotated-image publishing, capped detections, and disabled checkpoint wrappers.

## ROS2 Ultra Launch

```bash
ros2 launch ros2_groundingdino jetson_orin_nano_ultra.launch.py \
  groundingdino_dir:=/home/ada2/GroundingDINO \
  config:=/home/ada2/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  checkpoint:=/home/ada2/GroundingDINO/weights/groundingdino_swint_ogc.pth \
  input_image_topic:=/camera/camera/color/image_raw \
  initial_query:="stop sign, garbage bin" \
  precision:=fp32
```

Default ultra parameters:

- `precision=fp32`
- `image_size=224`
- `max_size=320`
- `frame_stride=3`
- `max_detections=20`
- `thresholds=0.35`
- `text_threshold=0.25`
- `publish_output_image=false`
- `publish_legacy_outputs=true`
- `publish_legacy_image=false`
- `empty_cache_every_n_frames=8`
- `torch_num_threads=2`
- `disable_model_checkpointing=true`

This means a 15 Hz stream becomes about 5 attempted inference frames per second before model latency is counted.

## Less Aggressive Fallback

If ultra mode runs but detection quality is too damaged, try:

```bash
ros2 launch ros2_groundingdino jetson_orin_nano_ultra.launch.py \
  image_size:=320 \
  max_size:=480 \
  frame_stride:=2 \
  max_detections:=30 \
  thresholds:=0.30 \
  precision:=fp32
```

## Why This Is Stronger Than FP16-Only

FP16 reduces tensor precision, but it does not remove the main GroundingDINO costs:

- image backbone feature extraction
- multi-scale transformer attention
- text branch execution per frame in the PyTorch path
- postprocess and drawing overhead

The ultra config attacks the workload directly:

- smaller image tensors
- fewer processed frames
- fewer boxes carried through postprocess
- no annotated output image
- no legacy image drawing
- no checkpoint wrappers during inference

## TensorRT Open-Vocab Export Target

The TensorRT path separates BERT/tokenization from the exported detector graph. This preserves runtime open-vocabulary behavior by feeding encoded text into the engine.

Export the most compressed ONNX shape:

```bash
GROUNDINGDINO_DIR=/home/ada2/GroundingDINO \
tools/tensorrt/export_jetson_ultra_onnx.sh \
  outputs/tensorrt/groundingdino_open_vocab_jetson_ultra.onnx \
  jetson-ultra
```

The `jetson-ultra` ONNX preset uses:

- image: `1x3x224x320`
- encoded text length: `64`
- hidden dim: `256`

Build a TensorRT engine:

```bash
tools/tensorrt/build_jetson_ultra_engine.sh \
  outputs/tensorrt/groundingdino_open_vocab_jetson_ultra.onnx \
  outputs/tensorrt/groundingdino_open_vocab_jetson_ultra.engine \
  fp16
```

TensorRT FP16 is not the same thing as PyTorch FP16. PyTorch FP16 can fail from unsupported ops, memory behavior, or runtime numerics while a TensorRT FP16 engine may still build and run. If TensorRT FP16 also fails, build `fp32` first to validate the exported graph.

INT8 should only be used with calibration:

```bash
CALIB_CACHE=/path/to/calibration.cache \
tools/tensorrt/build_jetson_ultra_engine.sh \
  outputs/tensorrt/groundingdino_open_vocab_jetson_ultra.onnx \
  outputs/tensorrt/groundingdino_open_vocab_jetson_ultra_int8.engine \
  int8-calib
```

## Expected Tradeoff

Ultra mode is intentionally harsh. It should be treated as a survivability baseline, not the final quality target.

Recommended sequence:

1. Confirm `jetson_orin_nano_ultra.launch.py` can stay alive on the Jetson.
2. Measure FPS, latency, RAM, and GPU memory.
3. Move from `224/320 stride=3` toward `320/480 stride=2` if there is headroom.
4. Only then compare TensorRT FP32/FP16 engine attempts.
