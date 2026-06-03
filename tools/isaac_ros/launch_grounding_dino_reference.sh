#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/jazzy/setup.bash

ISAAC_ROS_WS="${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}"
INPUT_IMAGE_WIDTH="${INPUT_IMAGE_WIDTH:-640}"
INPUT_IMAGE_HEIGHT="${INPUT_IMAGE_HEIGHT:-480}"
NETWORK_IMAGE_WIDTH="${NETWORK_IMAGE_WIDTH:-960}"
NETWORK_IMAGE_HEIGHT="${NETWORK_IMAGE_HEIGHT:-544}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.3}"
FORCE_ENGINE_UPDATE="${FORCE_ENGINE_UPDATE:-False}"

MODEL_FILE_PATH="${MODEL_FILE_PATH:-${ISAAC_ROS_WS}/isaac_ros_assets/models/grounding_dino/grounding_dino_model.onnx}"
ENGINE_FILE_PATH="${ENGINE_FILE_PATH:-${ISAAC_ROS_WS}/isaac_ros_assets/models/grounding_dino/grounding_dino_model.plan}"

if [ ! -f "$MODEL_FILE_PATH" ]; then
  echo "Missing ONNX model: $MODEL_FILE_PATH" >&2
  echo "Run the Isaac ROS model installer first." >&2
  exit 2
fi

if [ ! -f "$ENGINE_FILE_PATH" ]; then
  echo "Missing TensorRT plan: $ENGINE_FILE_PATH" >&2
  echo "Set FORCE_ENGINE_UPDATE=True to let TensorRTNode build from ONNX, or run the model installer." >&2
fi

exec ros2 launch isaac_ros_grounding_dino isaac_ros_grounding_dino.launch.py \
  input_image_width:="$INPUT_IMAGE_WIDTH" \
  input_image_height:="$INPUT_IMAGE_HEIGHT" \
  network_image_width:="$NETWORK_IMAGE_WIDTH" \
  network_image_height:="$NETWORK_IMAGE_HEIGHT" \
  model_file_path:="$MODEL_FILE_PATH" \
  engine_file_path:="$ENGINE_FILE_PATH" \
  confidence_threshold:="$CONFIDENCE_THRESHOLD" \
  force_engine_update:="$FORCE_ENGINE_UPDATE"
