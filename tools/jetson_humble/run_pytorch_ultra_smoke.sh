#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/humble/setup.bash

if [ -f "${ROS_WS_SETUP:-$HOME/ros2_ws/install/setup.bash}" ]; then
  # shellcheck disable=SC1090
  source "${ROS_WS_SETUP:-$HOME/ros2_ws/install/setup.bash}"
fi

GROUNDINGDINO_DIR="${GROUNDINGDINO_DIR:-/home/ada2/GroundingDINO}"
GROUNDINGDINO_CONFIG="${GROUNDINGDINO_CONFIG:-${GROUNDINGDINO_DIR}/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
GROUNDINGDINO_CHECKPOINT="${GROUNDINGDINO_CHECKPOINT:-${GROUNDINGDINO_DIR}/weights/groundingdino_swint_ogc.pth}"
INPUT_IMAGE_TOPIC="${INPUT_IMAGE_TOPIC:-/camera/camera/color/image_raw}"
INITIAL_QUERY="${INITIAL_QUERY:-stop sign, garbage bin}"

for path in "$GROUNDINGDINO_DIR" "$GROUNDINGDINO_CONFIG" "$GROUNDINGDINO_CHECKPOINT"; do
  if [ ! -e "$path" ]; then
    echo "Missing required path: $path" >&2
    exit 2
  fi
done

echo "Launching PyTorch GroundingDINO ultra smoke test."
echo "GROUNDINGDINO_DIR=$GROUNDINGDINO_DIR"
echo "INPUT_IMAGE_TOPIC=$INPUT_IMAGE_TOPIC"
echo "INITIAL_QUERY=$INITIAL_QUERY"

exec ros2 launch ros2_groundingdino jetson_orin_nano_ultra.launch.py \
  groundingdino_dir:="$GROUNDINGDINO_DIR" \
  config:="$GROUNDINGDINO_CONFIG" \
  checkpoint:="$GROUNDINGDINO_CHECKPOINT" \
  input_image_topic:="$INPUT_IMAGE_TOPIC" \
  initial_query:="$INITIAL_QUERY" \
  precision:=fp32 \
  image_size:=224 \
  max_size:=320 \
  frame_stride:=3 \
  max_detections:=20 \
  publish_output_image:=false \
  publish_legacy_outputs:=true \
  publish_legacy_image:=false
