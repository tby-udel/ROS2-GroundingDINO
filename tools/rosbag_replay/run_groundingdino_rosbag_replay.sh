#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <name> <bag_dir> <prompt> <labels_csv> <output_dir> [rate] [threshold] [text_threshold] [device]" >&2
  exit 2
fi

NAME="$1"
BAG_DIR="$2"
PROMPT="$3"
LABELS="$4"
OUTPUT_DIR="$5"
RATE="${6:-0.5}"
THRESHOLD="${7:-0.25}"
TEXT_THRESHOLD="${8:-0.20}"
DEVICE="${9:-cuda}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GROUNDINGDINO_DIR="${GROUNDINGDINO_DIR:-/home/boyang/safeai/GroundingDINO}"
GROUNDINGDINO_CONFIG="${GROUNDINGDINO_CONFIG:-${GROUNDINGDINO_DIR}/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
GROUNDINGDINO_CHECKPOINT="${GROUNDINGDINO_CHECKPOINT:-${GROUNDINGDINO_DIR}/weights/groundingdino_swint_ogc.pth}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
WORKSPACE_SETUP="${ROS2_GROUNDINGDINO_SETUP:-}"
if [[ -z "${WORKSPACE_SETUP}" && -f "${REPO_DIR}/../../install/setup.bash" ]]; then
  WORKSPACE_SETUP="${REPO_DIR}/../../install/setup.bash"
fi

RECORDER="${SCRIPT_DIR}/record_groundingdino_outputs.py"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

set +u
source "${ROS_SETUP}"
if [[ -n "${WORKSPACE_SETUP}" ]]; then
  source "${WORKSPACE_SETUP}"
fi
set -u

NODE_LOG="${LOG_DIR}/${NAME}_groundingdino_node.log"
RECORDER_LOG="${LOG_DIR}/${NAME}_recorder.log"
BAG_LOG="${LOG_DIR}/${NAME}_rosbag_play.log"

NODE_PID=""
RECORDER_PID=""

cleanup() {
  if [[ -n "${RECORDER_PID}" ]] && kill -0 "${RECORDER_PID}" 2>/dev/null; then
    kill "${RECORDER_PID}" 2>/dev/null || true
    wait "${RECORDER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${NODE_PID}" ]] && kill -0 "${NODE_PID}" 2>/dev/null; then
    kill "${NODE_PID}" 2>/dev/null || true
    wait "${NODE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

stdbuf -oL -eL ros2 launch ros2_groundingdino ada_reactive_perception.launch.py \
  groundingdino_dir:="${GROUNDINGDINO_DIR}" \
  config:="${GROUNDINGDINO_CONFIG}" \
  checkpoint:="${GROUNDINGDINO_CHECKPOINT}" \
  input_image_topic:=/camera/camera/color/image_raw \
  initial_query:="${PROMPT}" \
  publish_output_image:=true \
  publish_legacy_outputs:=true \
  thresholds:="${THRESHOLD}" \
  text_threshold:="${TEXT_THRESHOLD}" \
  device:="${DEVICE}" \
  >"${NODE_LOG}" 2>&1 &
NODE_PID=$!

echo "Started GroundingDINO node pid=${NODE_PID}; waiting for model readiness..."
for _ in $(seq 1 240); do
  if ! kill -0 "${NODE_PID}" 2>/dev/null; then
    echo "GroundingDINO node exited early. Log: ${NODE_LOG}" >&2
    exit 1
  fi
  if grep -q "GroundingDINO ready" "${NODE_LOG}" 2>/dev/null; then
    break
  fi
  sleep 1
done

if ! grep -q "GroundingDINO ready" "${NODE_LOG}" 2>/dev/null; then
  echo "Timed out waiting for GroundingDINO readiness. Log: ${NODE_LOG}" >&2
  exit 1
fi

python "${RECORDER}" \
  --name "${NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --labels "${LABELS}" \
  --fps 15 \
  --sample-every 45 \
  --idle-timeout 8 \
  >"${RECORDER_LOG}" 2>&1 &
RECORDER_PID=$!

sleep 2
echo "Playing ${BAG_DIR} at rate ${RATE}..."
ros2 bag play "${BAG_DIR}" \
  --rate "${RATE}" \
  --disable-keyboard-controls \
  --topics /camera/camera/color/image_raw \
  >"${BAG_LOG}" 2>&1

echo "Bag playback finished; waiting for recorder to flush..."
for _ in $(seq 1 120); do
  if ! kill -0 "${RECORDER_PID}" 2>/dev/null; then
    RECORDER_PID=""
    break
  fi
  sleep 1
done

if [[ -n "${RECORDER_PID}" ]] && kill -0 "${RECORDER_PID}" 2>/dev/null; then
  echo "Recorder did not exit after idle timeout; stopping it." >&2
  kill "${RECORDER_PID}" 2>/dev/null || true
  wait "${RECORDER_PID}" 2>/dev/null || true
  RECORDER_PID=""
fi

echo "Saved ${NAME} outputs to ${OUTPUT_DIR}"
