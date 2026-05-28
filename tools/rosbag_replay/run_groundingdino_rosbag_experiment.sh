#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <name> <bag_dir> <initial_prompt> <labels_csv> <output_dir> [rate] [threshold] [text_threshold] [device] [publish_output_image] [publish_legacy_outputs] [query_schedule] [play_timeout] [image_size] [max_size]" >&2
  exit 2
fi

NAME="$1"
BAG_DIR="$2"
INITIAL_PROMPT="$3"
LABELS="$4"
OUTPUT_DIR="$5"
RATE="${6:-1.0}"
THRESHOLD="${7:-0.25}"
TEXT_THRESHOLD="${8:-0.20}"
DEVICE="${9:-cuda}"
PUBLISH_OUTPUT_IMAGE="${10:-false}"
PUBLISH_LEGACY_OUTPUTS="${11:-false}"
QUERY_SCHEDULE="${12:-}"
PLAY_TIMEOUT="${13:-0}"
IMAGE_SIZE="${14:-800}"
MAX_SIZE="${15:-1333}"

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
PROFILER="${SCRIPT_DIR}/profile_groundingdino_runtime.py"
QUERY_PUBLISHER="${SCRIPT_DIR}/publish_query_schedule.py"
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
PROFILER_LOG="${LOG_DIR}/${NAME}_profiler.log"
QUERY_LOG="${LOG_DIR}/${NAME}_query_schedule.jsonl"
BAG_LOG="${LOG_DIR}/${NAME}_rosbag_play.log"

NODE_PID=""
RECORDER_PID=""
PROFILER_PID=""
QUERY_PID=""

cleanup() {
  for pid in "${QUERY_PID}" "${RECORDER_PID}" "${PROFILER_PID}" "${NODE_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

stdbuf -oL -eL ros2 launch ros2_groundingdino ada_reactive_perception.launch.py \
  groundingdino_dir:="${GROUNDINGDINO_DIR}" \
  config:="${GROUNDINGDINO_CONFIG}" \
  checkpoint:="${GROUNDINGDINO_CHECKPOINT}" \
  input_image_topic:=/camera/camera/color/image_raw \
  initial_query:="${INITIAL_PROMPT}" \
  publish_output_image:="${PUBLISH_OUTPUT_IMAGE}" \
  publish_legacy_outputs:="${PUBLISH_LEGACY_OUTPUTS}" \
  thresholds:="${THRESHOLD}" \
  text_threshold:="${TEXT_THRESHOLD}" \
  device:="${DEVICE}" \
  image_size:="${IMAGE_SIZE}" \
  max_size:="${MAX_SIZE}" \
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

python "${PROFILER}" \
  --name "${NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --idle-timeout 8 \
  >"${PROFILER_LOG}" 2>&1 &
PROFILER_PID=$!

if [[ "${PUBLISH_OUTPUT_IMAGE}" == "true" || "${PUBLISH_LEGACY_OUTPUTS}" == "true" ]]; then
  python "${RECORDER}" \
    --name "${NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --labels "${LABELS}" \
    --fps 15 \
    --sample-every 45 \
    --idle-timeout 8 \
    >"${RECORDER_LOG}" 2>&1 &
  RECORDER_PID=$!
fi

if [[ -n "${QUERY_SCHEDULE}" ]]; then
  python "${QUERY_PUBLISHER}" \
    --schedule "${QUERY_SCHEDULE}" \
    --log-path "${QUERY_LOG}" \
    >"${LOG_DIR}/${NAME}_query_publisher.log" 2>&1 &
  QUERY_PID=$!
fi

sleep 2
echo "Playing ${BAG_DIR} at rate ${RATE}..."
if [[ "${PLAY_TIMEOUT}" != "0" && "${PLAY_TIMEOUT}" != "0.0" ]]; then
  set +e
  timeout "${PLAY_TIMEOUT}" ros2 bag play "${BAG_DIR}" \
    --rate "${RATE}" \
    --disable-keyboard-controls \
    --topics /camera/camera/color/image_raw \
    >"${BAG_LOG}" 2>&1
  BAG_RC=$?
  set -e
  if [[ "${BAG_RC}" != "0" && "${BAG_RC}" != "124" && "${BAG_RC}" != "143" ]]; then
    echo "ros2 bag play failed with exit code ${BAG_RC}. Log: ${BAG_LOG}" >&2
    exit "${BAG_RC}"
  fi
else
  ros2 bag play "${BAG_DIR}" \
    --rate "${RATE}" \
    --disable-keyboard-controls \
    --topics /camera/camera/color/image_raw \
    >"${BAG_LOG}" 2>&1
fi

echo "Bag playback finished; waiting for recorders to flush..."
for pid_name in RECORDER_PID PROFILER_PID QUERY_PID; do
  pid="${!pid_name}"
  if [[ -z "${pid}" ]]; then
    continue
  fi
  for _ in $(seq 1 120); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      printf -v "${pid_name}" '%s' ""
      break
    fi
    sleep 1
  done
done

echo "Saved ${NAME} outputs to ${OUTPUT_DIR}"
