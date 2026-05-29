#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <output.onnx> [preset]" >&2
  echo "Example: $0 outputs/tensorrt/groundingdino_open_vocab_jetson_ultra.onnx jetson-ultra" >&2
  exit 2
fi

OUTPUT="$1"
PRESET="${2:-jetson-ultra}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GROUNDINGDINO_DIR="${GROUNDINGDINO_DIR:-/home/ada2/GroundingDINO}"
GROUNDINGDINO_CONFIG="${GROUNDINGDINO_CONFIG:-${GROUNDINGDINO_DIR}/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
GROUNDINGDINO_CHECKPOINT="${GROUNDINGDINO_CHECKPOINT:-${GROUNDINGDINO_DIR}/weights/groundingdino_swint_ogc.pth}"

python "${SCRIPT_DIR}/export_open_vocab_onnx.py" \
  --preset "${PRESET}" \
  --groundingdino-dir "${GROUNDINGDINO_DIR}" \
  --config "${GROUNDINGDINO_CONFIG}" \
  --checkpoint "${GROUNDINGDINO_CHECKPOINT}" \
  --output "${OUTPUT}"
