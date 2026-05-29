#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input.onnx> <output.engine> [precision]" >&2
  echo "precision: fp32 | fp16 | int8-calib" >&2
  exit 2
fi

ONNX="$1"
ENGINE="$2"
PRECISION="${3:-fp16}"

WORKSPACE_MB="${WORKSPACE_MB:-1024}"
MIN_SHAPE="${MIN_SHAPE:-image:1x3x224x320,encoded_text:1x64x256,text_token_mask:1x64,position_ids:1x64,text_self_attention_masks:1x64x64}"
OPT_SHAPE="${OPT_SHAPE:-${MIN_SHAPE}}"
MAX_SHAPE="${MAX_SHAPE:-${MIN_SHAPE}}"
CALIB_CACHE="${CALIB_CACHE:-}"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec not found. Install TensorRT or run this on a Jetson image that includes TensorRT." >&2
  exit 1
fi

common_args=(
  "--onnx=${ONNX}"
  "--saveEngine=${ENGINE}"
  "--memPoolSize=workspace:${WORKSPACE_MB}"
  "--minShapes=${MIN_SHAPE}"
  "--optShapes=${OPT_SHAPE}"
  "--maxShapes=${MAX_SHAPE}"
  "--builderOptimizationLevel=5"
  "--useCudaGraph"
  "--noDataTransfers"
  "--separateProfileRun"
  "--verbose"
)

case "${PRECISION}" in
  fp32)
    trtexec "${common_args[@]}"
    ;;
  fp16)
    trtexec "${common_args[@]}" --fp16
    ;;
  int8-calib)
    if [[ -z "${CALIB_CACHE}" ]]; then
      echo "Set CALIB_CACHE=/path/to/calibration.cache for int8-calib mode." >&2
      exit 2
    fi
    trtexec "${common_args[@]}" --int8 --fp16 "--calib=${CALIB_CACHE}"
    ;;
  *)
    echo "Unknown precision: ${PRECISION}" >&2
    exit 2
    ;;
esac
