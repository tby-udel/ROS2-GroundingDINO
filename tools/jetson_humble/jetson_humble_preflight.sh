#!/usr/bin/env bash
set -euo pipefail

section() {
  printf '\n== %s ==\n' "$1"
}

run() {
  local label="$1"
  shift
  printf '\n-- %s --\n' "$label"
  "$@" 2>&1 || true
}

section "Jetson / OS"
run "uname" uname -a
if [ -r /etc/os-release ]; then
  run "os-release" bash -lc 'grep -E "^(PRETTY_NAME|VERSION_ID|VERSION_CODENAME)=" /etc/os-release'
fi
if [ -r /etc/nv_tegra_release ]; then
  run "nv_tegra_release" cat /etc/nv_tegra_release
else
  echo "WARN: /etc/nv_tegra_release not found."
fi
run "nvidia-l4t-core" bash -lc 'dpkg-query -W nvidia-l4t-core 2>/dev/null || true'

section "Memory / Power"
run "free" free -h
run "disk" df -h /
if command -v nvpmodel >/dev/null 2>&1; then
  run "nvpmodel" nvpmodel -q
else
  echo "WARN: nvpmodel not found."
fi
if command -v tegrastats >/dev/null 2>&1; then
  echo "OK: tegrastats is available."
else
  echo "WARN: tegrastats not found."
fi

section "ROS Humble"
if [ -d /opt/ros/humble ]; then
  echo "OK: /opt/ros/humble exists."
else
  echo "FAIL: /opt/ros/humble not found."
fi
run "ros2 after sourcing humble" bash -lc 'source /opt/ros/humble/setup.bash 2>/dev/null && echo "ROS_DISTRO=${ROS_DISTRO}" && ros2 --help | head -5'

section "Python / PyTorch / CUDA"
python3 - <<'PY' || true
import importlib.util
import sys

print("python:", sys.version.replace("\n", " "))
for name in ["torch", "torchvision", "cv2", "transformers", "timm", "cv_bridge"]:
    spec = importlib.util.find_spec(name)
    print(f"{name}:", "OK" if spec else "MISSING")

try:
    import torch
    print("torch.__version__:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))
        free, total = torch.cuda.mem_get_info()
        print("cuda memory free/total GiB:", round(free / 2**30, 2), "/", round(total / 2**30, 2))
except Exception as exc:
    print("torch probe error:", repr(exc))
PY

section "TensorRT"
if command -v trtexec >/dev/null 2>&1; then
  run "trtexec version" bash -lc 'trtexec --version || trtexec --help | head -20'
else
  echo "INFO: trtexec not found on PATH. PyTorch smoke test can still run."
fi
run "TensorRT apt packages" bash -lc 'dpkg-query -W "tensorrt*" "libnvinfer*" 2>/dev/null | head -30'

section "GroundingDINO Paths"
GROUNDINGDINO_DIR="${GROUNDINGDINO_DIR:-/home/ada2/GroundingDINO}"
CHECKPOINT="${GROUNDINGDINO_CHECKPOINT:-${GROUNDINGDINO_DIR}/weights/groundingdino_swint_ogc.pth}"
CONFIG="${GROUNDINGDINO_CONFIG:-${GROUNDINGDINO_DIR}/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
for path in "$GROUNDINGDINO_DIR" "$CONFIG" "$CHECKPOINT"; do
  if [ -e "$path" ]; then
    echo "OK: $path"
  else
    echo "MISSING: $path"
  fi
done

section "Decision"
cat <<'EOF'
If ROS Humble and torch.cuda.is_available are both OK:
  run tools/jetson_humble/run_pytorch_ultra_smoke.sh first.

If PyTorch CUDA is missing:
  fix the Jetson PyTorch wheel/environment before debugging GroundingDINO.

If the PyTorch node loads but OOMs:
  close other processes, use max power, keep precision=fp32, image_size=224, max_size=320, frame_stride=3.

If PyTorch works but is too slow:
  move to scheduled 1-5 Hz inference, or try the TensorRT ONNX path later.
EOF
