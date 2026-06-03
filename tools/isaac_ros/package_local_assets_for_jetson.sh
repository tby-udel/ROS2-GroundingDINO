#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-/home/boyang/safeai/GroundingDINO/outputs/isaac_ros/jetson_transfer}"
isaac_ws="${ISAAC_ROS_WS:-/home/boyang/workspaces/isaac_ros-dev}"
model_dir="${isaac_ws}/isaac_ros_assets/models/grounding_dino"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

mkdir -p "$out_dir"
stamp="$(date +%Y%m%d_%H%M%S)"
manifest="${out_dir}/manifest_${stamp}.txt"
tarball="${out_dir}/grounding_dino_isaac_ros_transfer_${stamp}.tar.gz"

{
  echo "Created: $(date -Is)"
  echo "Isaac ROS workspace: $isaac_ws"
  echo "Model dir: $model_dir"
  echo "Repo dir: $repo_dir"
  echo
  echo "Files included:"
} > "$manifest"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/models/grounding_dino" "$tmp/docs" "$tmp/tools/isaac_ros"

if [ -f "$model_dir/grounding_dino_model.onnx" ]; then
  cp "$model_dir/grounding_dino_model.onnx" "$tmp/models/grounding_dino/"
  echo "models/grounding_dino/grounding_dino_model.onnx" >> "$manifest"
else
  echo "WARN: ONNX model not found at $model_dir/grounding_dino_model.onnx" | tee -a "$manifest"
fi

cat > "$tmp/README_JETSON_TRANSFER.txt" <<'EOF'
This bundle is for speeding up Jetson Grounding DINO deployment.

Important:
- The ONNX model is portable.
- The TensorRT .plan generated on x86 RTX 4090 is not treated as a Jetson deployment artifact.
- Build the .plan on the Jetson target so TensorRT can optimize for the target GPU, CUDA, TensorRT, and plugin versions.
EOF
cp "$repo_dir/docs/jetson_isaac_ros_fast_path.md" "$tmp/docs/" 2>/dev/null || true
cp "$repo_dir/tools/isaac_ros/jetson_preflight.sh" "$tmp/tools/isaac_ros/"
cp "$repo_dir/tools/isaac_ros/launch_grounding_dino_reference.sh" "$tmp/tools/isaac_ros/"
cp "$repo_dir/tools/isaac_ros/set_grounding_dino_prompt.sh" "$tmp/tools/isaac_ros/"

tar -C "$tmp" -czf "$tarball" .

echo "Tarball: $tarball" | tee -a "$manifest"
echo "Manifest: $manifest"
