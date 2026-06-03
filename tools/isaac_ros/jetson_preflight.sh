#!/usr/bin/env bash
set -euo pipefail

section() {
  printf '\n== %s ==\n' "$1"
}

have() {
  command -v "$1" >/dev/null 2>&1
}

print_cmd() {
  local label="$1"
  shift
  printf '\n-- %s --\n' "$label"
  "$@" 2>&1 || true
}

section "Host"
print_cmd "uname" uname -a
print_cmd "architecture" uname -m
if [ -r /etc/os-release ]; then
  print_cmd "os-release" bash -lc 'grep -E "^(PRETTY_NAME|VERSION_ID|VERSION_CODENAME)=" /etc/os-release'
fi

section "Jetson / JetPack"
if [ -r /etc/nv_tegra_release ]; then
  print_cmd "nv_tegra_release" cat /etc/nv_tegra_release
else
  echo "WARN: /etc/nv_tegra_release not found. This may not be a Jetson rootfs."
fi
print_cmd "nvidia-l4t-core" bash -lc 'dpkg-query -W nvidia-l4t-core 2>/dev/null || true'
if have jetson_release; then
  print_cmd "jetson_release" jetson_release
fi

section "GPU / Power / Memory"
print_cmd "free" free -h
print_cmd "disk" df -h /
if have nvpmodel; then
  print_cmd "nvpmodel" nvpmodel -q
else
  echo "WARN: nvpmodel not found."
fi
if have tegrastats; then
  echo "OK: tegrastats is available."
else
  echo "WARN: tegrastats not found."
fi

section "Docker"
if have docker; then
  print_cmd "docker version" docker --version
  print_cmd "docker runtimes" bash -lc 'docker info 2>/dev/null | sed -n "/Runtimes:/,/Default Runtime:/p"'
else
  echo "WARN: docker is not installed."
fi

section "ROS"
if [ -d /opt/ros/humble ]; then
  echo "OK: /opt/ros/humble exists."
fi
if [ -d /opt/ros/jazzy ]; then
  echo "OK: /opt/ros/jazzy exists."
fi
if have ros2; then
  print_cmd "ros2" bash -lc 'echo "ROS_DISTRO=${ROS_DISTRO:-unset}"; ros2 --help | head -5'
else
  echo "INFO: ros2 is not on PATH in this shell."
fi

section "Isaac ROS Compatibility Hint"
arch="$(uname -m)"
ubuntu_codename=""
if [ -r /etc/os-release ]; then
  ubuntu_codename="$(. /etc/os-release && printf '%s' "${VERSION_CODENAME:-}")"
fi
l4t_release=""
if [ -r /etc/nv_tegra_release ]; then
  l4t_release="$(sed -n 's/^# R\([0-9][0-9]\).*/R\1/p' /etc/nv_tegra_release | head -1)"
fi

if [ "$arch" != "aarch64" ]; then
  echo "WARN: This is not a Jetson/aarch64 machine."
elif [ "$ubuntu_codename" = "noble" ]; then
  echo "LIKELY PATH A: Ubuntu noble detected. Try Isaac ROS release-4.4 Jazzy/noble-jetpack first."
elif [ "$ubuntu_codename" = "jammy" ] || [ "$l4t_release" = "R36" ]; then
  echo "LIKELY PATH B: JetPack 6.x / Ubuntu 22.04 style environment detected."
  echo "Do not spend the first hours forcing release-4.4 Jazzy bare-metal packages."
  echo "Try an Isaac ROS container path first, or fall back to the local Humble wrapper."
else
  echo "UNKNOWN: Check JetPack/Ubuntu compatibility before installing Isaac ROS packages."
fi

section "Recommended First Commands Tomorrow"
cat <<'EOF'
1. Put Jetson in max-power mode before model build:
   sudo nvpmodel -m 0 || true
   sudo jetson_clocks || true

2. Watch memory while building/loading the engine:
   sudo tegrastats

3. Build the TensorRT plan on the Jetson itself.
   Do not reuse the x86 RTX 4090 .plan file as a deployment artifact.
EOF
