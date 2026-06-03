#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/jazzy/setup.bash

prompt="${1:-}"
if [ -z "$prompt" ]; then
  cat >&2 <<'EOF'
Usage:
  tools/isaac_ros/set_grounding_dino_prompt.sh "chair.box.stop sign."

Isaac ROS Grounding DINO expects period-separated classes, not comma-separated classes.
EOF
  exit 2
fi

ros2 service call /set_prompt isaac_ros_grounding_dino_interfaces/srv/SetPrompt \
  "{prompt: \"$prompt\"}"
