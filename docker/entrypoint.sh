#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="/root/ros2_ws"
LCFALL_MSGS_MARKER="${WORKSPACE_ROOT}/install/lcfall_msgs/share/lcfall_msgs/package.xml"
LCFALL_ROS2_MARKER="${WORKSPACE_ROOT}/install/lcfall_ros2/share/lcfall_ros2/package.xml"

export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
source /opt/ros/humble/setup.bash

if [[ -f "${WORKSPACE_ROOT}/install/setup.bash" ]]; then
  source "${WORKSPACE_ROOT}/install/setup.bash"
fi

if [[ "${SKIP_COLCON_BUILD:-0}" != "1" ]] && {
  [[ ! -f "${LCFALL_MSGS_MARKER}" ]] || [[ ! -f "${LCFALL_ROS2_MARKER}" ]];
}; then
  echo "[entrypoint] Building ROS workspace packages ..."
  cd "${WORKSPACE_ROOT}"
  colcon build --symlink-install --packages-select lcfall_msgs lcfall_ros2
  source "${WORKSPACE_ROOT}/install/setup.bash"
fi

exec "$@"
