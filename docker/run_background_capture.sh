#!/usr/bin/env bash
set -euo pipefail

PARAMS_FILE="${PARAMS_FILE:-/root/ros2_ws/install/lcfall_ros2/share/lcfall_ros2/config/params.yaml}"
LIVOX_CONFIG_PATH="${LIVOX_CONFIG_PATH:-/root/ros2_ws/src/LCFall/config/livox/MID360_config.json}"

mkdir -p /data/background

echo "[background-capture] Make sure the room is empty before capture starts."

if [[ ! -f "${LIVOX_CONFIG_PATH}" ]]; then
  echo "[background-capture] Livox config not found: ${LIVOX_CONFIG_PATH}" >&2
  exit 1
fi

exec ros2 launch lcfall_ros2 capture_background.launch.py \
  params_file:="${PARAMS_FILE}" \
  livox_config_path:="${LIVOX_CONFIG_PATH}"
