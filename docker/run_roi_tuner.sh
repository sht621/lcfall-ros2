#!/usr/bin/env bash
set -euo pipefail

ROI_CONFIG_PATH="${ROI_CONFIG_PATH:-/root/ros2_ws/src/LCFall/src/lcfall_ros2/config/params.yaml}"
LIVOX_CONFIG_PATH="${LIVOX_CONFIG_PATH:-/root/ros2_ws/src/LCFall/config/livox/MID360_config.json}"

for path in "${ROI_CONFIG_PATH}" "${LIVOX_CONFIG_PATH}" "${FASTRTPS_DEFAULT_PROFILES_FILE}"; do
  if [[ ! -r "${path}" ]]; then
    echo "[roi-tuner] Required file not readable: ${path}" >&2
    exit 1
  fi
done

ros2 launch lcfall_ros2 roi_tuner.launch.py \
  livox_config_path:="${LIVOX_CONFIG_PATH}" &
sensor_launch_pid=$!

cleanup() {
  if kill -0 "${sensor_launch_pid}" 2>/dev/null; then
    kill -TERM "${sensor_launch_pid}" 2>/dev/null || true
    wait "${sensor_launch_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

set +e
python3 /root/ros2_ws/src/LCFall/tools/roi_tuner/roi_tuner.py \
  --config "${ROI_CONFIG_PATH}"
result=$?
set -e

cleanup
trap - EXIT INT TERM
exit "${result}"
