#!/usr/bin/env bash
set -euo pipefail

PARAMS_FILE="${PARAMS_FILE:-/root/ros2_ws/install/lcfall_ros2/share/lcfall_ros2/config/params.yaml}"
LIVOX_CONFIG_PATH="${LIVOX_CONFIG_PATH:-/root/ros2_ws/src/LCFall/config/livox/MID360_config.json}"
BACKGROUND_MODEL_PATH="${BACKGROUND_MODEL_PATH:-/data/background/background_voxel_map.npz}"
CAMERA_CHECKPOINT_PATH="${CAMERA_CHECKPOINT_PATH:-/data/checkpoints/camera/best_model.pth}"
LIDAR_CHECKPOINT_PATH="${LIDAR_CHECKPOINT_PATH:-/data/checkpoints/lidar/best_model.pth}"
FUSION_CHECKPOINT_PATH="${FUSION_CHECKPOINT_PATH:-/data/checkpoints/fusion/best_model.pth}"
ENABLE_VISUALIZATION="${ENABLE_VISUALIZATION:-true}"

if [[ ! -f "${BACKGROUND_MODEL_PATH}" ]]; then
  echo "[app] Background model not found: ${BACKGROUND_MODEL_PATH}" >&2
  echo "[app] Run 'docker compose run --rm background-capture' in an empty room, then start the app again." >&2
  exit 1
fi

if [[ ! -f "${LIVOX_CONFIG_PATH}" ]]; then
  echo "[app] Livox config not found: ${LIVOX_CONFIG_PATH}" >&2
  exit 1
fi

missing_files=()
for path in \
  "${CAMERA_CHECKPOINT_PATH}" \
  "${LIDAR_CHECKPOINT_PATH}" \
  "${FUSION_CHECKPOINT_PATH}"
do
  if [[ ! -f "${path}" ]]; then
    missing_files+=("${path}")
  fi
done

if (( ${#missing_files[@]} > 0 )); then
  echo "[app] Required checkpoint files are missing:" >&2
  for path in "${missing_files[@]}"; do
    echo "  - ${path}" >&2
  done
  exit 1
fi

exec ros2 launch lcfall_ros2 lcfall.launch.py \
  params_file:="${PARAMS_FILE}" \
  livox_config_path:="${LIVOX_CONFIG_PATH}" \
  enable_visualization:="${ENABLE_VISUALIZATION}"
