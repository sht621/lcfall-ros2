#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

enable_visualization="${ENABLE_VISUALIZATION:-true}"
args=()

for arg in "$@"; do
  case "${arg}" in
    --no-vis)
      enable_visualization="false"
      ;;
    *)
      args+=("${arg}")
      ;;
  esac
done

cd "${REPO_ROOT}"
ENABLE_VISUALIZATION="${enable_visualization}" \
  exec ./docker/compose_with_video_devices.sh up "${args[@]}" app
