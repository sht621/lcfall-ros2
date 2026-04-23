#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || echo "${SCRIPT_DIR}")"

enable_visualization="${ENABLE_VISUALIZATION:-true}"
args=()
print_config=0

for arg in "$@"; do
  case "${arg}" in
    --no-vis)
      enable_visualization="false"
      ;;
    --config)
      print_config=1
      ;;
    *)
      args+=("${arg}")
      ;;
  esac
done

cd "${REPO_ROOT}"

if [[ "${print_config}" == "1" ]]; then
  exec ./docker/compose_with_video_devices.sh config
fi

ENABLE_VISUALIZATION="${enable_visualization}" \
  exec ./docker/compose_with_video_devices.sh up "${args[@]}" app
