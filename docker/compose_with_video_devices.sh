#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OVERRIDE_FILE="${REPO_ROOT}/docker/compose.video-devices.yaml"

mapfile -t VIDEO_DEVICES < <(ls /dev/video* 2>/dev/null | sort -V || true)

{
  echo "services:"
  echo "  app:"
  if [[ "${#VIDEO_DEVICES[@]}" -gt 0 ]]; then
    echo "    devices:"
    for device in "${VIDEO_DEVICES[@]}"; do
      echo "      - \"${device}:${device}\""
    done
  else
    echo "    # No /dev/video* devices were found on the host."
    echo "    devices: []"
  fi
} > "${OVERRIDE_FILE}"

if [[ "${#VIDEO_DEVICES[@]}" -gt 0 ]]; then
  printf '[compose] Added video devices: %s\n' "${VIDEO_DEVICES[*]}"
else
  echo "[compose] No /dev/video* devices found."
fi

exec docker compose \
  -f "${REPO_ROOT}/compose.yaml" \
  -f "${OVERRIDE_FILE}" \
  "$@"
