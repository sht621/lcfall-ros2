#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/compose.yaml"
DEVICE_OVERRIDE_FILE=""
cd "${REPO_ROOT}"

cleanup_device_override() {
  if [[ -n "${DEVICE_OVERRIDE_FILE}" && -f "${DEVICE_OVERRIDE_FILE}" ]]; then
    rm -f "${DEVICE_OVERRIDE_FILE}"
  fi
}

compose_with_video_devices() {
  local -a video_devices=()
  mapfile -t video_devices < <(
    find /dev -maxdepth 1 -type c -name 'video*' -print | sort -V
  )

  DEVICE_OVERRIDE_FILE="$(mktemp /tmp/lcfall-compose-video-devices.XXXXXX.yaml)"
  trap cleanup_device_override EXIT INT TERM
  {
    echo "services:"
    local service device
    for service in app roi-tuner; do
      echo "  ${service}:"
      if [[ "${#video_devices[@]}" -gt 0 ]]; then
        echo "    devices:"
        for device in "${video_devices[@]}"; do
          echo "      - \"${device}:${device}\""
        done
      else
        echo "    devices: []"
      fi
    done
  } > "${DEVICE_OVERRIDE_FILE}"

  if [[ "${#video_devices[@]}" -gt 0 ]]; then
    printf '[compose] Added video devices: %s\n' "${video_devices[*]}"
  else
    echo "[compose] No /dev/video* devices found."
  fi

  local result
  docker compose \
    -f "${COMPOSE_FILE}" \
    -f "${DEVICE_OVERRIDE_FILE}" \
    "$@"
  result=$?
  cleanup_device_override
  DEVICE_OVERRIDE_FILE=""
  trap - EXIT INT TERM
  return "${result}"
}

if [[ "${1:-}" == "__compose" ]]; then
  shift
  compose_with_video_devices "$@"
  exit $?
fi

enable_visualization="${ENABLE_VISUALIZATION:-true}"
case "${enable_visualization,,}" in
  false|0|off|no)
    ;;
  *)
    if [[ -z "${DISPLAY:-}" ]]; then
      mapfile -t x_sockets < <(
        find /tmp/.X11-unix -maxdepth 1 -type s -name 'X*' \
          -printf '%f\n' 2>/dev/null | sort -V
      )
      if [[ "${#x_sockets[@]}" -ne 1 ]]; then
        echo "[run] DISPLAY is unset and a unique X11 display could not be selected." >&2
        exit 1
      fi
      export DISPLAY=":${x_sockets[0]#X}"
    fi

    if [[ -n "${XAUTHORITY:-}" && -r "${XAUTHORITY}" ]]; then
      XAUTHORITY_HOST="${XAUTHORITY}"
    elif [[ -r "/run/user/$(id -u)/gdm/Xauthority" ]]; then
      XAUTHORITY_HOST="/run/user/$(id -u)/gdm/Xauthority"
    else
      user_home="$(getent passwd "$(id -u)" | cut -d: -f6)"
      if [[ -z "${user_home}" || ! -r "${user_home}/.Xauthority" ]]; then
        echo "[run] X11 authority file was not found." >&2
        exit 1
      fi
      XAUTHORITY_HOST="${user_home}/.Xauthority"
    fi
    export XAUTHORITY_HOST
    ;;
esac

compose_with_video_devices up --no-build app
