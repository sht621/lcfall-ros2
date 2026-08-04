#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/visualization/compose.yaml"

if [[ -z "${DISPLAY:-}" ]]; then
  mapfile -t x_sockets < <(
    find /tmp/.X11-unix -maxdepth 1 -type s -name 'X*' \
      -printf '%f\n' 2>/dev/null | sort -V
  )
  if [[ "${#x_sockets[@]}" -ne 1 ]]; then
    echo "[visualization] DISPLAY is unset and a unique X11 display could not be selected." >&2
    exit 1
  fi
  export DISPLAY=":${x_sockets[0]#X}"
fi

if [[ -n "${XAUTHORITY:-}" && -r "${XAUTHORITY}" ]]; then
  XAUTHORITY_HOST="${XAUTHORITY}"
elif [[ -r "/run/user/$(id -u)/gdm/Xauthority" ]]; then
  XAUTHORITY_HOST="/run/user/$(id -u)/gdm/Xauthority"
else
  viewer_home="$(getent passwd "$(id -u)" | cut -d: -f6)"
  if [[ -z "${viewer_home}" || ! -r "${viewer_home}/.Xauthority" ]]; then
    echo "[visualization] X11 authority file was not found." >&2
    exit 1
  fi
  XAUTHORITY_HOST="${viewer_home}/.Xauthority"
fi
export XAUTHORITY_HOST

cd "${REPO_ROOT}"
docker compose -f "${COMPOSE_FILE}" up --no-build visualization
