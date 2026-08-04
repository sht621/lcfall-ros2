#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROI_CONFIG="${REPO_ROOT}/src/lcfall_ros2/config/params.yaml"
cd "${REPO_ROOT}"

./scripts/setup_and_check.sh __configure_fastdds

if [[ ! -r "${ROI_CONFIG}" ]]; then
  echo "[roi-tuner] ROI config not readable: ${ROI_CONFIG}" >&2
  exit 1
fi

if [[ -z "${DISPLAY:-}" ]]; then
  mapfile -t x_sockets < <(find /tmp/.X11-unix -maxdepth 1 -type s -name 'X*' -printf '%f\n' 2>/dev/null | sort -V)
  if [[ "${#x_sockets[@]}" -ne 1 ]]; then
    echo "[roi-tuner] DISPLAY is unset and a unique X11 display could not be selected." >&2
    exit 1
  fi
  export DISPLAY=":${x_sockets[0]#X}"
fi

if [[ -n "${XAUTHORITY:-}" && -r "${XAUTHORITY}" ]]; then
  export XAUTHORITY_HOST="${XAUTHORITY}"
elif [[ -r "/run/user/$(id -u)/gdm/Xauthority" ]]; then
  XAUTHORITY_HOST="/run/user/$(id -u)/gdm/Xauthority"
  export XAUTHORITY_HOST
else
  user_home="$(getent passwd "$(id -u)" | cut -d: -f6)"
  if [[ -z "${user_home}" || ! -r "${user_home}/.Xauthority" ]]; then
    echo "[roi-tuner] X11 authority file was not found." >&2
    exit 1
  fi
  export XAUTHORITY_HOST="${user_home}/.Xauthority"
fi

docker compose -f compose.yaml config --quiet
exec ./scripts/run.sh __compose run --rm --no-deps roi-tuner
