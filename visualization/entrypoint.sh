#!/usr/bin/env bash
set -euo pipefail

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source /viewer_ws/install/setup.bash
set -u

exec "$@"
