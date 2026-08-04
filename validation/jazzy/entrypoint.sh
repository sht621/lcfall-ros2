#!/usr/bin/env bash
set -euo pipefail

set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source /jazzy_ws/install/setup.bash
set -u

exec "$@"
