#!/bin/bash
set -e
mkdir -p "${XDG_CACHE_HOME:-/tmp/edge-cache}"
source /opt/ros/jazzy/setup.bash
source /opt/edge_ws/install/setup.bash
exec "$@"
