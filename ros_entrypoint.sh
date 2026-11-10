#!/bin/bash
set -e -o pipefail

ros_setup=/opt/ros/jazzy/setup.bash
edge_setup=/opt/edge_ws/install/setup.bash
for setup_file in "$ros_setup" "$edge_setup"; do
    if [[ ! -r "$setup_file" ]]; then
        echo "Required ROS setup file is missing or unreadable: $setup_file" >&2
        exit 1
    fi
done
source "$ros_setup" --
source "$edge_setup" --
mkdir -p "${XDG_CACHE_HOME:-/tmp/edge-cache}"
if (( $# == 0 )); then
    echo 'No command supplied to ros_entrypoint.sh' >&2
    exit 1
fi
exec "$@"
