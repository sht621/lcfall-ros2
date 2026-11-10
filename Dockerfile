# syntax=docker/dockerfile:1
# Standard ROS only. Source/configuration artifact; no build has been run.
FROM ros:jazzy-ros-base
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
ARG LIBREALSENSE_COMMIT
ARG REALSENSE_ROS_COMMIT
ARG LIVOX_SDK2_COMMIT
ARG LIVOX_DRIVER_COMMIT
ARG BUILD_JOBS=1
ENV PYTHONUNBUFFERED=1 \
    ROS_INSTALL_BASE=/opt/ros/jazzy WORKSPACE_DIRECTORY=/opt/edge_ws \
    LD_LIBRARY_PATH=/usr/local/lib \
    CMAKE_PREFIX_PATH=/usr/local \
    RCUTILS_LOGGING_USE_STDOUT=1 \
    RCUTILS_LOGGING_BUFFERED_STREAM=1
COPY build-tools.sh /usr/local/lib/edge-build-tools.sh
RUN test "$(dpkg --print-architecture)" = arm64 && \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git cmake build-essential pkg-config python3-pip python3-dev \
    python3-colcon-common-extensions python3-numpy python3-opencv python3-yaml \
    python3-packaging python3-argcomplete python3-contextlib2 python3-future python3-netaddr python3-netifaces \
    libusb-1.0-0-dev libudev-dev libssl-dev libcurl4-openssl-dev \
    libgl1-mesa-dev libglfw3-dev libopencv-dev libpcl-dev libapr1-dev rapidjson-dev \
    ros-jazzy-cv-bridge ros-jazzy-message-filters ros-jazzy-tf2-ros \
    ros-jazzy-tf2-ros-py ros-jazzy-tf2-eigen ros-jazzy-tf2-geometry-msgs \
    ros-jazzy-pcl-conversions ros-jazzy-diagnostic-updater ros-jazzy-image-transport \
    ros-jazzy-camera-info-manager ros-jazzy-rosidl-default-generators \
    ros-jazzy-ament-cmake-auto ros-jazzy-rclcpp-components ros-jazzy-rclcpp-action \
    ros-jazzy-rosbag2 ros-jazzy-nav-msgs && apt-get clean

RUN --mount=type=cache,id=edge-standard-sources-v1,target=/var/cache/edge-build,sharing=locked <<'BUILD'
source /usr/local/lib/edge-build-tools.sh
mkdir -p /var/cache/edge-build/librealsense /var/cache/edge-build/Livox-SDK2
cd /var/cache/edge-build/librealsense
git-clone https://github.com/realsenseai/librealsense.git "$LIBREALSENSE_COMMIT" "$LIBREALSENSE_COMMIT"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFORCE_RSUSB_BACKEND=ON \
    -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_WITH_CUDA=OFF
cmake --build build -j "$BUILD_JOBS"
cmake --install build
cd /var/cache/edge-build/Livox-SDK2
git-clone https://github.com/Livox-SDK/Livox-SDK2.git "$LIVOX_SDK2_COMMIT" "$LIVOX_SDK2_COMMIT"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$BUILD_JOBS"
cmake --install build
ldconfig
BUILD
WORKDIR /opt/edge_ws
RUN --mount=type=cache,id=edge-standard-sources-v1,target=/var/cache/edge-build,sharing=locked <<'BUILD'
source /usr/local/lib/edge-build-tools.sh
mkdir -p /var/cache/edge-build/realsense-ros /var/cache/edge-build/livox_ros_driver2 src
cd /var/cache/edge-build/realsense-ros
git-clone https://github.com/realsenseai/realsense-ros.git "$REALSENSE_ROS_COMMIT" "$REALSENSE_ROS_COMMIT"
cp -a . /opt/edge_ws/src/realsense-ros
cd /var/cache/edge-build/livox_ros_driver2
git-clone https://github.com/Livox-SDK/livox_ros_driver2.git "$LIVOX_DRIVER_COMMIT" "$LIVOX_DRIVER_COMMIT"
cp -a . /opt/edge_ws/src/livox_ros_driver2
cp /opt/edge_ws/src/livox_ros_driver2/package_ROS2.xml /opt/edge_ws/src/livox_ros_driver2/package.xml
BUILD
COPY hailo-runtime-tools.sh /usr/local/lib/edge-hailo-runtime-tools.sh
RUN source /usr/local/lib/edge-hailo-runtime-tools.sh && install-hailort
COPY models/pose_model.yaml /models/pose_model.yaml
RUN source /usr/local/lib/edge-hailo-runtime-tools.sh && prepare-pose-model
COPY src/fall_detection_msgs/ src/fall_detection_msgs/
COPY src/fall_detection_edge/ src/fall_detection_edge/
RUN <<'BUILD'
source /opt/ros/jazzy/setup.bash
source /usr/local/lib/edge-build-tools.sh
EDGE_CMAKE_ARGS=(-DROS_EDITION=ROS2 -DDISTRO_ROS=jazzy)
colcon-build src /opt/edge_ws/install fall_detection_edge
BUILD
COPY config/livox/MID360_config.json /opt/edge-config/MID360_config.json
COPY --chmod=755 ros_entrypoint.sh /ros_entrypoint.sh
RUN if ! getent passwd 1000 > /dev/null; then useradd --uid 1000 --create-home --shell /bin/bash edge; fi
USER 1000:1000
ENV XDG_CACHE_HOME=/tmp/edge-cache ROS_LOG_DIR=/tmp/ros-logs \
    LIVOX_CONFIG_PATH=/opt/edge-config/MID360_config.json
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["ros2", "launch", "fall_detection_edge", "edge.launch.py", "use_edgerm:=false"]
