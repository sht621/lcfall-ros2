# SOURCE ONLY: Pi 5 ARM64 / Jazzy single image; not built or validated on MacBook.
ARG EDGE_ROS_BASE_IMAGE=ros:jazzy-ros-base
FROM ${EDGE_ROS_BASE_IMAGE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
# Exact upstream revisions must be supplied during the authorized Pi stage.
ARG LIBREALSENSE_REF
ARG REALSENSE_ROS_REF
ARG LIVOX_SDK2_REF
ARG LIVOX_DRIVER_REF
ARG HAILORT_DEB
ARG HAILORT_WHEEL
ARG BUILD_JOBS=2
ARG WITH_EDGERM=OFF
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV RCUTILS_LOGGING_USE_STDOUT=1
ENV RCUTILS_LOGGING_BUFFERED_STREAM=1

RUN test "$(dpkg --print-architecture)" = arm64 && \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git cmake build-essential pkg-config python3-pip python3-dev \
    python3-colcon-common-extensions python3-numpy python3-opencv python3-yaml \
    libusb-1.0-0-dev libudev-dev libssl-dev libcurl4-openssl-dev libgl1-mesa-dev libglfw3-dev \
    libpcl-dev rapidjson-dev \
    ros-jazzy-cv-bridge ros-jazzy-message-filters ros-jazzy-tf2-ros \
    ros-jazzy-tf2-eigen ros-jazzy-tf2-geometry-msgs ros-jazzy-tf2-sensor-msgs \
    ros-jazzy-pcl-conversions ros-jazzy-diagnostic-updater ros-jazzy-image-transport \
    ros-jazzy-camera-info-manager ros-jazzy-xacro ros-jazzy-rosidl-default-generators \
    ros-jazzy-ament-cmake-auto ros-jazzy-rclcpp-components ros-jazzy-nav-msgs \
    ros-jazzy-ament-cmake-python ros-jazzy-pybind11-vendor pybind11-dev && \
    apt-get clean

# D455 userspace RSUSB backend; no host kernel module rebuild inside the container.
RUN test -n "$LIBREALSENSE_REF" && \
    git clone https://github.com/IntelRealSense/librealsense.git /tmp/librealsense && \
    git -C /tmp/librealsense checkout --detach "$LIBREALSENSE_REF" && \
    cmake -S /tmp/librealsense -B /tmp/librealsense/build \
      -DCMAKE_BUILD_TYPE=Release -DFORCE_RSUSB_BACKEND=ON \
      -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF \
      -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_WITH_CUDA=OFF && \
    cmake --build /tmp/librealsense/build -j "$BUILD_JOBS" && \
    cmake --install /tmp/librealsense/build && ldconfig

RUN test -n "$LIVOX_SDK2_REF" && \
    git clone https://github.com/Livox-SDK/Livox-SDK2.git /tmp/Livox-SDK2 && \
    git -C /tmp/Livox-SDK2 checkout --detach "$LIVOX_SDK2_REF" && \
    cmake -S /tmp/Livox-SDK2 -B /tmp/Livox-SDK2/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /tmp/Livox-SDK2/build -j "$BUILD_JOBS" && \
    cmake --install /tmp/Livox-SDK2/build && ldconfig

WORKDIR /opt/edge_ws
RUN test -n "$REALSENSE_ROS_REF" && test -n "$LIVOX_DRIVER_REF" && \
    mkdir -p src && \
    git clone https://github.com/IntelRealSense/realsense-ros.git src/realsense-ros && \
    git -C src/realsense-ros checkout --detach "$REALSENSE_ROS_REF" && \
    git clone https://github.com/Livox-SDK/livox_ros_driver2.git src/livox_ros_driver2 && \
    git -C src/livox_ros_driver2 checkout --detach "$LIVOX_DRIVER_REF" && \
    cp src/livox_ros_driver2/package_ROS2.xml src/livox_ros_driver2/package.xml

COPY artifacts/ /tmp/hailo-artifacts/
RUN test -n "$HAILORT_DEB" && test -n "$HAILORT_WHEEL" && \
    apt-get update && apt-get install -y --no-install-recommends "/tmp/hailo-artifacts/$HAILORT_DEB" && \
    python3 -m pip install --break-system-packages --no-cache-dir "/tmp/hailo-artifacts/$HAILORT_WHEEL" && \
    apt-get clean

COPY src/fall_detection_msgs/ src/fall_detection_msgs/
COPY src/fall_detection_edge/ src/fall_detection_edge/
RUN source /opt/ros/jazzy/setup.bash && \
    colcon build --parallel-workers "$BUILD_JOBS" \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DROS_EDITION=ROS2 -DDISTRO_ROS=jazzy \
      -DBUILD_TESTING=OFF -DFALL_DETECTION_WITH_EDGERM="$WITH_EDGERM"

COPY config/livox/MID360_config.json /opt/edge-config/MID360_config.json
COPY docker/entrypoint.sh /edge-entrypoint.sh
RUN chmod +x /edge-entrypoint.sh && \
    if ! getent passwd 1000 > /dev/null; then \
      useradd --uid 1000 --create-home --shell /bin/bash edge; \
    fi
USER 1000:1000
ENV XDG_CACHE_HOME=/tmp/edge-cache
ENV ROS_LOG_DIR=/tmp/ros-logs
ENV LIVOX_CONFIG_PATH=/opt/edge-config/MID360_config.json
ENTRYPOINT ["/edge-entrypoint.sh"]
CMD ["ros2", "launch", "fall_detection_edge", "edge.launch.py"]
