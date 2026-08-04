FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-cjk \
    libgl1 \
    python3-colcon-common-extensions \
    python3-numpy \
    python3-opencv \
    python3-pil \
    ros-humble-cv-bridge \
    ros-humble-rmw-fastrtps-cpp \
    ros-humble-rosidl-default-generators \
    ros-humble-sensor-msgs-py \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /viewer_ws
COPY src/lcfall_msgs src/lcfall_msgs
COPY src/lcfall_ros2 src/lcfall_ros2

RUN . /opt/ros/humble/setup.sh \
    && colcon build --packages-select lcfall_msgs lcfall_ros2

COPY visualization/entrypoint.sh /viewer_entrypoint.sh
RUN chmod +x /viewer_entrypoint.sh

ENTRYPOINT ["/viewer_entrypoint.sh"]
CMD ["ros2", "launch", "lcfall_ros2", "visualization.launch.py"]
