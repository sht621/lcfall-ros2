FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    ros-jazzy-rmw-fastrtps-cpp \
    ros-jazzy-rosidl-default-generators \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /jazzy_ws

COPY src/lcfall_msgs /jazzy_ws/src/lcfall_msgs
COPY --chmod=755 validation/jazzy/entrypoint.sh /jazzy_test_entrypoint.sh
COPY validation/jazzy/fall_subscriber.py /jazzy_fall_subscriber.py

RUN . /opt/ros/jazzy/setup.sh && \
    colcon build --packages-select lcfall_msgs && \
    rm -rf build log

ENTRYPOINT ["/jazzy_test_entrypoint.sh"]
CMD ["python3", "/jazzy_fall_subscriber.py"]
