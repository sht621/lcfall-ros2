cmake_minimum_required(VERSION 3.8)
project(fall_detection_msgs)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PersonPose.msg"
  "msg/PoseFrame.msg"
  "msg/SyncPairInfo.msg"
  "msg/SynchronizedFrame.msg"
  "srv/SendSynchronizedFrame.srv"
  DEPENDENCIES builtin_interfaces std_msgs sensor_msgs
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
