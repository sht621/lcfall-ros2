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
option(FALL_DETECTION_WITH_EDGERM "Build the optional EdgeRM service binding" OFF)
if(FALL_DETECTION_WITH_EDGERM)
  # Requires EdgeRM-patched ROS 2 and the matching development libraries.
  # Keep the common message-only build independent of these dependencies.
  find_package(ament_cmake_python REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(rmw_edgerm_cpp REQUIRED)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  find_package(pybind11_vendor REQUIRED)
  find_package(pybind11 REQUIRED)
  pybind11_add_module(_service SHARED src/edgerm_service.cpp)
  target_compile_features(_service PRIVATE cxx_std_17)
  rosidl_get_typesupport_target(interfaces_target ${PROJECT_NAME} rosidl_typesupport_cpp)
  target_link_libraries(_service PRIVATE rmw_edgerm_cpp::rmw_edgerm "${interfaces_target}")
  ament_python_install_package(fall_detection_edgerm)
  install(TARGETS _service DESTINATION "${PYTHON_INSTALL_DIR}/fall_detection_edgerm")
endif()
ament_package()
