"""Start only the RealSense, Livox driver, and the independent ROI tuner."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from lcfall_ros2.device_recovery import repair_realsense_video_nodes


def generate_launch_description():
    repair_realsense_video_nodes()

    repo_root = "/root/ros2_ws/src/LCFall"
    default_livox_config = os.path.join(
        repo_root, "config", "livox", "MID360_config.json"
    )

    livox_config_arg = DeclareLaunchArgument(
        "livox_config_path", default_value=default_livox_config
    )

    camera_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="camera",
        namespace="camera",
        parameters=[{
            "camera_name": "camera",
            "rgb_camera.color_format": "RGB8",
            "rgb_camera.profile": "1280x720x30",
            "enable_color": True,
            "enable_depth": False,
            "enable_infra1": False,
            "enable_infra2": False,
            "enable_gyro": False,
            "enable_accel": False,
        }],
        remappings=[("camera/color/image_raw", "/camera/image_raw")],
        output="screen",
    )

    livox_node = Node(
        package="livox_ros_driver2",
        executable="livox_ros_driver2_node",
        name="livox_lidar_publisher",
        parameters=[{
            "xfer_format": 0,
            "multi_topic": 0,
            "data_src": 0,
            "publish_freq": 10.0,
            "output_data_type": 0,
            "user_config_path": LaunchConfiguration("livox_config_path"),
        }],
        output="screen",
    )

    return LaunchDescription([
        livox_config_arg,
        camera_node,
        livox_node,
    ])
