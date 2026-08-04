"""可視化ノードだけを起動するlaunchファイル."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("lcfall_ros2")
    default_params_file = os.path.join(
        package_share, "config", "params.yaml"
    )

    params_file = LaunchConfiguration("params_file")
    camera_topic = LaunchConfiguration("camera_topic")
    compressed_camera_topic = LaunchConfiguration(
        "compressed_camera_topic"
    )
    use_compressed_image = LaunchConfiguration("use_compressed_image")
    preprocessed_topic = LaunchConfiguration("preprocessed_topic")
    result_topic = LaunchConfiguration("result_topic")

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params_file,
            description="Path to the visualization parameters YAML file",
        ),
        DeclareLaunchArgument(
            "camera_topic",
            default_value="/camera/image_raw",
            description="Camera image topic",
        ),
        DeclareLaunchArgument(
            "compressed_camera_topic",
            default_value="/camera/visualization/compressed",
            description="Reduced JPEG camera topic for a remote viewer",
        ),
        DeclareLaunchArgument(
            "use_compressed_image",
            default_value="false",
            description="Subscribe to the compressed camera topic",
        ),
        DeclareLaunchArgument(
            "preprocessed_topic",
            default_value="/preprocessed/frame",
            description="Preprocessed skeleton and point-cloud topic",
        ),
        DeclareLaunchArgument(
            "result_topic",
            default_value="/fall_detection/result",
            description="Fall detection result topic",
        ),
        Node(
            package="lcfall_ros2",
            executable="visualization_node",
            name="visualization_node",
            parameters=[
                params_file,
                {
                    "camera_topic": camera_topic,
                    "compressed_camera_topic": compressed_camera_topic,
                    "use_compressed_image": use_compressed_image,
                    "preprocessed_topic": preprocessed_topic,
                    "result_topic": result_topic,
                },
            ],
            output="screen",
        ),
    ])
