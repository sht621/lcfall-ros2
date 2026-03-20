"""demo.launch.py: デモ起動 (可視化あり).

sync_preprocess_node + inference_node + alert_node + visualization_node
をすべて起動する。カメラ / LiDAR ドライバは別途起動する前提。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # パッケージ共有ディレクトリ
    pkg_share = get_package_share_directory("lcfall_ros2")
    default_params_file = os.path.join(pkg_share, "config", "params.yaml")

    # Launch 引数
    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Path to the parameters YAML file",
    )

    params_file = LaunchConfiguration("params_file")

    # sync_preprocess_node
    sync_preprocess = Node(
        package="lcfall_ros2",
        executable="sync_preprocess_node",
        name="sync_preprocess_node",
        parameters=[params_file],
        output="screen",
    )

    # inference_node
    inference = Node(
        package="lcfall_ros2",
        executable="inference_node",
        name="inference_node",
        parameters=[params_file],
        output="screen",
    )

    # alert_node
    alert = Node(
        package="lcfall_ros2",
        executable="alert_node",
        name="alert_node",
        output="screen",
    )

    # visualization_node
    visualization = Node(
        package="lcfall_ros2",
        executable="visualization_node",
        name="visualization_node",
        parameters=[params_file],
        output="screen",
    )

    return LaunchDescription([
        params_file_arg,
        sync_preprocess,
        inference,
        alert,
        visualization,
    ])
