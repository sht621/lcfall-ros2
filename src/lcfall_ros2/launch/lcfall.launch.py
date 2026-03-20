"""lcfall.launch.py: 転倒検知システム起動.

カメラドライバ + LiDAR ドライバ + 全ノードを一括起動する。
可視化はデフォルト ON。--ros-args で OFF にできる。

使い方:
    # 可視化あり (デフォルト)
    ros2 launch lcfall_ros2 lcfall.launch.py

    # 可視化なし
    ros2 launch lcfall_ros2 lcfall.launch.py enable_visualization:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # パッケージ共有ディレクトリ
    pkg_share = get_package_share_directory("lcfall_ros2")
    default_params_file = os.path.join(pkg_share, "config", "params.yaml")

    # ------------------------------------------------------------------
    # Launch 引数
    # ------------------------------------------------------------------
    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Path to the parameters YAML file",
    )

    enable_vis_arg = DeclareLaunchArgument(
        "enable_visualization",
        default_value="true",
        description="Enable visualization node (default: true)",
    )

    params_file = LaunchConfiguration("params_file")
    enable_vis = LaunchConfiguration("enable_visualization")

    # ------------------------------------------------------------------
    # センサドライバ: カメラ (RealSense)
    # ------------------------------------------------------------------
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
        }],
        remappings=[
            ("color/image_raw", "/camera/image_raw"),
        ],
        output="screen",
    )

    # ------------------------------------------------------------------
    # センサドライバ: LiDAR (Livox MID-360)
    # ------------------------------------------------------------------
    livox_node = Node(
        package="livox_ros_driver2",
        executable="livox_ros_driver2_node",
        name="livox_lidar_publisher",
        output="screen",
        parameters=[{
            "xfer_format": 0,       # PointCloud2 形式
            "multi_topic": 0,
            "data_src": 0,
            "publish_freq": 10.0,
            "output_data_type": 0,
        }],
    )

    # ------------------------------------------------------------------
    # 本体ノード
    # ------------------------------------------------------------------
    sync_preprocess = Node(
        package="lcfall_ros2",
        executable="sync_preprocess_node",
        name="sync_preprocess_node",
        parameters=[params_file],
        output="screen",
    )

    inference = Node(
        package="lcfall_ros2",
        executable="inference_node",
        name="inference_node",
        parameters=[params_file],
        output="screen",
    )

    alert = Node(
        package="lcfall_ros2",
        executable="alert_node",
        name="alert_node",
        output="screen",
    )

    # ------------------------------------------------------------------
    # 可視化ノード (条件付き起動)
    # ------------------------------------------------------------------
    visualization = Node(
        package="lcfall_ros2",
        executable="visualization_node",
        name="visualization_node",
        parameters=[params_file],
        output="screen",
        condition=IfCondition(enable_vis),
    )

    return LaunchDescription([
        params_file_arg,
        enable_vis_arg,
        # センサドライバ
        camera_node,
        livox_node,
        # 本体ノード
        sync_preprocess,
        inference,
        alert,
        visualization,
    ])
