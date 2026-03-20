"""capture_background.launch.py: 背景モデル取得 (1 コマンドで完結).

LiDAR ドライバ + capture_background ノードを起動し、
背景モデル取得完了後にすべてのノードを自動終了する。

使い方:
    ros2 launch lcfall_ros2 capture_background.launch.py

パラメータ指定例:
    ros2 launch lcfall_ros2 capture_background.launch.py \
        capture_frames:=50 voxel_size:=0.08 min_hits:=15
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    RegisterEventHandler,
    Shutdown,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ------------------------------------------------------------------
    # Launch 引数
    # ------------------------------------------------------------------
    args = [
        DeclareLaunchArgument("lidar_topic", default_value="/livox/lidar"),
        DeclareLaunchArgument(
            "output_path",
            default_value="/data/background/background_voxel_map.npz",
        ),
        DeclareLaunchArgument("capture_frames", default_value="30"),
        DeclareLaunchArgument("voxel_size", default_value="0.10"),
        DeclareLaunchArgument("min_hits", default_value="10"),
        DeclareLaunchArgument("roi_x_min", default_value="0.0"),
        DeclareLaunchArgument("roi_x_max", default_value="5.0"),
        DeclareLaunchArgument("roi_y_min", default_value="-2.0"),
        DeclareLaunchArgument("roi_y_max", default_value="2.0"),
        DeclareLaunchArgument("roi_z_min", default_value="0.1"),
        DeclareLaunchArgument("roi_z_max", default_value="2.0"),
    ]

    # ------------------------------------------------------------------
    # LiDAR ドライバ (Livox MID-360)
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
    # 背景モデル取得ノード
    # ------------------------------------------------------------------
    capture_node = Node(
        package="lcfall_ros2",
        executable="capture_background",
        name="capture_background_node",
        output="screen",
        parameters=[{
            "lidar_topic": LaunchConfiguration("lidar_topic"),
            "output_path": LaunchConfiguration("output_path"),
            "capture_frames": LaunchConfiguration("capture_frames"),
            "voxel_size": LaunchConfiguration("voxel_size"),
            "min_hits": LaunchConfiguration("min_hits"),
            "roi_x_min": LaunchConfiguration("roi_x_min"),
            "roi_x_max": LaunchConfiguration("roi_x_max"),
            "roi_y_min": LaunchConfiguration("roi_y_min"),
            "roi_y_max": LaunchConfiguration("roi_y_max"),
            "roi_z_min": LaunchConfiguration("roi_z_min"),
            "roi_z_max": LaunchConfiguration("roi_z_max"),
        }],
    )

    # ------------------------------------------------------------------
    # capture_node 終了時に launch 全体を自動シャットダウン
    # ------------------------------------------------------------------
    shutdown_on_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=capture_node,
            on_exit=[Shutdown(reason="Background capture completed.")],
        )
    )

    return LaunchDescription([
        *args,
        livox_node,
        capture_node,
        shutdown_on_exit,
    ])
