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
import json
import fcntl
import os
import socket
import struct
from ament_index_python.packages import get_package_share_directory
from lcfall_ros2.device_recovery import repair_realsense_video_nodes


def _get_local_ipv4_addresses() -> dict[str, str]:
    """Return IPv4 addresses keyed by interface name."""
    ipv4_by_ifname: dict[str, str] = {}
    for ifname in sorted(os.listdir("/sys/class/net")):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            packed = struct.pack("256s", ifname[:15].encode())
            ip_addr = socket.inet_ntoa(
                fcntl.ioctl(sock.fileno(), 0x8915, packed)[20:24]
            )
        except OSError:
            continue
        finally:
            sock.close()
        ipv4_by_ifname[ifname] = ip_addr
    return ipv4_by_ifname


def _warn_if_livox_host_ip_missing(config_path: str) -> None:
    """Print a clear diagnostic before Livox starts if host IP is unavailable."""
    try:
        with open(config_path, "r", encoding="utf-8") as fp:
            config = json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[lcfall.launch] Failed to read Livox config '{config_path}': {exc}")
        return

    host_net_info = config.get("MID360", {}).get("host_net_info", {})
    expected_host_ips = sorted({
        value
        for key, value in host_net_info.items()
        if key.endswith("_ip") and value
    })
    local_ipv4 = _get_local_ipv4_addresses()
    local_ip_values = set(local_ipv4.values())
    missing_host_ips = [ip for ip in expected_host_ips if ip not in local_ip_values]

    if not expected_host_ips:
        print(
            "[lcfall.launch] Livox host IP is not set in the JSON config. "
            "The driver may fail to bind UDP sockets."
        )
        return

    if missing_host_ips:
        local_summary = ", ".join(
            f"{ifname}={ip_addr}" for ifname, ip_addr in local_ipv4.items()
        ) or "none"
        missing_summary = ", ".join(missing_host_ips)
        print(
            "[lcfall.launch] WARNING: Livox host IP is not configured on this machine. "
            f"Expected: {missing_summary}. Local IPv4: {local_summary}. "
            "Livox may fail with 'bind failed'."
        )


def generate_launch_description():
    repaired_nodes = repair_realsense_video_nodes()
    if repaired_nodes:
        print(
            "[lcfall.launch] Repaired missing RealSense device nodes: "
            + ", ".join(repaired_nodes)
        )

    # パッケージ共有ディレクトリ
    pkg_share = get_package_share_directory("lcfall_ros2")
    default_params_file = os.path.join(pkg_share, "config", "params.yaml")
    livox_pkg_share = get_package_share_directory("livox_ros_driver2")
    default_livox_config_path = os.path.join(
        livox_pkg_share, "config", "MID360_config.json"
    )
    _warn_if_livox_host_ip_missing(default_livox_config_path)

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

    livox_config_arg = DeclareLaunchArgument(
        "livox_config_path",
        default_value=default_livox_config_path,
        description="Path to the Livox MID-360 JSON config",
    )

    params_file = LaunchConfiguration("params_file")
    enable_vis = LaunchConfiguration("enable_visualization")
    livox_config_path = LaunchConfiguration("livox_config_path")

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
            'enable_gyro': False,
            'enable_accel': False
        }],
        remappings=[
            ("camera/color/image_raw", "/camera/image_raw"),
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
            "user_config_path": livox_config_path,
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
        livox_config_arg,
        # センサドライバ
        camera_node,
        livox_node,
        # 本体ノード
        sync_preprocess,
        inference,
        alert,
        visualization,
    ])
