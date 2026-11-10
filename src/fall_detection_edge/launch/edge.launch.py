"""Launch sensor drivers, four independent processors and optional service bridge."""
import math
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def enabled(context, name):
    value = LaunchConfiguration(name).perform(context).lower()
    if value not in ('true', 'false'):
        raise ValueError(f'{name} must be true or false')
    return value == 'true'


def make_nodes(context):
    params_path = LaunchConfiguration('params_file').perform(context)
    with open(params_path, encoding='utf-8') as stream:
        settings = yaml.safe_load(stream)
    with open(LaunchConfiguration('mount_file').perform(context), encoding='utf-8') as stream:
        mount = yaml.safe_load(stream)
    if mount['parent_frame'] != 'fall_detection_frame' or mount['child_frame'] == mount['parent_frame']:
        raise ValueError('invalid mounting frame configuration')
    sync = settings['sync_node']['ros__parameters']
    pose = settings['pose_estimation_node']['ros__parameters']
    nodes = []
    if enabled(context, 'start_drivers'):
        nodes += [Node(
            package='realsense2_camera', executable='realsense2_camera_node',
            namespace='camera', name='camera', output='screen', parameters=[{
                'camera_name': 'camera', 'enable_color': True,
                'rgb_camera.color_format': 'RGB8',
                'rgb_camera.color_profile': f"{pose['image_width']}x{pose['image_height']}x30",
                'enable_depth': False, 'enable_infra1': False, 'enable_infra2': False,
                'enable_gyro': False, 'enable_accel': False,
            }], remappings=[('/camera/camera/color/image_raw', sync['rgb_topic'])]), Node(
            package='livox_ros_driver2', executable='livox_ros_driver2_node',
            name='livox_lidar_publisher', output='screen', parameters=[{
                'xfer_format': 0, 'multi_topic': 0, 'data_src': 0,
                'publish_freq': 10.0, 'output_data_type': 0,
                'frame_id': mount['child_frame'],
                'user_config_path': LaunchConfiguration('livox_config_path').perform(context),
                'cmdline_input_bd_code': '',
            }], remappings=[('/livox/lidar', sync['lidar_topic'])])]
    if enabled(context, 'publish_mount_tf'):
        translation = mount['translation_m']
        rotation = [math.radians(value) for value in mount['rpy_degrees']]
        if len(translation) != 3 or len(rotation) != 3 or not all(
                math.isfinite(v) for v in translation + rotation):
            raise ValueError('mount translation and rotation must contain three finite values each')
        nodes.append(Node(
            package='tf2_ros', executable='static_transform_publisher',
            name='fall_detection_mount_tf', output='screen', arguments=[
                '--x', str(translation[0]), '--y', str(translation[1]), '--z', str(translation[2]),
                '--roll', str(rotation[0]), '--pitch', str(rotation[1]), '--yaw', str(rotation[2]),
                '--frame-id', mount['parent_frame'], '--child-frame-id', mount['child_frame'],
            ]))
    names = ['frame_publisher_node', 'lidar_preprocess_node', 'pose_estimation_node', 'sync_node']
    bridge_overrides = {}
    if enabled(context, 'enable_bridge'):
        names.append('topic_to_service_bridge_node')
        # Empty launch arguments preserve the YAML setting. Compose explicitly
        # selects the mode; these overrides apply only to the bridge process.
        for name in ('use_edgerm', 'edgerm_verbose'):
            if LaunchConfiguration(name).perform(context) != '':
                bridge_overrides[name] = enabled(context, name)
        for name in ('edgerm_url', 'edgerm_runtime_name', 'service_name'):
            value = LaunchConfiguration(name).perform(context)
            if value != '':
                bridge_overrides[name] = value
    nodes += [Node(package='fall_detection_edge', executable=name, name=name,
                   parameters=([params_path, bridge_overrides]
                               if name == 'topic_to_service_bridge_node' else [params_path]),
                   output='screen') for name in names]
    # Required process exits stop the whole pipeline. Restart deliberately after
    # correction, clearing correlation/request state instead of mixing epochs.
    handlers = [RegisterEventHandler(OnProcessExit(
        target_action=node, on_exit=[EmitEvent(event=Shutdown(
            reason='Required edge process exited; inspect console and restart the whole pipeline'))]))
        for node in nodes]
    return handlers + nodes


def generate_launch_description():
    share = get_package_share_directory('fall_detection_edge')
    livox_share = get_package_share_directory('livox_ros_driver2')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=os.path.join(share, 'config', 'edge.yaml')),
        DeclareLaunchArgument('mount_file', default_value=os.path.join(share, 'config', 'mount.yaml')),
        DeclareLaunchArgument('livox_config_path', default_value=os.environ.get(
            'LIVOX_CONFIG_PATH', os.path.join(livox_share, 'config', 'MID360_config.json'))),
        DeclareLaunchArgument('start_drivers', default_value='true'),
        DeclareLaunchArgument('publish_mount_tf', default_value='true'),
        DeclareLaunchArgument('enable_bridge', default_value='true'),
        DeclareLaunchArgument('use_edgerm', default_value=''),
        DeclareLaunchArgument('edgerm_url', default_value=''),
        DeclareLaunchArgument('edgerm_runtime_name', default_value=''),
        DeclareLaunchArgument('edgerm_verbose', default_value=''),
        DeclareLaunchArgument('service_name', default_value=''),
        OpaqueFunction(function=make_nodes),
    ])
