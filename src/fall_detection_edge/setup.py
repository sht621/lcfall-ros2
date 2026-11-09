from glob import glob
from setuptools import find_packages, setup

package_name = 'fall_detection_edge'
setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Edge pipeline maintainers',
    maintainer_email='maintainers@example.com',
    description='Pi edge observation pipeline, design v2.6',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        f'{name} = {package_name}.{name}:main' for name in (
            'sync_node', 'lidar_preprocess_node', 'pose_estimation_node',
            'frame_publisher_node', 'topic_to_service_bridge_node',
        )
    ]},
)
