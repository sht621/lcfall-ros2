from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'lcfall_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament index registration
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # package.xml
        ('share/' + package_name, ['package.xml']),
        # launch files
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # config files
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Camera + LiDAR Late Fusion fall detection system for ROS2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sync_preprocess_node = lcfall_ros2.sync_preprocess_node:main',
            'inference_node = lcfall_ros2.inference_node:main',
            'alert_node = lcfall_ros2.alert_node:main',
            'visualization_node = lcfall_ros2.visualization_node:main',
            'capture_background = lcfall_ros2.capture_background:main',
        ],
    },
)
