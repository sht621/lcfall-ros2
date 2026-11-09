"""Static TF to robot-fixed fall_detection_frame, then inclusive fixed front ROI."""
import numpy as np
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener

from .common import Console, internal_qos, parameter, run, stamp_ns
from .pointcloud import make_cloud, read_xyz_tag, transform_xyz


class LidarPreprocessNode(Node):
    def __init__(self):
        super().__init__('lidar_preprocess_node')
        self.console = Console(self)
        self.target = parameter(self, 'target_frame', 'fall_detection_frame')
        if self.target != 'fall_detection_frame':
            raise ValueError('design v2.6 requires target_frame=fall_detection_frame')
        self.minimum = np.array([parameter(self, f'roi.{axis}_min', value)
                                 for axis, value in zip('xyz', (1.0, -2.0, 0.0))])
        self.maximum = np.array([parameter(self, f'roi.{axis}_max', value)
                                 for axis, value in zip('xyz', (8.0, 2.0, 2.0))])
        if not (np.isfinite(self.minimum).all() and np.isfinite(self.maximum).all()
                and (self.minimum <= self.maximum).all()):
            raise ValueError('ROI bounds must be finite and min <= max')
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.publisher = self.create_publisher(PointCloud2, '/points_roi', internal_qos())
        self.subscription = self.create_subscription(
            PointCloud2, '/sync/points', self.receive, internal_qos())

    def receive(self, cloud):
        try:
            stamp_ns(cloud.header.stamp)
            if not cloud.header.frame_id:
                raise ValueError('missing LiDAR frame_id')
            xyz, tags = read_xyz_tag(cloud)
            # Even a zero-point input must have a valid transform/contract.
            transform = self.buffer.lookup_transform(
                self.target, cloud.header.frame_id, Time.from_msg(cloud.header.stamp),
                timeout=Duration(seconds=0.0))
            xyz = transform_xyz(xyz, transform.transform)
            mask = (np.isfinite(xyz).all(axis=1)
                    & (xyz >= self.minimum).all(axis=1)
                    & (xyz <= self.maximum).all(axis=1))
            result = make_cloud(cloud.header, self.target, xyz[mask], tags[mask])
            self.publisher.publish(result)
            self.console.show('ok', f'ROI result published: {result.width} points (zero is valid)')
        except Exception as exc:
            self.console.show('failure', f'LiDAR preprocessing failed; no output: {exc}', True)


def main(args=None):
    run(LidarPreprocessNode, args)
