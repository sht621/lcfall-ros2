"""Exact 3-input collection, with no second approximate synchronization."""
import time

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from fall_detection_msgs.msg import PoseFrame, SyncPairInfo, SynchronizedFrame
from .collector import Collector
from .common import Console, internal_qos, parameter, positive, run, stamp_ns, steady_timer


class FramePublisherNode(Node):
    def __init__(self):
        super().__init__('frame_publisher_node')
        self.console = Console(self)
        timeout = positive(parameter(self, 'pending_timeout_sec', 1.0), 'pending_timeout_sec')
        limits = {name: int(positive(parameter(self, name + '_buffer_size', 10), name))
                  for name in ('pairs', 'poses', 'points')}
        history = int(positive(parameter(self, 'duplicate_history_size', 256), 'duplicate_history_size'))
        if history < max(limits.values()):
            raise ValueError('duplicate_history_size must cover the pending buffers')
        # Start with empty volatile state. Source stamps need agree with each
        # other, not necessarily this process's wall clock. Restart the managed
        # pipeline together rather than inventing a source-time epoch fence.
        self.publisher = self.create_publisher(SynchronizedFrame, '/synchronized_frame', internal_qos())
        self.collector = Collector(limits, timeout, self.publish,
                                   lambda key, msg: self.console.show(key, msg, True), history)
        self.subscriptions_owned = [
            self.create_subscription(SyncPairInfo, '/sync/pair_info', self.pair, internal_qos()),
            self.create_subscription(PoseFrame, '/pose_frame', self.pose, internal_qos()),
            self.create_subscription(PointCloud2, '/points_roi', self.points, internal_qos()),
        ]
        self.expiry_timer = steady_timer(self, min(0.05, timeout / 4), self.expire)

    def key(self, stamp):
        return stamp_ns(stamp)

    def receive(self, kind, key_function, message):
        try:
            key = key_function()
        except ValueError as exc:
            self.console.show('stamp', str(exc), True)
            return
        self.collector.add(kind, key, message, time.monotonic())

    def pair(self, msg):
        self.receive('pairs', lambda: (self.key(msg.rgb_stamp), self.key(msg.lidar_stamp)), msg)

    def pose(self, msg):
        self.receive('poses', lambda: self.key(msg.header.stamp), msg)

    def points(self, msg):
        self.receive('points', lambda: self.key(msg.header.stamp), msg)

    def expire(self):
        self.collector.expire(time.monotonic())

    def publish(self, pose, points):
        self.publisher.publish(SynchronizedFrame(pose=pose, points=points))
        self.console.show('ok', 'SynchronizedFrame published (exact original stamp pair)')


def main(args=None):
    run(FramePublisherNode, args)
