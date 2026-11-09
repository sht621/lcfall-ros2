"""Perform the pipeline's only approximate synchronization on raw sensor stamps."""
import time

import message_filters
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from fall_detection_msgs.msg import SyncPairInfo
from .common import Console, internal_qos, parameter, positive, run, stamp_ns, steady_timer


class CheckedSubscriber(message_filters.SimpleFilter):
    def __init__(self, node, msg_type, topic, qos, received):
        super().__init__()
        self.subscription = node.create_subscription(msg_type, topic, self.receive, qos)
        self.received = received
        self.console = node.console

    def receive(self, message):
        self.received()
        try:
            stamp_ns(message.header.stamp)
        except ValueError as exc:
            self.console.show('bad_stamp', str(exc), warning=True)
            return
        self.signalMessage(message)


class SyncNode(Node):
    def __init__(self):
        super().__init__('sync_node')
        self.console = Console(self)
        self.counts = [0, 0, 0]
        self.last_pair = time.monotonic()
        queue = int(positive(parameter(self, 'queue_size', 10), 'queue_size'))
        slop = positive(parameter(self, 'slop_sec', 0.0167), 'slop_sec')
        self.image_pub = self.create_publisher(Image, '/sync/image', internal_qos())
        self.points_pub = self.create_publisher(PointCloud2, '/sync/points', internal_qos())
        self.pair_pub = self.create_publisher(SyncPairInfo, '/sync/pair_info', internal_qos())
        self.image_sub = CheckedSubscriber(
            self, Image, parameter(self, 'rgb_topic', '/camera/camera/color/image_raw'),
            self.driver_qos('rgb'), lambda: self.increment(0))
        self.points_sub = CheckedSubscriber(
            self, PointCloud2, parameter(self, 'lidar_topic', '/livox/lidar'),
            self.driver_qos('lidar'), lambda: self.increment(1))
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.points_sub], queue_size=queue, slop=slop,
            allow_headerless=False)
        self.sync.registerCallback(self.paired)
        self.status_timer = steady_timer(self, 5.0, self.status)

    def driver_qos(self, prefix):
        reliability = parameter(self, prefix + '_reliability', 'best_effort')
        if reliability not in ('best_effort', 'reliable'):
            raise ValueError(f'{prefix}_reliability: use best_effort or reliable')
        # Best-effort request is compatible with both reliable and best-effort offers.
        return QoSProfile(
            depth=int(positive(parameter(self, prefix + '_depth', 10), prefix + '_depth')),
            history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE,
            reliability=(ReliabilityPolicy.RELIABLE if reliability == 'reliable'
                         else ReliabilityPolicy.BEST_EFFORT))

    def increment(self, index):
        self.counts[index] += 1

    def paired(self, image, points):
        pair = SyncPairInfo(rgb_stamp=image.header.stamp, lidar_stamp=points.header.stamp)
        self.pair_pub.publish(pair)
        self.image_pub.publish(image)
        self.points_pub.publish(points)
        self.increment(2)
        self.last_pair = time.monotonic()
        self.console.show('paired', 'ATS pair published with original RGB/LiDAR stamps')

    def status(self):
        if time.monotonic() - self.last_pair >= 5.0:
            self.console.show('no_pair',
                              f'No ATS pair for 5s; received RGB/LiDAR/pairs={self.counts}',
                              warning=True)


def main(args=None):
    run(SyncNode, args)
