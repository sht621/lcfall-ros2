#!/usr/bin/env python3
"""Minimal ROS 2 Jazzy subscriber used for cross-distribution DDS testing."""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from lcfall_msgs.msg import FallDetectionResult


class FallResultSubscriber(Node):
    def __init__(self) -> None:
        super().__init__("jazzy_fall_result_subscriber")
        self._count = 0
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._subscription = self.create_subscription(
            FallDetectionResult,
            "/fall_detection/result",
            self._on_result,
            qos,
        )
        self.get_logger().info(
            "Waiting for /fall_detection/result (Reliable, KeepLast(10))"
        )

    def _on_result(self, msg: FallDetectionResult) -> None:
        self._count += 1
        self.get_logger().info(
            "RECEIVED count=%d stamp=%d.%09d prediction=%s confidence=%.6f"
            % (
                self._count,
                msg.header.stamp.sec,
                msg.header.stamp.nanosec,
                str(msg.prediction),
                msg.confidence,
            )
        )


def main() -> None:
    rclpy.init()
    node = FallResultSubscriber()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
