#!/usr/bin/env python3
"""別PC可視化用に縮小JPEG画像を必要時だけ配信するノード."""

from __future__ import annotations

import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image


class RemoteImagePublisherNode(Node):
    """生カメラ画像をレート制限し、縮小JPEGとしてpublishする."""

    def __init__(self) -> None:
        super().__init__("remote_image_publisher_node")

        self.declare_parameter("input_topic", "/camera/image_raw")
        self.declare_parameter(
            "output_topic", "/camera/visualization/compressed"
        )
        self.declare_parameter("output_width", 640)
        self.declare_parameter("output_height", 360)
        self.declare_parameter("publish_fps", 10.0)
        self.declare_parameter("jpeg_quality", 70)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self._output_width = int(
            self.get_parameter("output_width").value
        )
        self._output_height = int(
            self.get_parameter("output_height").value
        )
        publish_fps = float(self.get_parameter("publish_fps").value)
        self._jpeg_quality = int(
            self.get_parameter("jpeg_quality").value
        )

        if self._output_width <= 0 or self._output_height <= 0:
            raise ValueError("output image dimensions must be positive")
        if publish_fps <= 0.0:
            raise ValueError("publish_fps must be greater than zero")
        if not 1 <= self._jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")

        self._minimum_interval = 1.0 / publish_fps
        self._last_publish_time = 0.0
        self._bridge = CvBridge()

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._publisher = self.create_publisher(
            CompressedImage, output_topic, qos
        )
        self.create_subscription(Image, input_topic, self._callback, qos)

        self.get_logger().info(
            "Remote image stream enabled "
            f"({self._output_width}x{self._output_height}, "
            f"JPEG quality={self._jpeg_quality}, {publish_fps:g} FPS, "
            f"topic={output_topic})."
        )

    def _callback(self, msg: Image) -> None:
        now = time.monotonic()
        if now - self._last_publish_time < self._minimum_interval:
            return

        try:
            image = self._bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
            resized = cv2.resize(
                image,
                (self._output_width, self._output_height),
                interpolation=cv2.INTER_AREA,
            )
            success, encoded = cv2.imencode(
                ".jpg",
                resized,
                [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
            )
            if not success:
                raise RuntimeError("cv2.imencode returned false")

            output = CompressedImage()
            output.header = msg.header
            output.format = "jpeg"
            output.data = np.asarray(encoded).tobytes()
            self._publisher.publish(output)
            self._last_publish_time = now
        except Exception as exc:
            self.get_logger().error(
                f"Remote image compression failed: {exc}",
                throttle_duration_sec=5.0,
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RemoteImagePublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
