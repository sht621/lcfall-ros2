#!/usr/bin/env python3
"""LiDAR-only online inference node for debugging."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import rclpy
from rclpy.node import Node

from lcfall_msgs.msg import FallDetectionResult, PreprocessedFrame
from lcfall_ros2.models.lidar_model import PointNet2GRUModel
from lcfall_ros2.utils.ring_buffer import RingBuffer
from lcfall_ros2.utils.tensor_utils import (
    global_normalize_pointclouds,
    pointcloud_list_to_array,
)


class LiDARFrameData:
    __slots__ = ("pointcloud_frame",)

    def __init__(self, pointcloud_frame: NDArray[np.float32]) -> None:
        self.pointcloud_frame = pointcloud_frame


class LiDARInferenceNode(Node):
    WINDOW_SIZE: int = 48

    def __init__(self) -> None:
        super().__init__("lidar_inference_node")

        self.declare_parameter("inference_stride", 10)
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("fall_decision_threshold", 0.35)
        self.declare_parameter(
            "lidar_checkpoint_path",
            "/data/checkpoints/lidar/best_model.pth",
        )

        self._stride = self.get_parameter("inference_stride").value
        self._device = self.get_parameter("device").value
        self._fall_threshold = self.get_parameter(
            "fall_decision_threshold"
        ).value
        self._lidar_checkpoint_path = self.get_parameter(
            "lidar_checkpoint_path"
        ).value

        self._buffer: RingBuffer[LiDARFrameData] = RingBuffer(self.WINDOW_SIZE)
        self._model = self._load_model()

        self._sub = self.create_subscription(
            PreprocessedFrame,
            "/preprocessed/frame",
            self._frame_callback,
            10,
        )
        self._pub = self.create_publisher(
            FallDetectionResult,
            "/fall_detection/lidar_result",
            10,
        )

    def _load_model(self):
        try:
            import torch

            model = PointNet2GRUModel(
                num_classes=2,
                num_points=256,
                hidden_size=512,
                num_gru_layers=2,
                dropout=0.5,
            )
            checkpoint = torch.load(
                self._lidar_checkpoint_path,
                map_location=self._device,
                weights_only=False,
            )
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
            model.to(self._device)
            model.eval()
            return model
        except Exception as exc:
            self.get_logger().error(
                f"Failed to load lidar-only model: {exc}"
            )
            return None

    def _frame_callback(self, msg: PreprocessedFrame) -> None:
        pointcloud = np.array(msg.pointcloud_frame, dtype=np.float32)
        self._buffer.append(LiDARFrameData(pointcloud))
        if self._buffer.should_infer(self._stride):
            self._run_inference(msg.header)

    def _run_inference(self, header) -> None:
        frames = self._buffer.get_ordered()
        if len(frames) < self.WINDOW_SIZE:
            return

        pointcloud_seq = pointcloud_list_to_array(
            [f.pointcloud_frame for f in frames]
        )
        if np.allclose(pointcloud_seq, 0.0):
            self._publish_result(header, 0, 0.0)
            return

        normalized_pc = global_normalize_pointclouds(pointcloud_seq)
        prediction, confidence = self._infer(normalized_pc)
        self._publish_result(header, prediction, confidence)

    def _infer(
        self, pointclouds: NDArray[np.float32]
    ) -> tuple[int, float]:
        if self._model is None:
            return 0, 0.0

        import torch

        pc_tensor = torch.from_numpy(pointclouds).unsqueeze(0).float()
        pc_tensor = pc_tensor.to(self._device)

        with torch.no_grad():
            output = self._model(pc_tensor)
            prob = torch.softmax(output, dim=1)
            fall_prob = prob[0, 1].item()
            prediction = 1 if fall_prob > self._fall_threshold else 0
            confidence = fall_prob
        return prediction, confidence

    def _publish_result(
        self, header, prediction: int, confidence: float
    ) -> None:
        msg = FallDetectionResult()
        msg.header = header
        msg.prediction = int(prediction)
        msg.confidence = float(confidence)
        self._pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LiDARInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
