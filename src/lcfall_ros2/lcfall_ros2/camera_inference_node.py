#!/usr/bin/env python3
"""Camera-only online inference node for debugging."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import rclpy
from rclpy.node import Node

from lcfall_msgs.msg import FallDetectionResult, PreprocessedFrame
from lcfall_ros2.utils.heatmap_generation import generate_heatmaps
from lcfall_ros2.utils.ring_buffer import RingBuffer
from lcfall_ros2.utils.tensor_utils import skeleton_list_to_array


class CameraFrameData:
    __slots__ = ("skeleton_2d",)

    def __init__(self, skeleton_2d: NDArray[np.float32]) -> None:
        self.skeleton_2d = skeleton_2d


class CameraInferenceNode(Node):
    WINDOW_SIZE: int = 48

    def __init__(self) -> None:
        super().__init__("camera_inference_node")

        self.declare_parameter("inference_stride", 10)
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter(
            "camera_config_path",
            "/root/ros2_ws/install/lcfall_ros2/share/lcfall_ros2/config/slowonly_r50_inference.py",
        )
        self.declare_parameter(
            "camera_checkpoint_path",
            "/data/checkpoints/camera/best_model.pth",
        )
        self.declare_parameter("fall_decision_threshold", 0.35)
        self.declare_parameter("camera_min_keypoint_score", 0.3)
        self.declare_parameter("camera_min_keypoints_per_frame", 5)
        self.declare_parameter("camera_min_valid_frames", 3)

        self._stride = self.get_parameter("inference_stride").value
        self._device = self.get_parameter("device").value
        self._camera_config_path = self.get_parameter("camera_config_path").value
        self._camera_checkpoint_path = self.get_parameter(
            "camera_checkpoint_path"
        ).value
        self._fall_threshold = self.get_parameter(
            "fall_decision_threshold"
        ).value
        self._min_keypoint_score = self.get_parameter(
            "camera_min_keypoint_score"
        ).value
        self._min_keypoints_per_frame = self.get_parameter(
            "camera_min_keypoints_per_frame"
        ).value
        self._min_valid_frames = self.get_parameter(
            "camera_min_valid_frames"
        ).value

        self._buffer: RingBuffer[CameraFrameData] = RingBuffer(self.WINDOW_SIZE)
        self._model = self._load_model()

        self._sub = self.create_subscription(
            PreprocessedFrame,
            "/preprocessed/frame",
            self._frame_callback,
            10,
        )
        self._pub = self.create_publisher(
            FallDetectionResult,
            "/fall_detection/camera_result",
            10,
        )

    def _load_model(self):
        try:
            import torch
            from mmaction.registry import MODELS
            from mmaction.utils import register_all_modules
            from mmengine.config import Config
            import mmaction.models  # noqa: F401

            register_all_modules(init_default_scope=True)
            cfg = Config.fromfile(self._camera_config_path)
            model = MODELS.build(cfg.model)
            checkpoint = torch.load(
                self._camera_checkpoint_path,
                map_location=self._device,
                weights_only=False,
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
            model.to(self._device)
            model.eval()
            return model
        except Exception as exc:
            self.get_logger().error(
                f"Failed to load camera-only model: {exc}"
            )
            return None

    def _frame_callback(self, msg: PreprocessedFrame) -> None:
        skeleton = np.array(msg.skeleton_2d, dtype=np.float32)
        self._buffer.append(CameraFrameData(skeleton))
        if self._buffer.should_infer(self._stride):
            self._run_inference(msg.header)

    def _run_inference(self, header) -> None:
        frames = self._buffer.get_ordered()
        if len(frames) < self.WINDOW_SIZE:
            return

        skeleton_seq = skeleton_list_to_array([f.skeleton_2d for f in frames])
        if not self._has_person_signal(skeleton_seq):
            self._publish_result(header, 0, 0.0)
            return

        heatmaps = generate_heatmaps(skeleton_seq)
        prediction, confidence = self._infer(heatmaps)
        self._publish_result(header, prediction, confidence)

    def _has_person_signal(self, skeleton_seq: NDArray[np.float32]) -> bool:
        valid_frames = 0
        reshaped = skeleton_seq.reshape(self.WINDOW_SIZE, 17, 3)
        for frame in reshaped:
            scores = frame[:, 2]
            if np.count_nonzero(scores >= self._min_keypoint_score) >= (
                self._min_keypoints_per_frame
            ):
                valid_frames += 1
        return valid_frames >= self._min_valid_frames

    def _infer(
        self, heatmaps: NDArray[np.float32]
    ) -> tuple[int, float]:
        if self._model is None:
            return 0, 0.0

        import torch

        heatmap_tensor = torch.from_numpy(heatmaps).unsqueeze(0).float()
        heatmap_tensor = heatmap_tensor.to(self._device)

        with torch.no_grad():
            features = self._model.backbone(heatmap_tensor)
            if isinstance(features, tuple):
                features = features[-1]
            output = self._model.cls_head(features)
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
    node = CameraInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
