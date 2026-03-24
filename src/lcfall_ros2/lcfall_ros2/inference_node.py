#!/usr/bin/env python3
"""inference_node: 48 フレーム蓄積 + heatmap 生成 + グローバル正規化 + 推論.

/preprocessed/frame を subscribe し、48 フレームの RingBuffer で蓄積。
inference_stride ごとに:
  1. skeleton_2d → heatmap (17, 48, 56, 56)
  2. pointcloud_frame → グローバル正規化 (48, 256, 3)
  3. CameraLiDARFusionModel で推論
  4. FallDetectionResult を /fall_detection/result に publish
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
import rclpy
from rclpy.node import Node

from lcfall_msgs.msg import PreprocessedFrame, FallDetectionResult

from lcfall_ros2.utils.ring_buffer import RingBuffer
from lcfall_ros2.utils.heatmap_generation import generate_heatmaps
from lcfall_ros2.utils.tensor_utils import (
    global_normalize_pointclouds,
    pointcloud_list_to_array,
    skeleton_list_to_array,
    to_torch_tensor,
)


# フレームデータ (skeleton + pointcloud のペア)
class FrameData:
    """1 フレーム分の前処理済みデータ."""

    __slots__ = ("skeleton_2d", "pointcloud_frame")

    def __init__(
        self,
        skeleton_2d: NDArray[np.float32],
        pointcloud_frame: NDArray[np.float32],
    ) -> None:
        self.skeleton_2d = skeleton_2d
        self.pointcloud_frame = pointcloud_frame


class InferenceNode(Node):
    """転倒検知推論ノード."""

    # 時系列窓長
    WINDOW_SIZE: int = 48

    def __init__(self) -> None:
        super().__init__("inference_node")

        # ==============================================================
        # パラメータ
        # ==============================================================
        self.declare_parameter("inference_stride", 10)
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("fall_decision_threshold", 0.35)

        # モデルチェックポイントパス (デフォルト値を設定)
        self.declare_parameter("camera_config_path", "/root/ros2_ws/install/lcfall_ros2/share/lcfall_ros2/config/slowonly_r50_inference.py")
        self.declare_parameter("camera_checkpoint_path", "/data/checkpoints/camera/best_model.pth")
        self.declare_parameter("lidar_checkpoint_path", "/data/checkpoints/lidar/best_model.pth")
        self.declare_parameter("fusion_checkpoint_path", "/data/checkpoints/fusion/best_model.pth")

        self._stride: int = (
            self.get_parameter("inference_stride").value
        )
        self._device: str = (
            self.get_parameter("device").value
        )
        self._fall_threshold: float = (
            self.get_parameter("fall_decision_threshold").value
        )
        self._camera_config_path: str = (
            self.get_parameter("camera_config_path").value
        )
        self._camera_checkpoint_path: str = (
            self.get_parameter("camera_checkpoint_path").value
        )
        self._lidar_checkpoint_path: str = (
            self.get_parameter("lidar_checkpoint_path").value
        )
        self._fusion_checkpoint_path: str = (
            self.get_parameter("fusion_checkpoint_path").value
        )

        # ==============================================================
        # リングバッファ (48 フレーム)
        # ==============================================================
        self._buffer: RingBuffer[FrameData] = RingBuffer(self.WINDOW_SIZE)

        # ==============================================================
        # 推論モデル
        # ==============================================================
        self._model = self._load_model()

        # ==============================================================
        # ROS 通信
        # ==============================================================
        self._sub = self.create_subscription(
            PreprocessedFrame,
            "/preprocessed/frame",
            self._frame_callback,
            10,
        )

        self._pub = self.create_publisher(
            FallDetectionResult,
            "/fall_detection/result",
            10,
        )

        self.get_logger().info(
            f"InferenceNode initialized "
            f"(window={self.WINDOW_SIZE}, stride={self._stride})"
        )

    # ==================================================================
    # モデル管理
    # ==================================================================

    def _load_model(self):
        """CameraLiDARFusionModel をロード.

        3 つのチェックポイント (Camera, LiDAR, FusionHead) が
        すべて設定されている場合のみモデルをロードする。
        いずれかが未設定の場合はダミー推論にフォールバック。

        Returns:
            CameraLiDARFusionModel またはNone（ダミー推論用）。
        """
        all_paths_set = all([
            self._camera_config_path,
            self._camera_checkpoint_path,
            self._lidar_checkpoint_path,
            self._fusion_checkpoint_path,
        ])

        if not all_paths_set:
            missing = []
            if not self._camera_config_path:
                missing.append("camera_config_path")
            if not self._camera_checkpoint_path:
                missing.append("camera_checkpoint_path")
            if not self._lidar_checkpoint_path:
                missing.append("lidar_checkpoint_path")
            if not self._fusion_checkpoint_path:
                missing.append("fusion_checkpoint_path")
            self.get_logger().warn(
                f"Missing model paths: {missing}. "
                f"Using dummy inference."
            )
            return None

        try:
            import torch
            from lcfall_ros2.models.fusion_model import (
                CameraLiDARFusionModel,
            )

            self.get_logger().info("Loading fusion model ...")
            self.get_logger().info(
                f"  Camera config:     {self._camera_config_path}"
            )
            self.get_logger().info(
                f"  Camera checkpoint: {self._camera_checkpoint_path}"
            )
            self.get_logger().info(
                f"  LiDAR checkpoint:  {self._lidar_checkpoint_path}"
            )
            self.get_logger().info(
                f"  Fusion checkpoint: {self._fusion_checkpoint_path}"
            )

            model = CameraLiDARFusionModel(
                camera_config_path=self._camera_config_path,
                camera_checkpoint_path=self._camera_checkpoint_path,
                lidar_checkpoint_path=self._lidar_checkpoint_path,
                fusion_checkpoint_path=self._fusion_checkpoint_path,
                num_classes=2,
                dropout=0.5,
                device=self._device,
            )

            model.to(self._device)
            model.eval()

            self.get_logger().info(
                "✓ Fusion model loaded successfully "
                f"(device={self._device})"
            )
            return model

        except Exception as e:
            self.get_logger().error(
                f"Failed to load model: {e}. "
                f"Falling back to dummy inference."
            )
            return None

    # ==================================================================
    # コールバック
    # ==================================================================

    def _frame_callback(self, msg: PreprocessedFrame) -> None:
        """前処理済みフレームを受信してバッファに蓄積."""
        skeleton = np.array(msg.skeleton_2d, dtype=np.float32)
        pointcloud = np.array(msg.pointcloud_frame, dtype=np.float32)

        frame = FrameData(skeleton, pointcloud)
        self._buffer.append(frame)

        # ログ: バッファ充填状況
        if not self._buffer.is_full:
            self.get_logger().info(
                f"Buffering: {self._buffer.count}/{self.WINDOW_SIZE}",
                throttle_duration_sec=2.0,
            )

        # 推論トリガー判定
        if self._buffer.should_infer(self._stride):
            self._run_inference(msg.header)

    # ==================================================================
    # 推論パイプライン
    # ==================================================================

    def _run_inference(self, header) -> None:
        """48 フレーム窓で推論を実行."""
        frames = self._buffer.get_ordered()
        if len(frames) < self.WINDOW_SIZE:
            return

        # skeleton 列 → (T, 51)
        skeleton_seq = skeleton_list_to_array(
            [f.skeleton_2d for f in frames]
        )

        # pointcloud 列 → (T, 256, 3)
        pointcloud_seq = pointcloud_list_to_array(
            [f.pointcloud_frame for f in frames]
        )

        # heatmap 生成: (17, 48, 56, 56)
        heatmaps = generate_heatmaps(skeleton_seq)

        # グローバル正規化: (48, 256, 3)
        normalized_pc = global_normalize_pointclouds(pointcloud_seq)

        # 推論実行
        prediction, confidence = self._infer(heatmaps, normalized_pc)

        # 結果 publish
        result_msg = FallDetectionResult()
        result_msg.header = header
        result_msg.prediction = int(prediction)
        result_msg.confidence = float(confidence)
        self._pub.publish(result_msg)

        label = "FALLING" if prediction == 1 else "NORMAL"
        self.get_logger().info(
            f"Inference result: {label} "
            f"(confidence={confidence:.3f})"
        )

    def _infer(
        self,
        heatmaps: NDArray[np.float32],
        pointclouds: NDArray[np.float32],
    ) -> tuple[int, float]:
        """融合モデルで推論.

        Args:
            heatmaps: (17, 48, 56, 56) heatmap テンソル。
            pointclouds: (48, 256, 3) 正規化済み点群。

        Returns:
            (prediction, confidence) タプル。
            prediction: 0=non-falling, 1=falling。
            confidence: [0.0, 1.0]。
        """
        if self._model is not None:
            return self._model_inference(heatmaps, pointclouds)
        else:
            return self._dummy_inference(heatmaps, pointclouds)

    def _model_inference(
        self,
        heatmaps: NDArray[np.float32],
        pointclouds: NDArray[np.float32],
    ) -> tuple[int, float]:
        """CameraLiDARFusionModel による推論.

        Args:
            heatmaps: (17, 48, 56, 56)
            pointclouds: (48, 256, 3)

        Returns:
            (prediction, confidence)
        """
        import torch

        # numpy → torch tensor, バッチ次元追加
        # heatmap: (17, 48, 56, 56) → (1, 17, 48, 56, 56)
        heatmap_tensor = torch.from_numpy(heatmaps).unsqueeze(0).float()
        # pointcloud: (48, 256, 3) → (1, 48, 256, 3)
        pc_tensor = torch.from_numpy(pointclouds).unsqueeze(0).float()

        # デバイスに転送
        heatmap_tensor = heatmap_tensor.to(self._device)
        pc_tensor = pc_tensor.to(self._device)

        with torch.no_grad():
            output = self._model(heatmap_tensor, pc_tensor)  # (1, 2)
            prob = torch.softmax(output, dim=1)  # (1, 2)

            # 学習時のラベル定義: class 0 = non-falling, class 1 = falling
            fall_prob = prob[0, 1].item()
            prediction = 1 if fall_prob > self._fall_threshold else 0
            confidence = fall_prob

        return prediction, confidence


    def _dummy_inference(
        self,
        heatmaps: NDArray[np.float32],
        pointclouds: NDArray[np.float32],
    ) -> tuple[int, float]:
        """ダミー推論 (開発・テスト用).

        常に non-falling を返す。

        Args:
            heatmaps: (17, 48, 56, 56)
            pointclouds: (48, 256, 3)

        Returns:
            (0, 0.0) = non-falling
        """
        return 0, 0.0


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
