#!/usr/bin/env python3
"""inference_node: 48 フレーム蓄積 + heatmap 生成 + グローバル正規化 + 推論.

/preprocessed/frame を subscribe し、48 フレームの RingBuffer で蓄積。
inference_stride ごとに:
  1. skeleton_2d → heatmap (17, 48, 56, 56)
  2. pointcloud_frame → グローバル正規化 (48, 256, 3)
  3. 融合モデルで推論 (現在はダミー)
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
        self.declare_parameter("model_path", "")
        self.declare_parameter("device", "cuda:0")

        self._stride: int = (
            self.get_parameter("inference_stride").value
        )
        self._model_path: str = (
            self.get_parameter("model_path").value
        )
        self._device: str = (
            self.get_parameter("device").value
        )

        # ==============================================================
        # リングバッファ (48 フレーム)
        # ==============================================================
        self._buffer: RingBuffer[FrameData] = RingBuffer(self.WINDOW_SIZE)

        # ==============================================================
        # 推論モデル (将来差し替え)
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
        """推論モデルをロード.

        Returns:
            モデルオブジェクト。未確定のため None を返す。
        """
        if self._model_path:
            self.get_logger().info(
                f"Loading model from {self._model_path} ..."
            )
            # TODO: 実際のモデルロード処理
            # model = torch.load(self._model_path, map_location=self._device)
            # model.eval()
            # return model
            self.get_logger().warn(
                "Model loading not yet implemented. Using dummy inference."
            )
        else:
            self.get_logger().warn(
                "No model_path specified. Using dummy inference."
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
        """実際のモデル推論.

        TODO: 融合モデルの実装後に差し替える。

        Args:
            heatmaps: (17, 48, 56, 56)
            pointclouds: (48, 256, 3)

        Returns:
            (prediction, confidence)
        """
        # TODO: 以下のような処理を実装する
        # heatmap_tensor = to_torch_tensor(heatmaps).unsqueeze(0)
        # pc_tensor = to_torch_tensor(pointclouds).unsqueeze(0)
        # with torch.no_grad():
        #     output = self._model(heatmap_tensor, pc_tensor)
        #     prob = torch.softmax(output, dim=1)
        #     confidence = prob[0, 1].item()
        #     prediction = 1 if confidence > 0.5 else 0
        # return prediction, confidence
        self.get_logger().warn(
            "Model inference not implemented. Falling back to dummy."
        )
        return self._dummy_inference(heatmaps, pointclouds)

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
