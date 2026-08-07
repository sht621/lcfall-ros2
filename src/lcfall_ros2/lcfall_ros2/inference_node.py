#!/usr/bin/env python3
"""inference_node: 48 フレーム蓄積 + heatmap 生成 + グローバル正規化 + 推論.

/preprocessed/frame を subscribe し、48 フレームの RingBuffer で蓄積。
inference_stride ごとに:
  1. skeleton_2d → heatmap (17, 48, 64, 64)
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
    POSTURE_WINDOW_SIZE: int = 12
    SENSOR_VALIDITY_WINDOW_SIZE: int = 3
    MIN_VALID_KEYPOINTS: int = 8
    MIN_UNIQUE_LIDAR_POINTS: int = 20

    def __init__(self) -> None:
        super().__init__("inference_node")

        # ==============================================================
        # パラメータ
        # ==============================================================
        self.declare_parameter("inference_stride", 10)
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("fall_decision_threshold", 0.5)
        self.declare_parameter("image_width", 1280)
        self.declare_parameter("image_height", 720)

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
        self._image_width: int = self.get_parameter("image_width").value
        self._image_height: int = self.get_parameter("image_height").value
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

        3 つのチェックポイント (Camera, LiDAR, Fusion全体) が
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

        # 学習時と同じPoseCompact済みheatmapを生成: (17, 48, 64, 64)
        heatmaps = generate_heatmaps(
            skeleton_seq,
            image_width=self._image_width,
            image_height=self._image_height,
        )

        # グローバル正規化: (48, 256, 3)
        normalized_pc = global_normalize_pointclouds(pointcloud_seq)

        # 学習モデルと実機姿勢判定を併用する。学習データとの環境差で
        # fusion出力が過度にNORMALへ偏る場合も、継続した横臥姿勢を検知する。
        _, learned_confidence = self._infer(heatmaps, normalized_pc)
        posture_confidence = self._posture_fall_probability(skeleton_seq)
        confidence = max(learned_confidence, posture_confidence)

        # カメラ姿勢推定が継続して利用できず、ROI内には点群がある場合に
        # 限ってLiDARの形状ルールでモデルの見逃しを補う。モデル入力と
        # 推論処理は変更せず、モデルが閾値以上ならLiDAR補助も適用しない。
        lidar_posture_confidence = 0.0
        lidar_rule_enabled = (
            learned_confidence < self._fall_threshold
            and self._camera_is_unusable(skeleton_seq)
            and self._lidar_person_is_present(pointcloud_seq)
        )
        if lidar_rule_enabled:
            lidar_posture_confidence = (
                self._lidar_posture_fall_probability(pointcloud_seq)
            )
            confidence = max(confidence, lidar_posture_confidence)
        prediction = confidence > self._fall_threshold

        # 結果 publish
        result_msg = FallDetectionResult()
        result_msg.header = header
        result_msg.prediction = bool(prediction)
        result_msg.confidence = float(confidence)
        self._pub.publish(result_msg)

        label = "FALLING" if prediction else "NORMAL"
        self.get_logger().info(
            f"Inference result: {label} "
            f"(confidence={confidence:.6f}, "
            f"model={learned_confidence:.6f}, "
            f"posture={posture_confidence:.6f}, "
            f"lidar_posture={lidar_posture_confidence:.6f}, "
            f"lidar_rule={'on' if lidar_rule_enabled else 'off'})"
        )

    @staticmethod
    def _posture_fall_probability(
        skeleton_sequence: NDArray[np.float32],
    ) -> float:
        """体幹角度と骨格BBox縦横比から継続した横臥姿勢を計算する."""
        frame_scores: list[float] = []
        # 約2秒分に限定し、過去の転倒姿勢を長く保持しない。
        for frame in skeleton_sequence[
            -InferenceNode.POSTURE_WINDOW_SIZE:
        ]:
            keypoints = frame.reshape(17, 3)

            # COCO keypoints: left/right shoulder=5/6, left/right hip=11/12。
            # 腕や脚を含む全身の外接幅は、腕を横に広げただけでも横長に
            # なるため使わない。両肩中点から両腰中点への体幹軸だけを見る。
            torso_indices = np.asarray([5, 6, 11, 12])
            torso = keypoints[torso_indices]
            if np.any(torso[:, 2] < 0.4):
                frame_scores.append(0.0)
                continue

            shoulder_midpoint = torso[0:2, :2].mean(axis=0)
            hip_midpoint = torso[2:4, :2].mean(axis=0)
            torso_vector = shoulder_midpoint - hip_midpoint
            horizontal = abs(float(torso_vector[0]))
            vertical = abs(float(torso_vector[1]))
            torso_length = float(np.hypot(horizontal, vertical))
            if torso_length < 1.0e-3:
                frame_scores.append(0.0)
                continue

            angle_deg = float(np.degrees(np.arctan2(horizontal, vertical)))

            # 実機の通常姿勢で観測した揺れ（おおむね0〜22度）は低く抑え、
            # 転倒方向へ傾くと明確に上がるS字カーブにする。
            # 10度以下=0、35度=0.5、60度以上=1。
            normalized_angle = float(
                np.clip((angle_deg - 10.0) / 50.0, 0.0, 1.0)
            )
            score = normalized_angle * normalized_angle * (
                3.0 - 2.0 * normalized_angle
            )

            # 物を拾う動作では体幹だけが水平に近づく一方、脚を含む骨格BBoxは
            # 縦長のままになる。逆に腕を広げるとBBoxだけが横長になる。
            # 両条件を満たした場合だけ高くなるよう、小さい側を採用する。
            valid = keypoints[:, 2] >= 0.4
            if int(valid.sum()) < 8:
                frame_scores.append(0.0)
                continue

            xy = keypoints[valid, :2]
            bbox_width = float(np.ptp(xy[:, 0]))
            bbox_height = float(np.ptp(xy[:, 1]))
            if bbox_width < 1.0e-3:
                frame_scores.append(0.0)
                continue

            height_width_ratio = bbox_height / bbox_width
            # H/W >= 1.25: 縦長で0、H/W <= 0.60: 横長で1。
            bbox_score = float(
                np.clip((1.25 - height_width_ratio) / 0.65, 0.0, 1.0)
            )
            frame_scores.append(min(score, bbox_score))

        if len(frame_scores) < 4:
            return 0.0
        # 4番目に高い値を使い、短時間のキーポイント崩れでは発火させない。
        return float(np.partition(np.asarray(frame_scores), -4)[-4])

    @staticmethod
    def _skeleton_frame_is_valid(frame: NDArray[np.float32]) -> bool:
        """カメラ姿勢判定に使える骨格が含まれるかを返す."""
        if frame.size != 17 * 3 or not np.all(np.isfinite(frame)):
            return False

        keypoints = frame.reshape(17, 3)
        valid = keypoints[:, 2] >= 0.4
        if int(valid.sum()) < InferenceNode.MIN_VALID_KEYPOINTS:
            return False

        # カメラpostureと同じく、両肩・両腰がすべて見えることを要求する。
        torso_indices = np.asarray([5, 6, 11, 12])
        return bool(np.all(valid[torso_indices]))

    @staticmethod
    def _camera_is_unusable(
        skeleton_sequence: NDArray[np.float32],
    ) -> bool:
        """直近フレームで骨格を継続して取得できないかを返す."""
        recent = skeleton_sequence[
            -InferenceNode.SENSOR_VALIDITY_WINDOW_SIZE:
        ]
        if len(recent) < InferenceNode.SENSOR_VALIDITY_WINDOW_SIZE:
            return False
        return not any(
            InferenceNode._skeleton_frame_is_valid(frame)
            for frame in recent
        )

    @staticmethod
    def _valid_lidar_points(
        frame: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """整形済みROI点群から有効なユニーク点だけを返す."""
        if frame.size == 0 or frame.size % 3 != 0:
            return np.empty((0, 3), dtype=np.float32)

        points = frame.reshape(-1, 3)
        finite = np.all(np.isfinite(points), axis=1)
        nonzero = np.any(np.abs(points) > 1.0e-6, axis=1)
        points = points[finite & nonzero]
        if len(points) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # 点不足時の重複サンプリングを人物点数として数えない。
        return np.unique(points, axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _lidar_frame_has_person_points(
        frame: NDArray[np.float32],
    ) -> bool:
        """手動ROI内に人物候補として十分な点があるかを返す."""
        points = InferenceNode._valid_lidar_points(frame)
        return len(points) >= InferenceNode.MIN_UNIQUE_LIDAR_POINTS

    @staticmethod
    def _lidar_person_is_present(
        pointcloud_sequence: NDArray[np.float32],
    ) -> bool:
        """直近3フレーム中2フレーム以上にROI点群があるかを返す."""
        recent = pointcloud_sequence[
            -InferenceNode.SENSOR_VALIDITY_WINDOW_SIZE:
        ]
        if len(recent) < InferenceNode.SENSOR_VALIDITY_WINDOW_SIZE:
            return False
        valid_frames = sum(
            InferenceNode._lidar_frame_has_person_points(frame)
            for frame in recent
        )
        return valid_frames >= 2

    @staticmethod
    def _lidar_posture_fall_probability(
        pointcloud_sequence: NDArray[np.float32],
    ) -> float:
        """ROI点群の高さと水平長から継続した横臥姿勢を計算する."""
        frame_scores: list[float] = []
        for frame in pointcloud_sequence[
            -InferenceNode.POSTURE_WINDOW_SIZE:
        ]:
            points = InferenceNode._valid_lidar_points(frame)
            if len(points) < InferenceNode.MIN_UNIQUE_LIDAR_POINTS:
                frame_scores.append(0.0)
                continue

            # 高さは外れ値に影響されにくい5–95%幅を使う。
            height = float(
                np.percentile(points[:, 2], 95)
                - np.percentile(points[:, 2], 5)
            )

            # XY平面の主軸へ投影し、前後・左右・斜めの転倒を同じように
            # 評価できる方向非依存の水平長を求める。
            xy = points[:, :2]
            xy_centered = xy - np.median(xy, axis=0)
            covariance = np.cov(xy_centered, rowvar=False)
            if covariance.shape != (2, 2) or not np.all(
                np.isfinite(covariance)
            ):
                frame_scores.append(0.0)
                continue

            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            main_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
            projected = xy_centered @ main_axis
            horizontal = float(
                np.percentile(projected, 95)
                - np.percentile(projected, 5)
            )
            if horizontal < 1.0e-3:
                frame_scores.append(0.0)
                continue

            # カメラBBoxと同じ比率: H/W >= 1.25で0、<= 0.60で1。
            # S字カーブにして、確率が不自然に跳ねないよう連続化する。
            height_width_ratio = height / horizontal
            normalized = float(
                np.clip((1.25 - height_width_ratio) / 0.65, 0.0, 1.0)
            )
            score = normalized * normalized * (3.0 - 2.0 * normalized)
            frame_scores.append(score)

        if len(frame_scores) < 4:
            return 0.0
        # カメラpostureと同様、4フレーム未満の点群崩れを無視する。
        return float(np.partition(np.asarray(frame_scores), -4)[-4])

    def _infer(
        self,
        heatmaps: NDArray[np.float32],
        pointclouds: NDArray[np.float32],
    ) -> tuple[bool, float]:
        """融合モデルで推論.

        Args:
            heatmaps: (17, 48, 64, 64) heatmap テンソル。
            pointclouds: (48, 256, 3) 正規化済み点群。

        Returns:
            (prediction, confidence) タプル。
            prediction: false=non-falling, true=falling。
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
    ) -> tuple[bool, float]:
        """CameraLiDARFusionModel による推論.

        Args:
            heatmaps: (17, 48, 64, 64)
            pointclouds: (48, 256, 3)

        Returns:
            (prediction, confidence)
        """
        import torch

        # numpy → torch tensor, バッチ次元追加
        # heatmap: (17, 48, 64, 64) → (1, 17, 48, 64, 64)
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
            prediction = fall_prob > self._fall_threshold
            confidence = fall_prob

        return prediction, confidence


    def _dummy_inference(
        self,
        heatmaps: NDArray[np.float32],
        pointclouds: NDArray[np.float32],
    ) -> tuple[bool, float]:
        """ダミー推論 (開発・テスト用).

        常に non-falling を返す。

        Args:
            heatmaps: (17, 48, 64, 64)
            pointclouds: (48, 256, 3)

        Returns:
            (False, 0.0) = non-falling
        """
        return False, 0.0


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
