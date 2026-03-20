#!/usr/bin/env python3
"""visualization_node: デモ用可視化ノード.

1 ウィンドウ内の左右 2 画面表示:
  左: カメラ画像 + skeleton overlay (緑)
  右: 点群の正面投影 (緑)

転倒検知結果:
  - prediction == 1: FALLING を赤文字で表示
  - 48 フレーム蓄積前: Waiting for 48 frames + バッファ数表示
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, PointCloud2

from lcfall_msgs.msg import FallDetectionResult, PreprocessedFrame

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from cv_bridge import CvBridge
    HAS_CVBRIDGE = True
except ImportError:
    HAS_CVBRIDGE = False


# COCO 17 keypoint skeleton 接続定義
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # 顔
    (5, 6), (5, 7), (7, 9), (6, 8),          # 上半身
    (8, 10), (5, 11), (6, 12), (11, 12),      # 体幹
    (11, 13), (13, 15), (12, 14), (14, 16),   # 下半身
]

# Skeleton のキーポイント数 / 次元
NUM_KEYPOINTS = 17
KEYPOINT_DIMS = 3

# 点群設定
NUM_POINTS = 256

# 表示設定
VIS_WIDTH = 640
VIS_HEIGHT = 480
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# skeleton 描画しきい値
SKELETON_SCORE_THRESHOLD = 0.3


class VisualizationNode(Node):
    """デモ用可視化ノード."""

    def __init__(self) -> None:
        super().__init__("visualization_node")

        if not HAS_CV2:
            self.get_logger().error(
                "OpenCV (cv2) is required for visualization_node."
            )
            raise SystemExit("cv2 not available")

        # ==============================================================
        # パラメータ
        # ==============================================================
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("lidar_topic", "/livox/lidar")
        self.declare_parameter("image_width", 1280)
        self.declare_parameter("image_height", 720)
        self.declare_parameter("window_name", "LCFall ROS2 Demo")

        camera_topic: str = self.get_parameter("camera_topic").value
        lidar_topic: str = self.get_parameter("lidar_topic").value
        self._image_width: int = self.get_parameter("image_width").value
        self._image_height: int = self.get_parameter("image_height").value
        self._window_name: str = self.get_parameter("window_name").value

        # ==============================================================
        # 状態変数
        # ==============================================================
        self._latest_image: Optional[NDArray[np.uint8]] = None
        self._latest_points: Optional[NDArray[np.float32]] = None
        self._latest_result: Optional[FallDetectionResult] = None
        self._latest_skeleton: Optional[NDArray[np.float32]] = None
        self._latest_preprocessed_pc: Optional[NDArray[np.float32]] = None
        self._frame_count: int = 0
        self._bridge = CvBridge() if HAS_CVBRIDGE else None

        # ==============================================================
        # ROS 通信
        # ==============================================================
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.create_subscription(
            Image, camera_topic, self._image_callback, qos
        )
        self.create_subscription(
            PointCloud2, lidar_topic, self._lidar_callback, qos
        )
        self.create_subscription(
            FallDetectionResult,
            "/fall_detection/result",
            self._result_callback,
            10,
        )
        # 前処理済みフレーム (skeleton + 前景点群)
        self.create_subscription(
            PreprocessedFrame,
            "/preprocessed/frame",
            self._preprocessed_callback,
            10,
        )

        # 描画タイマー (30 FPS)
        self.create_timer(1.0 / 30.0, self._draw)

        self.get_logger().info("VisualizationNode initialized.")

    # ==================================================================
    # コールバック
    # ==================================================================

    def _image_callback(self, msg: Image) -> None:
        """カメラ画像を更新."""
        try:
            if self._bridge is not None:
                self._latest_image = self._bridge.imgmsg_to_cv2(
                    msg, desired_encoding="bgr8"
                )
            else:
                data = np.frombuffer(msg.data, dtype=np.uint8)
                image = data.reshape((msg.height, msg.width, 3))
                if msg.encoding == "rgb8":
                    image = image[:, :, ::-1].copy()
                self._latest_image = image
        except Exception as e:
            self.get_logger().error(
                f"Image conversion error: {e}",
                throttle_duration_sec=5.0,
            )

    def _lidar_callback(self, msg: PointCloud2) -> None:
        """LiDAR 生点群を更新 (正面投影用)."""
        try:
            from sensor_msgs_py import point_cloud2 as pc2

            points = pc2.read_points_numpy(
                msg, field_names=("x", "y", "z"), skip_nans=True
            )
            self._latest_points = points.astype(np.float32)
        except Exception as e:
            self.get_logger().error(
                f"PointCloud2 read error: {e}",
                throttle_duration_sec=5.0,
            )

    def _result_callback(self, msg: FallDetectionResult) -> None:
        """転倒検知結果を更新."""
        self._latest_result = msg

    def _preprocessed_callback(self, msg: PreprocessedFrame) -> None:
        """前処理済みフレーム (skeleton + 前景点群) を更新."""
        self._latest_skeleton = np.array(
            msg.skeleton_2d, dtype=np.float32
        )
        self._latest_preprocessed_pc = np.array(
            msg.pointcloud_frame, dtype=np.float32
        )
        self._frame_count += 1

    # ==================================================================
    # 描画
    # ==================================================================

    def _draw(self) -> None:
        """1 ウィンドウ左右 2 画面を描画."""
        # 左画面: カメラ画像 + skeleton
        left = self._draw_camera_panel()

        # 右画面: 点群正面投影
        right = self._draw_pointcloud_panel()

        # 左右結合
        canvas = np.hstack([left, right])

        # 転倒/待機状態のオーバーレイ
        self._draw_status_overlay(canvas)

        cv2.imshow(self._window_name, canvas)
        cv2.waitKey(1)

    def _draw_camera_panel(self) -> NDArray[np.uint8]:
        """左画面: カメラ画像 + skeleton overlay."""
        if self._latest_image is not None:
            panel = cv2.resize(
                self._latest_image, (VIS_WIDTH, VIS_HEIGHT)
            )
        else:
            panel = np.zeros(
                (VIS_HEIGHT, VIS_WIDTH, 3), dtype=np.uint8
            )
            cv2.putText(
                panel, "No Camera", (VIS_WIDTH // 4, VIS_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2,
            )

        # skeleton overlay
        if self._latest_skeleton is not None:
            self._draw_skeleton_overlay(panel, self._latest_skeleton)

        return panel

    def _draw_skeleton_overlay(
        self,
        panel: NDArray[np.uint8],
        skeleton: NDArray[np.float32],
    ) -> None:
        """skeleton を画像上に描画 (緑).

        skeleton は (51,) = (x_norm, y_norm, score) × 17 keypoints。
        正規化座標をパネルサイズにスケーリングして描画する。

        Args:
            panel: 描画先画像 (VIS_WIDTH × VIS_HEIGHT)。
            skeleton: (51,) float32 正規化 skeleton。
        """
        panel_h, panel_w = panel.shape[:2]
        keypoints = []

        for k in range(NUM_KEYPOINTS):
            x_norm = skeleton[k * KEYPOINT_DIMS + 0]
            y_norm = skeleton[k * KEYPOINT_DIMS + 1]
            score = skeleton[k * KEYPOINT_DIMS + 2]

            px = int(x_norm * panel_w)
            py = int(y_norm * panel_h)
            keypoints.append((px, py, score))

        # キーポイント描画
        for px, py, score in keypoints:
            if score > SKELETON_SCORE_THRESHOLD:
                cv2.circle(panel, (px, py), 4, COLOR_GREEN, -1)

        # ボーン (接続線) 描画
        for i, j in SKELETON_CONNECTIONS:
            if (keypoints[i][2] > SKELETON_SCORE_THRESHOLD and
                    keypoints[j][2] > SKELETON_SCORE_THRESHOLD):
                pt1 = (keypoints[i][0], keypoints[i][1])
                pt2 = (keypoints[j][0], keypoints[j][1])
                cv2.line(panel, pt1, pt2, COLOR_GREEN, 2)

    def _draw_pointcloud_panel(self) -> NDArray[np.uint8]:
        """右画面: 前景点群の正面投影 (XZ 平面).

        /preprocessed/frame の前景点群を優先的に使用する。
        未受信の場合は /livox/lidar の生点群を表示する。
        """
        panel = np.zeros(
            (VIS_HEIGHT, VIS_WIDTH, 3), dtype=np.uint8
        )

        # 前処理済み前景点群を優先使用
        points = None
        if (self._latest_preprocessed_pc is not None and
                not np.allclose(self._latest_preprocessed_pc, 0.0)):
            # (768,) → (256, 3)
            points = self._latest_preprocessed_pc.reshape(NUM_POINTS, 3)
            # 全ゼロ点を除外
            non_zero_mask = np.any(points != 0.0, axis=1)
            points = points[non_zero_mask]
        elif self._latest_points is not None:
            points = self._latest_points

        if points is None or points.shape[0] == 0:
            cv2.putText(
                panel, "No LiDAR", (VIS_WIDTH // 4, VIS_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2,
            )
            return panel

        # 正面投影: X → 画面横, Z → 画面縦 (上下反転)
        x = points[:, 0]
        z = points[:, 2]

        # 自動スケーリング
        x_min, x_max = x.min(), x.max()
        z_min, z_max = z.min(), z.max()

        x_range = max(x_max - x_min, 0.1)
        z_range = max(z_max - z_min, 0.1)

        margin = 20
        w = VIS_WIDTH - 2 * margin
        h = VIS_HEIGHT - 2 * margin

        px = ((x - x_min) / x_range * w + margin).astype(np.int32)
        pz = (h - (z - z_min) / z_range * h + margin).astype(np.int32)

        # 点を描画 (緑)
        for i in range(min(len(px), 5000)):
            cv2.circle(panel, (int(px[i]), int(pz[i])), 2, COLOR_GREEN, -1)

        return panel

    def _draw_status_overlay(self, canvas: NDArray[np.uint8]) -> None:
        """転倒/待機状態のオーバーレイ表示."""
        total_width = canvas.shape[1]

        if self._latest_result is not None:
            if self._latest_result.prediction == 1:
                # FALLING 表示 (赤)
                text = "FALLING"
                font_scale = 2.0
                thickness = 4
                text_size = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, thickness,
                )[0]
                x = (total_width - text_size[0]) // 2
                y = 60
                cv2.putText(
                    canvas, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, COLOR_RED, thickness,
                )

            # confidence 表示
            conf_text = f"Confidence: {self._latest_result.confidence:.3f}"
            cv2.putText(
                canvas, conf_text, (10, canvas.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1,
            )
        else:
            # 48 フレーム蓄積前
            wait_text = "Waiting for 48 frames"
            buf_text = f"Buffer: {self._frame_count}/48"
            cv2.putText(
                canvas, wait_text,
                (total_width // 2 - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2,
            )
            cv2.putText(
                canvas, buf_text,
                (total_width // 2 - 80, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1,
            )

    def destroy_node(self) -> None:
        """ノード破棄時にウィンドウを閉じる."""
        if HAS_CV2:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
