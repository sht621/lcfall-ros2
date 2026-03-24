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

from collections import deque
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, PointCloud2

from lcfall_msgs.msg import FallDetectionResult, PreprocessedFrame
from lcfall_ros2.utils.lidar_preprocessing import pointcloud2_to_numpy

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
COLOR_GRID = (70, 70, 70)
COLOR_AXIS = (180, 180, 180)
COLOR_BG = (18, 18, 18)
COLOR_BONE = (80, 220, 255)
COLOR_JOINT = (20, 255, 120)
COLOR_TEXT_MUTED = (170, 170, 170)
COLOR_GRAPH_LINE = (80, 220, 255)
COLOR_GRAPH_FILL = (40, 110, 140)
COLOR_GRAPH_ALERT = (0, 90, 220)

# skeleton 描画しきい値
SKELETON_SCORE_THRESHOLD = 0.3
GRID_STEP_METERS = 0.5
PROB_HISTORY_SIZE = 120
BOTTOM_BAR_HEIGHT = 150


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
        self.declare_parameter("window_width", 1600)
        self.declare_parameter("window_height", 900)
        self.declare_parameter("roi_x_min", 0.0)
        self.declare_parameter("roi_x_max", 4.0)
        self.declare_parameter("roi_y_min", -2.0)
        self.declare_parameter("roi_y_max", 2.0)
        self.declare_parameter("roi_z_min", -1.2)
        self.declare_parameter("roi_z_max", 1.0)

        camera_topic: str = self.get_parameter("camera_topic").value
        lidar_topic: str = self.get_parameter("lidar_topic").value
        self._image_width: int = self.get_parameter("image_width").value
        self._image_height: int = self.get_parameter("image_height").value
        self._window_name: str = self.get_parameter("window_name").value
        self._window_width: int = self.get_parameter("window_width").value
        self._window_height: int = self.get_parameter("window_height").value
        self._roi_min = np.array([
            self.get_parameter("roi_x_min").value,
            self.get_parameter("roi_y_min").value,
            self.get_parameter("roi_z_min").value,
        ], dtype=np.float32)
        self._roi_max = np.array([
            self.get_parameter("roi_x_max").value,
            self.get_parameter("roi_y_max").value,
            self.get_parameter("roi_z_max").value,
        ], dtype=np.float32)

        # ==============================================================
        # 状態変数
        # ==============================================================
        self._latest_image: Optional[NDArray[np.uint8]] = None
        self._latest_points: Optional[NDArray[np.float32]] = None
        self._latest_result: Optional[FallDetectionResult] = None
        self._latest_skeleton: Optional[NDArray[np.float32]] = None
        self._latest_preprocessed_pc: Optional[NDArray[np.float32]] = None
        self._frame_count: int = 0
        self._prob_history: deque[float] = deque(maxlen=PROB_HISTORY_SIZE)
        self._bridge = CvBridge() if HAS_CVBRIDGE else None
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self._window_name,
            self._window_width,
            self._window_height,
        )

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
            self._latest_points = pointcloud2_to_numpy(msg)
        except Exception as e:
            self.get_logger().error(
                f"PointCloud2 read error: {e}",
                throttle_duration_sec=5.0,
            )

    def _result_callback(self, msg: FallDetectionResult) -> None:
        """転倒検知結果を更新."""
        self._latest_result = msg
        self._prob_history.append(float(np.clip(msg.confidence, 0.0, 1.0)))

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
        top_canvas = np.hstack([left, right])

        # 転倒/待機状態のオーバーレイ
        self._draw_status_overlay(top_canvas)

        bottom_bar = np.full(
            (BOTTOM_BAR_HEIGHT, top_canvas.shape[1], 3),
            COLOR_BG,
            dtype=np.uint8,
        )
        self._draw_probability_history(bottom_bar)

        canvas = np.vstack([top_canvas, bottom_bar])

        display = self._fit_canvas_to_window(canvas)
        cv2.imshow(self._window_name, display)
        cv2.waitKey(1)

    def _fit_canvas_to_window(
        self,
        canvas: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """描画キャンバス全体をウィンドウサイズへ拡大縮小."""
        canvas_h, canvas_w = canvas.shape[:2]
        scale = min(
            self._window_width / max(canvas_w, 1),
            self._window_height / max(canvas_h, 1),
        )

        if scale <= 1.0:
            # 縮小も許可して、常にウィンドウ内に収める。
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_LINEAR

        target_w = max(1, int(round(canvas_w * scale)))
        target_h = max(1, int(round(canvas_h * scale)))
        return cv2.resize(canvas, (target_w, target_h), interpolation=interp)

    def _draw_camera_panel(self) -> NDArray[np.uint8]:
        """左画面: カメラ画像 + skeleton overlay."""
        if self._latest_image is not None:
            panel = cv2.resize(
                self._latest_image, (VIS_WIDTH, VIS_HEIGHT)
            )
        else:
            panel = np.full(
                (VIS_HEIGHT, VIS_WIDTH, 3), COLOR_BG, dtype=np.uint8
            )
            cv2.putText(
                panel, "No Camera", (VIS_WIDTH // 4, VIS_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2,
            )

        # skeleton overlay
        if self._latest_skeleton is not None:
            self._draw_skeleton_overlay(panel, self._latest_skeleton)
            if not np.any(self._latest_skeleton[2::3] > SKELETON_SCORE_THRESHOLD):
                cv2.putText(
                    panel,
                    "No person detected",
                    (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    COLOR_TEXT_MUTED,
                    2,
                    cv2.LINE_AA,
                )

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

            px = int(np.clip(x_norm, 0.0, 1.0) * (panel_w - 1))
            py = int(np.clip(y_norm, 0.0, 1.0) * (panel_h - 1))
            keypoints.append((px, py, score))

        # キーポイント描画
        for px, py, score in keypoints:
            if score > SKELETON_SCORE_THRESHOLD:
                cv2.circle(panel, (px, py), 6, COLOR_JOINT, -1, cv2.LINE_AA)
                cv2.circle(panel, (px, py), 10, COLOR_JOINT, 1, cv2.LINE_AA)

        # ボーン (接続線) 描画
        for i, j in SKELETON_CONNECTIONS:
            if (keypoints[i][2] > SKELETON_SCORE_THRESHOLD and
                    keypoints[j][2] > SKELETON_SCORE_THRESHOLD):
                pt1 = (keypoints[i][0], keypoints[i][1])
                pt2 = (keypoints[j][0], keypoints[j][1])
                cv2.line(panel, pt1, pt2, COLOR_BONE, 3, cv2.LINE_AA)

    def _draw_pointcloud_panel(self) -> NDArray[np.uint8]:
        """右画面: 斜視の単一ビュー点群表示."""
        panel = np.full((VIS_HEIGHT, VIS_WIDTH, 3), COLOR_BG, dtype=np.uint8)
        points = self._get_display_points()
        rect = (18, 18, VIS_WIDTH - 36, VIS_HEIGHT - 36)
        self._draw_oblique_view(panel, rect, points)
        return panel

    def _get_display_points(self) -> NDArray[np.float32]:
        """表示用の前景点群を取得."""
        if self._latest_preprocessed_pc is None:
            return np.empty((0, 3), dtype=np.float32)

        points = self._latest_preprocessed_pc.reshape(NUM_POINTS, 3)
        non_zero_mask = np.any(points != 0.0, axis=1)
        return points[non_zero_mask]

    def _draw_oblique_view(
        self,
        panel: NDArray[np.uint8],
        rect: tuple[int, int, int, int],
        points: NDArray[np.float32],
    ) -> None:
        """床グリッド付きの斜視点群ビューを描画."""
        x0, y0, width, height = rect
        view = panel[y0:y0 + height, x0:x0 + width]
        view[:] = (28, 28, 28)
        cv2.rectangle(panel, (x0, y0), (x0 + width, y0 + height), COLOR_AXIS, 1)
        self._draw_floor_grid(view)
        self._draw_oblique_axes(view)

        if points.shape[0] == 0:
            cv2.putText(
                view,
                "No foreground points",
                (18, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR_TEXT_MUTED,
                2,
                cv2.LINE_AA,
            )
            return

        order = np.argsort(points[:, 1] - points[:, 2] * 0.2)
        for idx in order:
            px, py = self._project_point(points[idx], width, height)
            if 0 <= px < width and 0 <= py < height:
                color = self._height_to_color(points[idx, 2])
                cv2.circle(view, (px, py), 3, color, -1, cv2.LINE_AA)

    def _draw_floor_grid(self, view: NDArray[np.uint8]) -> None:
        """斜視図用の床グリッドを描画."""
        height, width = view.shape[:2]
        x_values = np.arange(
            float(self._roi_min[0]), float(self._roi_max[0]) + 1e-6, GRID_STEP_METERS
        )
        y_values = np.arange(
            float(self._roi_min[1]), float(self._roi_max[1]) + 1e-6, GRID_STEP_METERS
        )

        for x in x_values:
            p0 = self._project_xyz(x, float(self._roi_min[1]), float(self._roi_min[2]), width, height)
            p1 = self._project_xyz(x, float(self._roi_max[1]), float(self._roi_min[2]), width, height)
            cv2.line(view, p0, p1, COLOR_GRID, 1, cv2.LINE_AA)
            cv2.putText(
                view, f"{x:.1f}", (p1[0] + 2, p1[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT_MUTED, 1, cv2.LINE_AA,
            )

        for y in y_values:
            p0 = self._project_xyz(float(self._roi_min[0]), y, float(self._roi_min[2]), width, height)
            p1 = self._project_xyz(float(self._roi_max[0]), y, float(self._roi_min[2]), width, height)
            cv2.line(view, p0, p1, COLOR_GRID, 1, cv2.LINE_AA)
            cv2.putText(
                view, f"{y:.1f}", (p0[0] - 22, p0[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT_MUTED, 1, cv2.LINE_AA,
            )

    def _draw_oblique_axes(self, view: NDArray[np.uint8]) -> None:
        """斜視図の基準軸を描画."""
        height, width = view.shape[:2]
        origin = self._project_xyz(
            float(self._roi_min[0]),
            0.0,
            float(self._roi_min[2]),
            width,
            height,
        )
        x_axis = self._project_xyz(
            float(self._roi_max[0]),
            0.0,
            float(self._roi_min[2]),
            width,
            height,
        )
        y_axis = self._project_xyz(
            float(self._roi_min[0]),
            float(self._roi_max[1]),
            float(self._roi_min[2]),
            width,
            height,
        )

        cv2.line(view, origin, x_axis, COLOR_AXIS, 1, cv2.LINE_AA)
        cv2.line(view, origin, y_axis, COLOR_AXIS, 1, cv2.LINE_AA)

    @staticmethod
    def _height_to_color(value: float) -> tuple[int, int, int]:
        """高さに応じた点群色を返す."""
        t = float(np.clip((value + 1.2) / 2.4, 0.0, 1.0))
        blue = int(255 * (1.0 - t))
        green = int(180 + 60 * t)
        red = int(255 * t)
        return (blue, green, red)

    def _project_point(
        self,
        point: NDArray[np.float32],
        width: int,
        height: int,
    ) -> tuple[int, int]:
        """点群1点を斜視図へ投影."""
        return self._project_xyz(
            float(point[0]), float(point[1]), float(point[2]), width, height
        )

    def _project_xyz(
        self,
        x: float,
        y: float,
        z: float,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        """XYZ を斜視図ピクセル座標へ変換."""
        x_min, x_max = float(self._roi_min[0]), float(self._roi_max[0])
        y_min, y_max = float(self._roi_min[1]), float(self._roi_max[1])
        z_min, z_max = float(self._roi_min[2]), float(self._roi_max[2])

        x_centered = x - (x_min + x_max) * 0.5
        y_centered = y - (y_min + y_max) * 0.5
        z_offset = z - z_min

        x_span = max(x_max - x_min, 1e-6)
        y_span = max(y_max - y_min, 1e-6)
        z_span = max(z_max - z_min, 1e-6)

        theta = np.deg2rad(45.0)
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))

        # 45度の斜視投影:
        # X は右奥、Y は左奥へ同じ角度で効かせる。
        sx = (x_centered / x_span) * cos_t
        sy = (y_centered / y_span) * sin_t
        sz = z_offset / z_span

        px = int(
            np.clip(
                width * (0.50 + 0.78 * (sx - sy)),
                0.0,
                width - 1.0,
            )
        )
        py = int(
            np.clip(
                height * (0.82 - 0.32 * (sx + sy) - 0.58 * sz),
                0.0,
                height - 1.0,
            )
        )
        return px, py

    def _draw_status_overlay(self, canvas: NDArray[np.uint8]) -> None:
        """転倒/待機状態のオーバーレイ表示."""
        total_width = canvas.shape[1]

        if self._latest_result is not None:
            if self._latest_result.prediction == 1:
                # 転倒時は画面全体を赤枠で囲って視認性を上げる。
                cv2.rectangle(
                    canvas,
                    (4, 4),
                    (canvas.shape[1] - 5, canvas.shape[0] - 5),
                    COLOR_RED,
                    6,
                    cv2.LINE_AA,
                )
                cv2.rectangle(
                    canvas,
                    (14, 14),
                    (canvas.shape[1] - 15, canvas.shape[0] - 15),
                    COLOR_RED,
                    2,
                    cv2.LINE_AA,
                )
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

            # confidence には fall probability を表示する
            conf_text = f"Fall Prob: {self._latest_result.confidence:.3f}"
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

    def _draw_probability_history(self, canvas: NDArray[np.uint8]) -> None:
        """画面下部に転倒確率の折れ線グラフを描画."""
        graph_margin = 16
        graph_height = canvas.shape[0] - 34
        graph_width = canvas.shape[1] - graph_margin * 2
        x0 = graph_margin
        y0 = canvas.shape[0] - graph_height - 12

        # 背景パネル
        cv2.rectangle(
            canvas,
            (x0, y0),
            (x0 + graph_width, y0 + graph_height),
            (24, 24, 24),
            -1,
        )
        cv2.rectangle(
            canvas,
            (x0, y0),
            (x0 + graph_width, y0 + graph_height),
            COLOR_AXIS,
            1,
        )

        # ガイドライン
        for frac in (0.25, 0.5, 0.75):
            gy = y0 + int((1.0 - frac) * graph_height)
            cv2.line(
                canvas,
                (x0, gy),
                (x0 + graph_width, gy),
                COLOR_GRID,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            "Fall Probability History",
            (x0 + 8, y0 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

        if len(self._prob_history) < 2:
            cv2.putText(
                canvas,
                "Waiting for probability history",
                (x0 + 12, y0 + graph_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                COLOR_TEXT_MUTED,
                1,
                cv2.LINE_AA,
            )
            return

        values = np.asarray(self._prob_history, dtype=np.float32)
        xs = np.linspace(x0 + 8, x0 + graph_width - 8, len(values))
        ys = y0 + (1.0 - values) * (graph_height - 12) + 6
        points = np.stack([xs, ys], axis=1).astype(np.int32)

        # 塗りつぶし
        fill_points = np.vstack([
            points,
            np.array([[x0 + graph_width - 8, y0 + graph_height - 6]], dtype=np.int32),
            np.array([[x0 + 8, y0 + graph_height - 6]], dtype=np.int32),
        ])
        cv2.fillPoly(canvas, [fill_points], COLOR_GRAPH_FILL)

        # しきい値線
        threshold_y = y0 + int((1.0 - 0.5) * graph_height)
        cv2.line(
            canvas,
            (x0, threshold_y),
            (x0 + graph_width, threshold_y),
            COLOR_GRAPH_ALERT,
            1,
            cv2.LINE_AA,
        )

        cv2.polylines(
            canvas,
            [points],
            isClosed=False,
            color=COLOR_GRAPH_LINE,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        latest = float(values[-1])
        cv2.putText(
            canvas,
            f"{latest:.2f}",
            (x0 + graph_width - 52, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
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
