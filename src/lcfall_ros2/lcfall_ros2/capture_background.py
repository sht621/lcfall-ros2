#!/usr/bin/env python3
"""背景モデル作成プログラム (capture_background.py).

Livox MID-360 の LiDAR 点群を一定フレーム数蓄積し、
ボクセルベースの背景モデルを npz 形式で保存する。

転倒検知システム本体とは独立して実行する。
部屋に人がいない状態で実行すること。

使い方:
    # ROS2 ノードとして実行
    ros2 run lcfall_ros2 capture_background

    # パラメータ指定
    ros2 run lcfall_ros2 capture_background --ros-args \
        -p output_path:=/data/background/background_voxel_map.npz \
        -p capture_frames:=30 \
        -p voxel_size:=0.10 \
        -p min_hits:=10
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2


class CaptureBackgroundNode(Node):
    """背景モデル取得ノード.

    指定フレーム数の点群を蓄積し、各ボクセルの出現回数をカウント。
    閾値以上出現したボクセルを背景として npz に保存する。
    """

    def __init__(self) -> None:
        super().__init__("capture_background_node")

        # ==============================================================
        # パラメータ
        # ==============================================================
        self.declare_parameter("lidar_topic", "/livox/lidar")
        self.declare_parameter(
            "output_path",
            "/data/background/background_voxel_map.npz",
        )
        self.declare_parameter("capture_frames", 200)
        self.declare_parameter("voxel_size", 0.05)
        self.declare_parameter("min_hits", 5)

        # ROI
        self.declare_parameter("roi_x_min", 0.0)
        self.declare_parameter("roi_x_max", 5.0)
        self.declare_parameter("roi_y_min", -2.0)
        self.declare_parameter("roi_y_max", 2.0)
        self.declare_parameter("roi_z_min", 0.1)
        self.declare_parameter("roi_z_max", 2.0)

        # 座標変換 (LiDAR センサ座標 → 部屋座標)
        self.declare_parameter("apply_coordinate_transform", True)
        self.declare_parameter("lidar_roll", 1.1)    # X軸回転 [deg]
        self.declare_parameter("lidar_pitch", 27.8)   # Y軸回転 [deg]
        self.declare_parameter("lidar_yaw", 0.0)      # Z軸回転 [deg]

        # パラメータ取得
        lidar_topic: str = self.get_parameter("lidar_topic").value
        self._output_path: str = self.get_parameter("output_path").value
        self._capture_frames: int = self.get_parameter("capture_frames").value
        self._voxel_size: float = self.get_parameter("voxel_size").value
        self._min_hits: int = self.get_parameter("min_hits").value

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

        self._apply_transform: bool = self.get_parameter("apply_coordinate_transform").value
        self._rotation_matrix = None
        if self._apply_transform:
            from lcfall_ros2.utils.lidar_preprocessing import create_rotation_matrix
            lidar_roll = self.get_parameter("lidar_roll").value
            lidar_pitch = self.get_parameter("lidar_pitch").value
            lidar_yaw = self.get_parameter("lidar_yaw").value
            self._rotation_matrix = create_rotation_matrix(lidar_roll, lidar_pitch, lidar_yaw)

        # ==============================================================
        # 状態変数
        # ==============================================================
        # ボクセル出現カウント: {(ix, iy, iz): count}
        self._voxel_counts: Dict[Tuple[int, int, int], int] = {}
        self._frame_count: int = 0
        self._finished: bool = False

        # ==============================================================
        # ROS 通信
        # ==============================================================
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._sub = self.create_subscription(
            PointCloud2, lidar_topic, self._lidar_callback, qos
        )

        self.get_logger().info(
            f"Background capture started.\n"
            f"  Topic     : {lidar_topic}\n"
            f"  Frames    : {self._capture_frames}\n"
            f"  Voxel size: {self._voxel_size} m\n"
            f"  Min hits  : {self._min_hits}\n"
            f"  ROI min   : {self._roi_min}\n"
            f"  ROI max   : {self._roi_max}\n"
            f"  Output    : {self._output_path}\n"
            f"\n"
            f"部屋に人がいない状態で実行してください"
        )

    # ==================================================================
    # コールバック
    # ==================================================================

    def _lidar_callback(self, msg: PointCloud2) -> None:
        """LiDAR 点群を受信してボクセルカウントを蓄積."""
        if self._finished:
            return

        try:
            raw_points = pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True
            )
            if isinstance(raw_points, np.ndarray) and raw_points.dtype.names:
                points = np.column_stack(
                    [raw_points["x"], raw_points["y"], raw_points["z"]]
                ).astype(np.float32, copy=False)
            else:
                points = np.asarray(list(raw_points), dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"PointCloud2 read error: {e}")
            return

        if points.shape[0] == 0:
            self.get_logger().warn("Empty pointcloud received, skipping.")
            return

        # 座標変換: センサ座標 → 部屋座標 (前処理ノードと完全に一致させる)
        if self._apply_transform:
            from lcfall_ros2.utils.lidar_preprocessing import apply_lidar_rotation
            points = apply_lidar_rotation(
                points, rotation_matrix=self._rotation_matrix
            )

        # ROI フィルタ
        mask = (
            (points[:, 0] >= self._roi_min[0]) &
            (points[:, 0] <= self._roi_max[0]) &
            (points[:, 1] >= self._roi_min[1]) &
            (points[:, 1] <= self._roi_max[1]) &
            (points[:, 2] >= self._roi_min[2]) &
            (points[:, 2] <= self._roi_max[2])
        )
        points = points[mask]

        if points.shape[0] == 0:
            self.get_logger().warn("No points within ROI, skipping.")
            return

        # ボクセル化してカウント
        voxel_indices = np.floor(
            (points - self._roi_min) / self._voxel_size
        ).astype(np.int64)

        for idx in voxel_indices:
            key = (int(idx[0]), int(idx[1]), int(idx[2]))
            self._voxel_counts[key] = self._voxel_counts.get(key, 0) + 1

        self._frame_count += 1
        self.get_logger().info(
            f"Captured frame {self._frame_count}/{self._capture_frames} "
            f"({points.shape[0]} points in ROI, "
            f"{len(self._voxel_counts)} unique voxels)"
        )

        # 蓄積完了
        if self._frame_count >= self._capture_frames:
            self._save_background_model()
            self._finished = True

    # ==================================================================
    # 背景モデル保存
    # ==================================================================

    def _save_background_model(self) -> None:
        """蓄積したボクセルカウントから背景モデルを npz に保存."""
        # min_hits 以上出現したボクセルを背景とする
        bg_voxels = [
            voxel for voxel, count in self._voxel_counts.items()
            if count >= self._min_hits
        ]

        if not bg_voxels:
            self.get_logger().error(
                f"No voxels with >= {self._min_hits} hits. "
                f"Increase capture_frames or decrease min_hits."
            )
            return

        voxel_indices = np.array(bg_voxels, dtype=np.int64)

        # 出力ディレクトリ作成
        output_path = Path(self._output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # npz 保存
        np.savez(
            str(output_path),
            voxel_indices=voxel_indices,
            voxel_size=np.float64(self._voxel_size),
            roi_min=self._roi_min,
            roi_max=self._roi_max,
            metadata=np.array({
                "sensor": "Livox MID-360",
                "created_at": datetime.now().isoformat(),
                "capture_frames": self._capture_frames,
                "min_hits": self._min_hits,
                "total_unique_voxels": len(self._voxel_counts),
                "background_voxels": len(bg_voxels),
            }),
        )

        self.get_logger().info(
            f"\n"
            f"Background model saved!\n"
            f"  Output        : {output_path}\n"
            f"  Total voxels  : {len(self._voxel_counts)}\n"
            f"  Background    : {len(bg_voxels)} voxels "
            f"(min_hits >= {self._min_hits})\n"
            f"  Voxel size    : {self._voxel_size} m\n"
            f"  Frames used   : {self._frame_count}\n"
            f"\n"
            f"  You can now start the fall detection system."
        )

        # ノード終了をリクエスト
        self.get_logger().info("Shutting down capture node...")
        raise SystemExit(0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CaptureBackgroundNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        node.get_logger().warn("Capture interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
