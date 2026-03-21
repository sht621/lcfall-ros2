#!/usr/bin/env python3
"""sync_preprocess_node: カメラ + LiDAR 時刻同期と 1 フレーム前処理.

/camera/image_raw と /livox/lidar を ApproximateTimeSynchronizer で同期し、
各同期ペアに対して以下の 1 フレーム前処理を行う:
  - カメラ: 2D skeleton 抽出 (MMPoseInferencer: RTMDet-M + ViTPose-S)
  - LiDAR:  座標変換 → ROI → ボクセル背景差分 → 256 点整形

結果を PreprocessedFrame として /preprocessed/frame に publish する。
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, PointCloud2
import message_filters

from lcfall_msgs.msg import PreprocessedFrame

from lcfall_ros2.utils.skeleton_extraction import SkeletonExtractor
from lcfall_ros2.utils.lidar_preprocessing import (
    pointcloud2_to_numpy,
    apply_roi,
    apply_lidar_rotation,
    create_rotation_matrix,
    reshape_pointcloud,
)
from lcfall_ros2.utils.background_subtraction import BackgroundModel


class SyncPreprocessNode(Node):
    """カメラ + LiDAR 同期前処理ノード."""

    def __init__(self) -> None:
        super().__init__("sync_preprocess_node")

        # ==============================================================
        # パラメータ宣言
        # ==============================================================
        # カメラ
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("image_width", 1280)
        self.declare_parameter("image_height", 720)
        self.declare_parameter("skeleton_device", "cuda:0")

        # LiDAR
        self.declare_parameter("lidar_topic", "/livox/lidar")

        # 同期
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop", 0.05)

        # ROI
        self.declare_parameter("roi_x_min", 0.0)
        self.declare_parameter("roi_x_max", 5.0)
        self.declare_parameter("roi_y_min", -2.0)
        self.declare_parameter("roi_y_max", 2.0)
        self.declare_parameter("roi_z_min", 0.1)
        self.declare_parameter("roi_z_max", 2.0)

        # 背景モデル
        self.declare_parameter(
            "background_model_path",
            "/data/background/background_voxel_map.npz",
        )

        # 座標変換 (LiDAR センサ座標 → 部屋座標)
        self.declare_parameter("apply_coordinate_transform", True)
        self.declare_parameter("lidar_roll", 1.1)    # X軸回転 [deg]
        self.declare_parameter("lidar_pitch", 27.8)   # Y軸回転 [deg]
        self.declare_parameter("lidar_yaw", 0.0)      # Z軸回転 [deg]

        # ==============================================================
        # パラメータ取得
        # ==============================================================
        camera_topic: str = (
            self.get_parameter("camera_topic").value
        )
        lidar_topic: str = (
            self.get_parameter("lidar_topic").value
        )
        self._image_width: int = (
            self.get_parameter("image_width").value
        )
        self._image_height: int = (
            self.get_parameter("image_height").value
        )
        skeleton_device: str = (
            self.get_parameter("skeleton_device").value
        )
        sync_queue_size: int = (
            self.get_parameter("sync_queue_size").value
        )
        sync_slop: float = (
            self.get_parameter("sync_slop").value
        )
        bg_model_path: str = (
            self.get_parameter("background_model_path").value
        )

        self._roi_min: Tuple[float, float, float] = (
            self.get_parameter("roi_x_min").value,
            self.get_parameter("roi_y_min").value,
            self.get_parameter("roi_z_min").value,
        )
        self._roi_max: Tuple[float, float, float] = (
            self.get_parameter("roi_x_max").value,
            self.get_parameter("roi_y_max").value,
            self.get_parameter("roi_z_max").value,
        )
        self._apply_transform: bool = (
            self.get_parameter("apply_coordinate_transform").value
        )

        # 回転行列を起動時に計算してキャッシュ
        self._rotation_matrix = None
        if self._apply_transform:
            lidar_roll: float = self.get_parameter("lidar_roll").value
            lidar_pitch: float = self.get_parameter("lidar_pitch").value
            lidar_yaw: float = self.get_parameter("lidar_yaw").value
            self._rotation_matrix = create_rotation_matrix(
                lidar_roll, lidar_pitch, lidar_yaw
            )
            self.get_logger().info(
                f"LiDAR rotation enabled: "
                f"roll={lidar_roll}°, pitch={lidar_pitch}°, yaw={lidar_yaw}°"
            )

        # ==============================================================
        # Skeleton 抽出器 (GPU)
        # ==============================================================
        self.get_logger().info(
            f"Initializing SkeletonExtractor on {skeleton_device} ..."
        )
        self._skeleton_extractor = SkeletonExtractor(device=skeleton_device)

        # ==============================================================
        # 背景モデル読み込み
        # ==============================================================
        self.get_logger().info(
            f"Loading background model from {bg_model_path} ..."
        )
        try:
            self._bg_model = BackgroundModel(bg_model_path)
            self.get_logger().info("Background model loaded successfully.")
        except FileNotFoundError as e:
            self.get_logger().error(str(e))
            raise SystemExit(
                f"Background model not found: {bg_model_path}"
            )

        # ==============================================================
        # CvBridge (Image → numpy 変換)
        # ==============================================================
        try:
            from cv_bridge import CvBridge
            self._bridge = CvBridge()
        except ImportError:
            self.get_logger().warn(
                "cv_bridge not available; "
                "will attempt manual Image→numpy conversion."
            )
            self._bridge = None

        # ==============================================================
        # ROS 通信
        # ==============================================================
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        image_sub = message_filters.Subscriber(
            self, Image, camera_topic, qos_profile=qos
        )
        lidar_sub = message_filters.Subscriber(
            self, PointCloud2, lidar_topic, qos_profile=qos
        )

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, lidar_sub],
            queue_size=sync_queue_size,
            slop=sync_slop,
        )
        self._sync.registerCallback(self._sync_callback)

        # Publisher
        self._pub = self.create_publisher(
            PreprocessedFrame, "/preprocessed/frame", 10
        )

        self.get_logger().info("SyncPreprocessNode initialized.")

    # ==================================================================
    # コールバック
    # ==================================================================

    def _sync_callback(
        self, img_msg: Image, cloud_msg: PointCloud2
    ) -> None:
        """同期コールバック: 画像と点群を並列処理して publish."""
        skeleton_result: Optional[NDArray[np.float32]] = None
        pointcloud_result: Optional[NDArray[np.float32]] = None

        def _process_image() -> None:
            nonlocal skeleton_result
            skeleton_result = self._extract_skeleton(img_msg)

        def _process_lidar() -> None:
            nonlocal pointcloud_result
            pointcloud_result = self._process_pointcloud(cloud_msg)

        # 画像 (GPU) と点群 (CPU) を並列実行
        t_img = threading.Thread(target=_process_image)
        t_lid = threading.Thread(target=_process_lidar)
        t_img.start()
        t_lid.start()
        t_img.join()
        t_lid.join()

        if skeleton_result is None or pointcloud_result is None:
            self.get_logger().warn("Skipping frame due to processing error.")
            return

        # publish
        msg = PreprocessedFrame()
        msg.header = cloud_msg.header
        msg.skeleton_2d = skeleton_result.tolist()
        msg.pointcloud_frame = pointcloud_result.tolist()
        self._pub.publish(msg)

    # ==================================================================
    # カメラ前処理
    # ==================================================================

    def _extract_skeleton(
        self, img_msg: Image
    ) -> Optional[NDArray[np.float32]]:
        """画像から 1 人分の 2D skeleton を抽出.

        Returns:
            (51,) float32 配列。エラー時は None。
        """
        try:
            image = self._image_msg_to_numpy(img_msg)
            if image is None:
                return np.zeros(51, dtype=np.float32)

            return self._skeleton_extractor.extract(
                image, self._image_width, self._image_height
            )

        except Exception as e:
            self.get_logger().error(f"Skeleton extraction error: {e}")
            return None

    def _image_msg_to_numpy(
        self, img_msg: Image
    ) -> Optional[NDArray[np.uint8]]:
        """Image メッセージを BGR numpy 配列に変換."""
        if self._bridge is not None:
            return self._bridge.imgmsg_to_cv2(
                img_msg, desired_encoding="bgr8"
            )

        # cv_bridge なしの場合のフォールバック
        try:
            import numpy as np
            data = np.frombuffer(img_msg.data, dtype=np.uint8)
            if img_msg.encoding in ("rgb8", "bgr8"):
                image = data.reshape(
                    (img_msg.height, img_msg.width, 3)
                )
                if img_msg.encoding == "rgb8":
                    image = image[:, :, ::-1].copy()  # RGB → BGR
                return image
            else:
                self.get_logger().warn(
                    f"Unsupported encoding: {img_msg.encoding}"
                )
                return None
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return None

    # ==================================================================
    # LiDAR 前処理
    # ==================================================================

    def _process_pointcloud(
        self, cloud_msg: PointCloud2
    ) -> Optional[NDArray[np.float32]]:
        """LiDAR 点群の 1 フレーム前処理.

        PointCloud2 → numpy → (変換) → ROI → 背景差分 → 256 点整形 → flatten

        Returns:
            (768,) float32 配列。エラー時は None。
        """
        try:
            # PointCloud2 → numpy (N, 3)
            points = pointcloud2_to_numpy(cloud_msg)

            if points.shape[0] == 0:
                return np.zeros(256 * 3, dtype=np.float32)

            # 座標変換: センサ座標 → 部屋座標
            if self._apply_transform:
                points = apply_lidar_rotation(
                    points, rotation_matrix=self._rotation_matrix
                )

            # ROI フィルタ
            points = apply_roi(points, self._roi_min, self._roi_max)

            # ボクセル背景差分
            points = self._bg_model.remove_background(points)

            # 256 点へ整形
            shaped = reshape_pointcloud(points)

            return shaped.flatten()

        except Exception as e:
            self.get_logger().error(f"LiDAR processing error: {e}")
            return None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SyncPreprocessNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
