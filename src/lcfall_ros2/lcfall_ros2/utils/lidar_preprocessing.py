"""LiDAR 点群前処理ユーティリティ.

PointCloud2 → numpy 変換、座標変換（回転）、ROI フィルタ、256 点への整形を提供する。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2


# 整形後の固定点数
NUM_POINTS: int = 256
POINT_DIMS: int = 3

# LCFall LiDAR キャリブレーション既定値 (度)
# センサ座標 → 部屋座標への変換パラメータ
DEFAULT_LIDAR_ROLL: float = 1.1    # X軸回転 [deg]
DEFAULT_LIDAR_PITCH: float = 27.8  # Y軸回転 [deg] (カメラマウント角度由来)
DEFAULT_LIDAR_YAW: float = 0.0    # Z軸回転 [deg]


def pointcloud2_to_numpy(
    cloud_msg: PointCloud2,
) -> NDArray[np.float32]:
    """PointCloud2 メッセージを (N, 3) numpy 配列に変換.

    Args:
        cloud_msg: sensor_msgs/PointCloud2 メッセージ。

    Returns:
        (N, 3) float32 配列。NaN を含む点は除外済み。
    """
    points = pc2.read_points_numpy(
        cloud_msg, field_names=("x", "y", "z"), skip_nans=True
    )
    return points.astype(np.float32)


def apply_roi(
    points: NDArray[np.float32],
    roi_min: Tuple[float, float, float],
    roi_max: Tuple[float, float, float],
) -> NDArray[np.float32]:
    """ROI (Region of Interest) フィルタを適用.

    Args:
        points: (N, 3) 点群。
        roi_min: (x_min, y_min, z_min)。
        roi_max: (x_max, y_max, z_max)。

    Returns:
        ROI 内の点のみ (M, 3)。
    """
    if points.shape[0] == 0:
        return points

    mask = (
        (points[:, 0] >= roi_min[0]) & (points[:, 0] <= roi_max[0]) &
        (points[:, 1] >= roi_min[1]) & (points[:, 1] <= roi_max[1]) &
        (points[:, 2] >= roi_min[2]) & (points[:, 2] <= roi_max[2])
    )
    return points[mask]


def apply_transform(
    points: NDArray[np.float32],
    rotation_matrix: Optional[NDArray[np.float32]] = None,
    translation: Optional[NDArray[np.float32]] = None,
) -> NDArray[np.float32]:
    """座標変換 / 回転補正を適用.

    Args:
        points: (N, 3) 点群。
        rotation_matrix: (3, 3) 回転行列。None なら回転しない。
        translation: (3,) 並進ベクトル。None なら並進しない。

    Returns:
        変換後の点群 (N, 3)。
    """
    if points.shape[0] == 0:
        return points

    result = points.copy()
    if rotation_matrix is not None:
        result = result @ rotation_matrix.T
    if translation is not None:
        result = result + translation
    return result


def create_rotation_matrix(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> NDArray[np.float32]:
    """3 軸回転行列を生成 (Rz · Ry · Rx 順).

    LCFall オフライン実装 (step2_filtering.py) と同じ回転行列を構成する。
    角度は度 (degree) で指定する。

    Args:
        roll_deg: X 軸回転角度 [deg]。
        pitch_deg: Y 軸回転角度 [deg]。
        yaw_deg: Z 軸回転角度 [deg]。

    Returns:
        (3, 3) 回転行列 (float32)。
    """
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)

    # Rx (Roll)
    cr, sr = np.cos(roll), np.sin(roll)
    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ], dtype=np.float32)

    # Ry (Pitch)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ], dtype=np.float32)

    # Rz (Yaw)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ], dtype=np.float32)

    # 合成: Rz * Ry * Rx
    return (Rz @ Ry @ Rx).astype(np.float32)


def apply_lidar_rotation(
    points: NDArray[np.float32],
    roll_deg: float = DEFAULT_LIDAR_ROLL,
    pitch_deg: float = DEFAULT_LIDAR_PITCH,
    yaw_deg: float = DEFAULT_LIDAR_YAW,
    rotation_matrix: Optional[NDArray[np.float32]] = None,
) -> NDArray[np.float32]:
    """LiDAR 点群にキャリブレーション回転を適用.

    センサ座標系から部屋座標系への変換。
    事前に計算した回転行列を渡すことで毎フレームの再計算を避けられる。

    Args:
        points: (N, 3) 点群。
        roll_deg: X 軸回転角度 [deg] (既定: 1.1)。
        pitch_deg: Y 軸回転角度 [deg] (既定: 27.8)。
        yaw_deg: Z 軸回転角度 [deg] (既定: 0.0)。
        rotation_matrix: 事前計算済み回転行列。None なら都度計算。

    Returns:
        回転後の点群 (N, 3)。
    """
    if rotation_matrix is None:
        rotation_matrix = create_rotation_matrix(
            roll_deg, pitch_deg, yaw_deg
        )
    return apply_transform(points, rotation_matrix=rotation_matrix)




def reshape_pointcloud(
    points: NDArray[np.float32],
    target_num: int = NUM_POINTS,
) -> NDArray[np.float32]:
    """前景点群を固定点数に整形.

    整形ルール:
        - M > target_num  : ランダムサンプリングで target_num 点
        - M = target_num  : そのまま
        - 0 < M < target_num : 重複サンプリングで target_num 点
        - M = 0            : 全 0 埋め

    Args:
        points: (M, 3) 前景点群。
        target_num: 整形後の点数 (デフォルト 256)。

    Returns:
        (target_num, 3) float32 配列。
    """
    num_points = points.shape[0]

    if num_points == 0:
        return np.zeros((target_num, POINT_DIMS), dtype=np.float32)

    if num_points == target_num:
        return points.copy()

    if num_points > target_num:
        # ランダムサンプリングで削減 (replace=False: 重複なし)
        indices = np.random.choice(num_points, target_num, replace=False)
        return points[indices]

    # 0 < num_points < target_num: 重複サンプリングで増量
    indices = np.random.choice(num_points, target_num, replace=True)
    return points[indices]
