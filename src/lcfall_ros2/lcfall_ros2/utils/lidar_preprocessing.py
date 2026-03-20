"""LiDAR 点群前処理ユーティリティ.

PointCloud2 → numpy 変換、ROI フィルタ、256 点への整形を提供する。
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
