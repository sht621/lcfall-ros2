"""テンソル変換・正規化ユーティリティ.

inference_node で使用する:
- 48 フレーム窓の LiDAR グローバル正規化
- numpy → PyTorch テンソル変換
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# 点群固定寸法
NUM_POINTS: int = 256
POINT_DIMS: int = 3


def global_normalize_pointclouds(
    pointcloud_sequence: NDArray[np.float32],
) -> NDArray[np.float32]:
    """48 フレーム窓の LiDAR グローバル正規化.

    学習時 (`lidar/training/dataset_online_retrain.py`) と同じ規則:
    - 先頭から「viable」な基準フレームを探す
    - 基準フレームのユニーク点群の重心を引く
    - 基準フレームの距離分布 95 パーセンタイルで全フレームを割る

    Args:
        pointcloud_sequence: (T, NUM_POINTS, 3) float32。

    Returns:
        (T, NUM_POINTS, 3) 正規化済み点群。
    """
    centroid, scale = _compute_reference_frame(pointcloud_sequence)
    return ((pointcloud_sequence - centroid) / scale).astype(np.float32)


def _compute_reference_frame(
    sequence: NDArray[np.float32],
) -> Tuple[NDArray[np.float32], float]:
    """学習時と同じ基準重心・スケールを計算."""
    for frame in sequence:
        valid = frame[np.linalg.norm(frame, axis=1) > 0]
        if len(valid) == 0:
            continue

        unique_points = np.unique(valid, axis=0)
        if len(unique_points) < 4:
            continue

        centroid = unique_points.mean(axis=0)
        distances = np.linalg.norm(unique_points - centroid, axis=1)
        scale = float(np.percentile(distances, 95)) if len(distances) > 0 else 0.0
        if scale >= 1.0e-2:
            return centroid.astype(np.float32), scale

    return np.zeros(3, dtype=np.float32), 1.0


def _is_empty_frame(frame: NDArray[np.float32]) -> bool:
    """フレームが空点群 (全ゼロ) かどうか判定.

    Args:
        frame: (N, 3) 点群。

    Returns:
        全要素がゼロなら True。
    """
    return np.allclose(frame, 0.0)


def pointcloud_list_to_array(
    frames: list[NDArray[np.float32]],
    num_points: int = NUM_POINTS,
) -> NDArray[np.float32]:
    """flatten 済み点群リストを (T, num_points, 3) 配列に変換.

    Args:
        frames: T 個の (768,) float32 配列のリスト。
        num_points: 1 フレームの点数 (デフォルト 256)。

    Returns:
        (T, num_points, 3) float32 配列。
    """
    T = len(frames)
    result = np.zeros((T, num_points, POINT_DIMS), dtype=np.float32)
    for t, frame in enumerate(frames):
        result[t] = frame.reshape(num_points, POINT_DIMS)
    return result


def skeleton_list_to_array(
    frames: list[NDArray[np.float32]],
) -> NDArray[np.float32]:
    """flatten 済み skeleton リストを (T, 51) 配列に変換.

    Args:
        frames: T 個の (51,) float32 配列のリスト。

    Returns:
        (T, 51) float32 配列。
    """
    return np.array(frames, dtype=np.float32)


def to_torch_tensor(array: NDArray[np.float32]):
    """numpy 配列を PyTorch テンソルに変換.

    PyTorch が利用可能な場合のみ変換する。
    利用不可の場合は numpy 配列をそのまま返す。

    Args:
        array: numpy 配列。

    Returns:
        torch.Tensor または numpy 配列。
    """
    try:
        import torch
        return torch.from_numpy(array)
    except ImportError:
        return array
