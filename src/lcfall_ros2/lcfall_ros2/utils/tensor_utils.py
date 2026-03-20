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

    1 フレーム目を基準フレームとし、基準フレームの重心を原点に移動し、
    基準フレームの最大距離で全フレームをスケーリングする。

    - 1 フレーム目が空点群 (全ゼロ) なら、次の非空フレームを基準にする
    - 窓全体が空点群なら、ゼロ埋め系列のまま返す

    Args:
        pointcloud_sequence: (T, NUM_POINTS, 3) float32。
            各フレームは (256, 3) の点群。
            全ゼロのフレームは空点群とみなす。

    Returns:
        (T, NUM_POINTS, 3) 正規化済み点群。
    """
    T = pointcloud_sequence.shape[0]
    result = pointcloud_sequence.copy()

    # 基準フレームを探す
    ref_idx = _find_reference_frame(result)
    if ref_idx is None:
        # 全フレーム空 → そのまま返す
        return result

    ref_frame = result[ref_idx]  # (256, 3)

    # 基準フレームの重心
    centroid = ref_frame.mean(axis=0)  # (3,)

    # 全フレームから重心を引く
    for t in range(T):
        if _is_empty_frame(result[t]):
            continue
        result[t] -= centroid

    # 基準フレームの点群の最大距離
    ref_centered = result[ref_idx]
    max_dist = np.max(np.linalg.norm(ref_centered, axis=1))

    if max_dist > 1e-6:
        for t in range(T):
            if _is_empty_frame(pointcloud_sequence[t]):
                # 元が空のフレームは正規化後もゼロのまま
                result[t] = np.zeros_like(result[t])
            else:
                result[t] /= max_dist

    return result


def _find_reference_frame(
    sequence: NDArray[np.float32],
) -> Optional[int]:
    """基準フレームのインデックスを取得.

    1 フレーム目から順に非空フレームを探す。

    Args:
        sequence: (T, N, 3) 点群列。

    Returns:
        基準フレームのインデックス。全て空なら None。
    """
    for t in range(sequence.shape[0]):
        if not _is_empty_frame(sequence[t]):
            return t
    return None


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
