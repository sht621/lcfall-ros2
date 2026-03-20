"""Skeleton 座標から heatmap を生成するユーティリティ.

48 フレーム分の skeleton_2d を蓄積した後、
(17, T, 56, 56) の heatmap 列を生成する。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Heatmap 解像度
HEATMAP_SIZE: int = 56

# COCO 17 keypoints
NUM_KEYPOINTS: int = 17
KEYPOINT_DIMS: int = 3  # (x_norm, y_norm, score)


def generate_heatmaps(
    skeleton_sequence: NDArray[np.float32],
    heatmap_size: int = HEATMAP_SIZE,
    sigma: float = 2.0,
) -> NDArray[np.float32]:
    """skeleton_2d 列から heatmap テンソルを生成.

    各フレーム・各キーポイントに対し、正規化座標を heatmap_size に
    スケールし、Gaussian splat で heatmap を生成する。

    Args:
        skeleton_sequence: (T, 51) の skeleton 列。
            各行は (x_norm, y_norm, score) × 17 keypoints。
        heatmap_size: heatmap の縦横ピクセル数 (デフォルト 56)。
        sigma: Gaussian の標準偏差 (ピクセル単位)。

    Returns:
        (17, T, heatmap_size, heatmap_size) float32 heatmap テンソル。
        検出失敗フレーム (全ゼロ skeleton) の heatmap もゼロになる。
    """
    T = skeleton_sequence.shape[0]
    heatmaps = np.zeros(
        (NUM_KEYPOINTS, T, heatmap_size, heatmap_size),
        dtype=np.float32,
    )

    # Gaussian カーネル用の座標グリッド (事前計算)
    y_grid, x_grid = np.mgrid[0:heatmap_size, 0:heatmap_size].astype(
        np.float32
    )

    for t in range(T):
        frame_skeleton = skeleton_sequence[t]  # (51,)

        for k in range(NUM_KEYPOINTS):
            x_norm = frame_skeleton[k * KEYPOINT_DIMS + 0]
            y_norm = frame_skeleton[k * KEYPOINT_DIMS + 1]
            score = frame_skeleton[k * KEYPOINT_DIMS + 2]

            # スコアがゼロ → 未検出 → heatmap もゼロのまま
            if score <= 0.0:
                continue

            # 正規化座標 → heatmap ピクセル座標
            cx = x_norm * (heatmap_size - 1)
            cy = y_norm * (heatmap_size - 1)

            # Gaussian splat
            gauss = np.exp(
                -((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
                / (2.0 * sigma ** 2)
            )
            heatmaps[k, t] = gauss * score

    return heatmaps
