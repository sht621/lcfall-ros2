"""Skeleton 座標から heatmap を生成するユーティリティ.

48 フレーム分の skeleton_2d を蓄積した後、
(17, T, 56, 56) の heatmap 列を生成する。

heatmap 生成には mmpose の MSRAHeatmap codec を使用する。
mmpose が利用できない場合は同等のフォールバック実装を使用する。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Heatmap 解像度 (ViTPose-S のデフォルト出力解像度に合わせる)
HEATMAP_SIZE: int = 56

# COCO 17 keypoints
NUM_KEYPOINTS: int = 17
KEYPOINT_DIMS: int = 3  # (x_norm, y_norm, score)

# mmpose MSRAHeatmap デフォルト設定
# input_size は正規化座標 [0,1] → heatmap_size にスケーリングするため
# heatmap_size と同じ値を使用する
DEFAULT_SIGMA: float = 2.0

# mmpose codec の利用可否
try:
    from mmpose.codecs import MSRAHeatmap
    HAS_MMPOSE_CODEC = True
except ImportError:
    HAS_MMPOSE_CODEC = False


def generate_heatmaps(
    skeleton_sequence: NDArray[np.float32],
    heatmap_size: int = HEATMAP_SIZE,
    sigma: float = DEFAULT_SIGMA,
) -> NDArray[np.float32]:
    """skeleton_2d 列から heatmap テンソルを生成.

    mmpose の MSRAHeatmap codec を使用して、MSRA 方式の Gaussian heatmap
    を生成する。mmpose がインストールされていない場合は同等のフォールバック
    実装を使用する。

    Args:
        skeleton_sequence: (T, 51) の skeleton 列。
            各行は (x_norm, y_norm, score) × 17 keypoints。
        heatmap_size: heatmap の縦横ピクセル数 (デフォルト 56)。
        sigma: Gaussian の標準偏差 (ピクセル単位, デフォルト 2.0)。

    Returns:
        (17, T, heatmap_size, heatmap_size) float32 heatmap テンソル。
        検出失敗フレーム (全ゼロ skeleton) の heatmap もゼロになる。
    """
    if HAS_MMPOSE_CODEC:
        return _generate_heatmaps_mmpose(
            skeleton_sequence, heatmap_size, sigma
        )
    else:
        return _generate_heatmaps_fallback(
            skeleton_sequence, heatmap_size, sigma
        )


# ==================================================================
# mmpose codec を使用した heatmap 生成
# ==================================================================

def _generate_heatmaps_mmpose(
    skeleton_sequence: NDArray[np.float32],
    heatmap_size: int,
    sigma: float,
) -> NDArray[np.float32]:
    """mmpose MSRAHeatmap codec を使用した heatmap 生成.

    MSRAHeatmap.encode() は入力座標を input_size → heatmap_size に
    スケーリングする。正規化座標 [0,1] を使用するため、
    input_size = (1, 1) として渡し、スケーリングを自前で行う。

    Args:
        skeleton_sequence: (T, 51) skeleton 列。
        heatmap_size: heatmap 解像度。
        sigma: Gaussian sigma。

    Returns:
        (17, T, heatmap_size, heatmap_size) heatmap テンソル。
    """
    T = skeleton_sequence.shape[0]
    heatmaps = np.zeros(
        (NUM_KEYPOINTS, T, heatmap_size, heatmap_size),
        dtype=np.float32,
    )

    # MSRAHeatmap codec: input_size と heatmap_size を同じにすると
    # scale_factor = 1.0 になり、入力座標がそのまま heatmap 座標になる
    codec = MSRAHeatmap(
        input_size=(heatmap_size, heatmap_size),
        heatmap_size=(heatmap_size, heatmap_size),
        sigma=sigma,
    )

    for t in range(T):
        frame = skeleton_sequence[t]  # (51,)

        # (17, 3) に reshape: (x_norm, y_norm, score)
        kpts_raw = frame.reshape(NUM_KEYPOINTS, KEYPOINT_DIMS)

        # 正規化座標 → heatmap ピクセル座標
        keypoints = np.zeros((1, NUM_KEYPOINTS, 2), dtype=np.float32)
        keypoints[0, :, 0] = kpts_raw[:, 0] * (heatmap_size - 1)  # x
        keypoints[0, :, 1] = kpts_raw[:, 1] * (heatmap_size - 1)  # y

        # score を visibility として使用
        # mmpose は keypoints_visible < 0.5 のキーポイントをスキップ
        keypoints_visible = np.zeros(
            (1, NUM_KEYPOINTS), dtype=np.float32
        )
        keypoints_visible[0, :] = kpts_raw[:, 2]

        encoded = codec.encode(keypoints, keypoints_visible)
        # encoded['heatmaps'] shape: (17, H, W)
        heatmaps[:, t, :, :] = encoded["heatmaps"]

    return heatmaps


# ==================================================================
# フォールバック実装 (mmpose なし環境用)
# ==================================================================

def _generate_heatmaps_fallback(
    skeleton_sequence: NDArray[np.float32],
    heatmap_size: int,
    sigma: float,
) -> NDArray[np.float32]:
    """mmpose と同等の MSRA 方式 Gaussian heatmap のフォールバック実装.

    mmpose の generate_gaussian_heatmaps と同じアルゴリズム:
    - 3-sigma ルールで Gaussian の有効範囲を限定
    - 中心値 = 1.0 (正規化なし)
    - keypoint_visible < 0.5 のキーポイントはスキップ

    Args:
        skeleton_sequence: (T, 51) skeleton 列。
        heatmap_size: heatmap 解像度。
        sigma: Gaussian sigma。

    Returns:
        (17, T, heatmap_size, heatmap_size) heatmap テンソル。
    """
    T = skeleton_sequence.shape[0]
    W = H = heatmap_size
    heatmaps = np.zeros(
        (NUM_KEYPOINTS, T, H, W), dtype=np.float32
    )

    # 3-sigma rule
    radius = int(sigma * 3)
    gaussian_size = 2 * radius + 1
    x = np.arange(0, gaussian_size, 1, dtype=np.float32)
    y = x[:, None]
    x0 = y0 = gaussian_size // 2

    for t in range(T):
        frame = skeleton_sequence[t]  # (51,)

        for k in range(NUM_KEYPOINTS):
            x_norm = frame[k * KEYPOINT_DIMS + 0]
            y_norm = frame[k * KEYPOINT_DIMS + 1]
            score = frame[k * KEYPOINT_DIMS + 2]

            # mmpose 準拠: visible < 0.5 はスキップ
            if score < 0.5:
                continue

            # 正規化座標 → heatmap ピクセル座標 (整数化)
            mu_x = int(x_norm * (W - 1) + 0.5)
            mu_y = int(y_norm * (H - 1) + 0.5)

            # 境界チェック
            left = mu_x - radius
            top = mu_y - radius
            right = mu_x + radius + 1
            bottom = mu_y + radius + 1

            if left >= W or top >= H or right < 0 or bottom < 0:
                continue

            # Gaussian (中心 = 1.0, mmpose と同じ)
            gaussian = np.exp(
                -((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)
            )

            # 有効領域のクリッピング
            g_x1 = max(0, -left)
            g_x2 = min(W, right) - left
            g_y1 = max(0, -top)
            g_y2 = min(H, bottom) - top

            h_x1 = max(0, left)
            h_x2 = min(W, right)
            h_y1 = max(0, top)
            h_y2 = min(H, bottom)

            np.maximum(
                heatmaps[k, t, h_y1:h_y2, h_x1:h_x2],
                gaussian[g_y1:g_y2, g_x1:g_x2],
                out=heatmaps[k, t, h_y1:h_y2, h_x1:h_x2],
            )

    return heatmaps
