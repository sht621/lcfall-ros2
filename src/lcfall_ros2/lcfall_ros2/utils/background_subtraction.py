"""ボクセル背景差分による前景点群抽出.

npz 形式の事前取得済み背景モデルを読み込み、
点群から背景ボクセルに該当する点を除去して前景点群を返す。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class BackgroundModel:
    """ボクセルベースの背景モデル.

    背景モデルは別プログラムで事前取得し npz に保存する。
    本クラスは読み込みと背景判定のみ行う。

    npz に含まれる想定キー:
        - voxel_indices : 背景ボクセルの整数 index 集合 (N, 3)
        - voxel_size    : float  ボクセル 1 辺の長さ [m]
        - roi_min       : (3,)  ROI 最小値 [x, y, z]
        - roi_max       : (3,)  ROI 最大値 [x, y, z]
    """

    def __init__(self, npz_path: str | Path) -> None:
        """背景モデルを npz ファイルから読み込む.

        Args:
            npz_path: 背景モデルファイルのパス。
                      存在しない場合は FileNotFoundError を送出する。
        """
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Background model not found: {path}"
            )

        data = np.load(str(path), allow_pickle=True)

        self.voxel_size: float = float(data["voxel_size"])
        self.roi_min: NDArray[np.float32] = data["roi_min"].astype(np.float32)
        self.roi_max: NDArray[np.float32] = data["roi_max"].astype(np.float32)

        # 背景ボクセル index を set に変換して高速 lookup
        voxel_indices = data["voxel_indices"]  # (N, 3) int
        self._bg_voxels: set[tuple[int, int, int]] = set(
            map(tuple, voxel_indices.tolist())
        )

    # ------------------------------------------------------------------
    # 公開メソッド
    # ------------------------------------------------------------------

    def remove_background(
        self, points: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """点群から背景に該当する点を除去して前景点群を返す.

        処理フロー:
            1. 各点が属するボクセル index を算出
            2. 背景ボクセルに含まれる点を除去
            3. 残った **元の生点群** を返す

        Args:
            points: (M, 3) の生点群 (ROI 適用済み想定)。

        Returns:
            前景点群 (M', 3)。M' <= M。
        """
        if points.shape[0] == 0:
            return points

        voxel_indices = self._point_to_voxel(points)

        # 背景ボクセルに **含まれない** 点のみ残す
        foreground_mask = np.array(
            [tuple(idx) not in self._bg_voxels for idx in voxel_indices],
            dtype=bool,
        )
        return points[foreground_mask]

    # ------------------------------------------------------------------
    # 内部ユーティリティ
    # ------------------------------------------------------------------

    def _point_to_voxel(
        self, points: NDArray[np.float32]
    ) -> NDArray[np.int64]:
        """点座標をボクセル index に変換.

        Args:
            points: (M, 3)

        Returns:
            (M, 3) の整数ボクセル index。
        """
        return np.floor(
            (points - self.roi_min) / self.voxel_size
        ).astype(np.int64)
