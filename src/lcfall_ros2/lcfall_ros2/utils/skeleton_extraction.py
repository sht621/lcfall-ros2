"""2D skeleton 抽出ユーティリティ.

MMPoseInferencer (RTMDet-M + ViTPose-S) を用いて画像から
1 人分の 2D keypoint を抽出する。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


# COCO 17 keypoints
NUM_KEYPOINTS: int = 17
KEYPOINT_DIMS: int = 3  # (x_norm, y_norm, score)
SKELETON_LENGTH: int = NUM_KEYPOINTS * KEYPOINT_DIMS  # 51


class SkeletonExtractor:
    """MMPoseInferencer をラップした skeleton 抽出クラス.

    初期化時に GPU 上へモデルをロードし、
    extract() で画像から 1 人分の正規化 skeleton 座標を返す。
    """

    def __init__(self, device: str = "cuda:0") -> None:
        """モデル初期化.

        Args:
            device: 推論デバイス (例: 'cuda:0', 'cpu')。
        """
        self._device = device
        self._inferencer = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """MMPoseInferencer を初期化."""
        try:
            # ViTPose-S の backbone registry を有効化する。
            import mmpretrain  # noqa: F401
            from mmpose.apis import MMPoseInferencer

            self._inferencer = MMPoseInferencer(
                pose2d="vitpose-s",
                det_model="rtmdet_m_8xb32-300e_coco",
                det_cat_ids=[0],  # person のみ
                device=self._device,
            )
        except ImportError:
            # mmpose がインストールされていない環境ではダミーモードで動作
            self._inferencer = None
        except Exception:
            # モデルのダウンロードやロードに失敗した場合
            self._inferencer = None

    def extract(
        self,
        image: NDArray[np.uint8],
        image_width: int,
        image_height: int,
    ) -> NDArray[np.float32]:
        """画像から 1 人分の 2D skeleton を抽出.

        検出された人物のうち、bbox 面積が最大のものを選択する。
        検出失敗時は全ゼロ skeleton を返す。

        Args:
            image: BGR 画像 (H, W, 3) uint8。
            image_width: 画像幅 (正規化用)。
            image_height: 画像高さ (正規化用)。

        Returns:
            (51,) float32 配列: (x_norm, y_norm, score) × 17 keypoints。
            x_norm, y_norm は [0, 1] 正規化座標。
        """
        if self._inferencer is None:
            # モデル未ロード → ゼロ skeleton
            return np.zeros(SKELETON_LENGTH, dtype=np.float32)

        try:
            results = next(
                self._inferencer(image, return_vis=False)
            )

            predictions = results.get("predictions", [[]])
            if not predictions or not predictions[0]:
                return np.zeros(SKELETON_LENGTH, dtype=np.float32)

            # bbox 面積最大の人物を選択
            persons = predictions[0]
            best_person = self._select_best_person(persons)
            if best_person is None:
                return np.zeros(SKELETON_LENGTH, dtype=np.float32)

            keypoints = np.array(
                best_person["keypoints"], dtype=np.float32
            )  # (17, 2)
            scores = np.array(
                best_person["keypoint_scores"], dtype=np.float32
            )  # (17,)

            return self._normalize_and_flatten(
                keypoints, scores, image_width, image_height
            )

        except Exception:
            return np.zeros(SKELETON_LENGTH, dtype=np.float32)

    @staticmethod
    def _select_best_person(
        persons: list[dict],
    ) -> Optional[dict]:
        """bbox 面積最大の人物を選択.

        Args:
            persons: MMPose の検出結果リスト。

        Returns:
            最適な人物の辞書、または None。
        """
        best = None
        best_area = -1.0
        for person in persons:
            bbox = person.get("bbox")
            if bbox:
                bbox_array = np.asarray(bbox, dtype=np.float32).reshape(-1)
                if bbox_array.shape[0] >= 4:
                    area = float(
                        max(bbox_array[2] - bbox_array[0], 0.0) *
                        max(bbox_array[3] - bbox_array[1], 0.0)
                    )
                else:
                    area = -1.0
            else:
                scores = np.asarray(
                    person.get("keypoint_scores", []), dtype=np.float32
                )
                area = float(scores.mean()) if scores.size > 0 else -1.0
            if area > best_area:
                best_area = area
                best = person
        return best

    @staticmethod
    def _normalize_and_flatten(
        keypoints: NDArray[np.float32],
        scores: NDArray[np.float32],
        width: int,
        height: int,
    ) -> NDArray[np.float32]:
        """keypoint 座標を [0, 1] に正規化し、flatten.

        Args:
            keypoints: (17, 2) pixel 座標。
            scores: (17,) 信頼度スコア。
            width: 画像幅。
            height: 画像高さ。

        Returns:
            (51,) = (x_norm, y_norm, score) × 17。
        """
        result = np.zeros(SKELETON_LENGTH, dtype=np.float32)
        num_points = min(NUM_KEYPOINTS, keypoints.shape[0], scores.shape[0])
        for i in range(num_points):
            result[i * KEYPOINT_DIMS + 0] = keypoints[i, 0] / width
            result[i * KEYPOINT_DIMS + 1] = keypoints[i, 1] / height
            result[i * KEYPOINT_DIMS + 2] = scores[i]
        return result
