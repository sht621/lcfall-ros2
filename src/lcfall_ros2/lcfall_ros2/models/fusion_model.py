#!/usr/bin/env python3
"""カメラ・LiDAR 融合転倒検知モデル (Fusion Model).

Late Fusion 方式:
  - Stream A: PoseC3D (ResNet3dSlowOnly-50) → 512次元特徴
  - Stream B: PointNet++ + GRU → 512次元特徴
  - Fusion Head: 連結(1024次元) → MLP → 2クラス分類

オフライン実装 (offline/fusionmodel.py) からの移植。
インポートパスをオンライン ROS2 パッケージに適合させている。
"""

import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path

# LiDAR モデル（同一パッケージ内）
from lcfall_ros2.models.lidar_model import PointNet2GRUModel as LiDARModel

# MMAction2 のインポート
try:
    from mmaction.registry import MODELS
    from mmengine.config import Config
    import mmaction.models  # レジストリにモデルを登録
    from mmaction.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    HAS_MMACTION = True
except ImportError:
    MODELS = None
    Config = None
    HAS_MMACTION = False


class FusionHead(nn.Module):
    """融合ヘッド（MLP）.

    カメラとLiDARの特徴を連結し、最終的な分類を行う。
    構造: 連結 (512+512=1024) → Linear(1024,512) → BN → ReLU → Dropout → Linear(512,2)
    """

    def __init__(
        self,
        camera_feature_dim: int = 512,
        lidar_feature_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super(FusionHead, self).__init__()

        self.camera_feature_dim = camera_feature_dim
        self.lidar_feature_dim = lidar_feature_dim
        self.fusion_dim = camera_feature_dim + lidar_feature_dim

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        camera_features: torch.Tensor,
        lidar_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            camera_features: (B, camera_feature_dim)
            lidar_features: (B, lidar_feature_dim)

        Returns:
            output: (B, num_classes)
        """
        fused_features = torch.cat(
            [camera_features, lidar_features], dim=1
        )  # (B, 1024)
        output = self.fusion_mlp(fused_features)  # (B, num_classes)
        return output


class CameraLiDARFusionModel(nn.Module):
    """カメラ・LiDAR 融合転倒検知モデル.

    Late Fusion 方式で2つのストリームを統合:
      - Stream A: PoseC3D (カメラ) → backbone → GAP → 512次元
      - Stream B: PointNet++ + GRU (LiDAR) → extract_features → 512次元
      - FusionHead で連結 → 分類
    """

    def __init__(
        self,
        camera_config_path: str,
        camera_checkpoint_path: str,
        lidar_checkpoint_path: str,
        fusion_checkpoint_path: str,
        num_classes: int = 2,
        dropout: float = 0.5,
        device: str = "cuda:0",
    ):
        """
        Args:
            camera_config_path: PoseC3D の設定ファイルパス
            camera_checkpoint_path: PoseC3D のチェックポイントパス
            lidar_checkpoint_path: LiDAR モデルのチェックポイントパス
            fusion_checkpoint_path: Fusion Head のチェックポイントパス
            num_classes: クラス数
            dropout: Fusion Head のドロップアウト率
            device: ロード先デバイス
        """
        super(CameraLiDARFusionModel, self).__init__()

        self.num_classes = num_classes
        self._device_str = device

        # カメラモデル（PoseC3D）をロード
        self.camera_model = self._load_camera_model(
            camera_config_path, camera_checkpoint_path
        )

        # LiDAR モデル（PointNet++ + GRU）をロード
        self.lidar_model = self._load_lidar_model(lidar_checkpoint_path)

        # Fusion Head
        self.fusion_head = FusionHead(
            camera_feature_dim=512,
            lidar_feature_dim=512,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Fusion Head の重みをロード
        self._load_fusion_head(fusion_checkpoint_path)

        # バックボーンを固定（推論時はすべて固定）
        self._freeze_all()

    def _load_camera_model(
        self, config_path: str, checkpoint_path: str
    ) -> nn.Module:
        """PoseC3D モデルをロード."""
        if not HAS_MMACTION:
            raise ImportError(
                "MMAction2 is required for camera model. "
                "Install with: pip install mmaction>=1.0.0"
            )

        # 設定ファイルをロード
        cfg = Config.fromfile(config_path)

        # モデルを構築
        model = MODELS.build(cfg.model)

        # チェックポイントをロード
        checkpoint = torch.load(
            checkpoint_path, map_location=self._device_str
        )
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        return model

    def _load_lidar_model(self, checkpoint_path: str) -> nn.Module:
        """LiDAR モデル（PointNet++ + GRU）をロード."""
        model = LiDARModel(
            num_classes=2,
            num_points=256,
            hidden_size=512,
            num_gru_layers=2,
            dropout=0.5,
        )

        checkpoint = torch.load(
            checkpoint_path, map_location=self._device_str
        )
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # module. プレフィックスを除去（DataParallel 対応）
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {
                k.replace("module.", ""): v
                for k, v in state_dict.items()
            }

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        return model

    def _load_fusion_head(self, checkpoint_path: str) -> None:
        """Fusion Head の重みをロード."""
        checkpoint = torch.load(
            checkpoint_path, map_location=self._device_str
        )
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # module. プレフィックスを除去
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {
                k.replace("module.", ""): v
                for k, v in state_dict.items()
            }

        # fusion_head. プレフィックスのキーのみ抽出
        fusion_keys = {
            k.replace("fusion_head.", ""): v
            for k, v in state_dict.items()
            if k.startswith("fusion_head.")
        }

        if fusion_keys:
            self.fusion_head.load_state_dict(fusion_keys, strict=False)
        else:
            # プレフィックスなしの場合はそのままロード
            self.fusion_head.load_state_dict(state_dict, strict=False)

    def _freeze_all(self) -> None:
        """推論時: すべてのパラメータを固定."""
        for param in self.parameters():
            param.requires_grad = False

    def extract_camera_features(
        self, skeleton_data: torch.Tensor
    ) -> torch.Tensor:
        """カメラ特徴を抽出（PoseC3D backbone）.

        Args:
            skeleton_data: (B, C, T, H, W) - 3D ヒートマップ

        Returns:
            features: (B, 512)
        """
        features = self.camera_model.backbone(skeleton_data)

        # Global Average Pooling
        if isinstance(features, tuple):
            features = features[-1]

        # (B, C, T, H, W) → (B, C)
        features = features.mean(dim=[2, 3, 4])

        return features

    def extract_lidar_features(
        self, pointcloud_data: torch.Tensor
    ) -> torch.Tensor:
        """LiDAR 特徴を抽出（PointNet++ + GRU）.

        Args:
            pointcloud_data: (B, T, N, 3) - 時系列点群

        Returns:
            features: (B, 512)
        """
        features = self.lidar_model.extract_features(pointcloud_data)
        return features

    def forward(
        self,
        skeleton_data: torch.Tensor,
        pointcloud_data: torch.Tensor,
    ) -> torch.Tensor:
        """順伝播.

        Args:
            skeleton_data: (B, C, T, H, W) - カメラ入力（3D ヒートマップ）
            pointcloud_data: (B, T, N, 3) - LiDAR 入力（時系列点群）

        Returns:
            output: (B, num_classes)
        """
        camera_features = self.extract_camera_features(
            skeleton_data
        )  # (B, 512)
        lidar_features = self.extract_lidar_features(
            pointcloud_data
        )  # (B, 512)
        output = self.fusion_head(
            camera_features, lidar_features
        )  # (B, num_classes)

        return output
