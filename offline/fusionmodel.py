#!/usr/bin/env python3
"""
カメラ・LiDAR融合転倒検知モデル

Late Fusion方式:
- Stream A: PoseC3D (ResNet3dSlowOnly-50) → 512次元特徴
- Stream B: PointNet++ + GRU → 512次元特徴
- Fusion Head: 連結(1024次元) → MLP → 2クラス分類
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import sys
from pathlib import Path

# LiDARモデルのインポート
# 注意: LOSO実験で学習されたモデルと同じ定義を使用する必要がある
_lidar_model_path = str(Path(__file__).parent.parent / 'lidar' / 'loso_training')
if _lidar_model_path not in sys.path:
    sys.path.insert(0, _lidar_model_path)
from lidar.loso_training.model import PointNet2GRUModel as LiDARModel

# MMAction2のインポート
try:
    from mmaction.registry import MODELS
    from mmengine.config import Config
except ImportError:
    print("Warning: MMAction2 not found. Camera model will not be available.")
    MODELS = None
    Config = None


class FusionHead(nn.Module):
    """
    融合ヘッド（MLP）
    
    カメラとLiDARの特徴を連結し、最終的な分類を行う。
    """
    
    def __init__(
        self,
        camera_feature_dim: int = 512,
        lidar_feature_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        """
        Args:
            camera_feature_dim: カメラ特徴の次元数
            lidar_feature_dim: LiDAR特徴の次元数
            num_classes: クラス数
            dropout: ドロップアウト率
        """
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
            nn.Linear(512, num_classes)
        )
    
    def forward(self, camera_features: torch.Tensor, lidar_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            camera_features: (B, camera_feature_dim)
            lidar_features: (B, lidar_feature_dim)
            
        Returns:
            output: (B, num_classes)
        """
        # 特徴を連結
        fused_features = torch.cat([camera_features, lidar_features], dim=1)  # (B, 1024)
        
        # 分類
        output = self.fusion_mlp(fused_features)  # (B, num_classes)
        
        return output


class CameraLiDARFusionModel(nn.Module):
    """
    カメラ・LiDAR融合転倒検知モデル
    
    Late Fusion方式で2つのストリームを統合:
    - Stream A: PoseC3D (カメラ)
    - Stream B: PointNet++ + GRU (LiDAR)
    """
    
    def __init__(
        self,
        camera_config_path: str,
        camera_checkpoint_path: str,
        lidar_checkpoint_path: str,
        num_classes: int = 2,
        freeze_backbones: bool = True,
        dropout: float = 0.5
    ):
        """
        Args:
            camera_config_path: PoseC3Dの設定ファイルパス
            camera_checkpoint_path: PoseC3Dのチェックポイントパス
            lidar_checkpoint_path: LiDARモデルのチェックポイントパス
            num_classes: クラス数
            freeze_backbones: バックボーンを固定するか
            dropout: Fusion Headのドロップアウト率
        """
        super(CameraLiDARFusionModel, self).__init__()
        
        self.num_classes = num_classes
        self.freeze_backbones = freeze_backbones
        
        # カメラモデル（PoseC3D）をロード
        self.camera_model = self._load_camera_model(camera_config_path, camera_checkpoint_path)
        
        # LiDARモデル（PointNet++ + GRU）をロード
        self.lidar_model = self._load_lidar_model(lidar_checkpoint_path)
        
        # バックボーンを固定
        if freeze_backbones:
            self._freeze_backbones()
        
        # Fusion Head
        self.fusion_head = FusionHead(
            camera_feature_dim=512,
            lidar_feature_dim=512,
            num_classes=num_classes,
            dropout=dropout
        )
    
    def _load_camera_model(self, config_path: str, checkpoint_path: str) -> nn.Module:
        """
        PoseC3Dモデルをロード
        """
        if MODELS is None or Config is None:
            raise ImportError("MMAction2 is required for camera model")
        
        # 設定ファイルをロード
        cfg = Config.fromfile(config_path)
        
        # モデルを構築
        model = MODELS.build(cfg.model)
        
        # チェックポイントをロード
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        return model
    
    def _load_lidar_model(self, checkpoint_path: str) -> nn.Module:
        """
        LiDARモデル（PointNet++ + GRU）をロード
        """
        # モデルを構築（256点、hidden_size=512）
        model = LiDARModel(
            num_classes=2,
            num_points=256,
            hidden_size=512,
            num_gru_layers=2,
            dropout=0.5
        )
        
        # チェックポイントをロード
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # module. プレフィックスを除去（DataParallel対応）
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        return model
    
    def _freeze_backbones(self):
        """
        バックボーン（PoseC3D、PointNet++ + GRU）を固定
        """
        # カメラモデルを固定
        for param in self.camera_model.parameters():
            param.requires_grad = False
        
        # LiDARモデルを固定
        for param in self.lidar_model.parameters():
            param.requires_grad = False
        
        print("✓ Backbones frozen (Camera + LiDAR)")
    
    def extract_camera_features(self, skeleton_data: torch.Tensor) -> torch.Tensor:
        """
        カメラ特徴を抽出（PoseC3D backbone）
        
        Args:
            skeleton_data: (B, C, T, H, W) - 3Dヒートマップ
            
        Returns:
            features: (B, 512)
        """
        # PoseC3Dのbackboneで特徴抽出
        # backbone.forward() → (B, 512, T', H', W')
        features = self.camera_model.backbone(skeleton_data)
        
        # Global Average Pooling
        # (B, 512, T', H', W') → (B, 512)
        if isinstance(features, tuple):
            features = features[-1]  # 最後の出力を使用
        
        # (B, C, T, H, W) → (B, C)
        features = features.mean(dim=[2, 3, 4])  # 時間・空間次元で平均
        
        return features
    
    def extract_lidar_features(self, pointcloud_data: torch.Tensor) -> torch.Tensor:
        """
        LiDAR特徴を抽出（PointNet++ + GRU）
        
        Args:
            pointcloud_data: (B, T, N, 3) - 時系列点群
            
        Returns:
            features: (B, 512)
        """
        # LiDARモデルのextract_features()を使用
        features = self.lidar_model.extract_features(pointcloud_data)
        
        return features
    
    def forward(
        self,
        skeleton_data: torch.Tensor,
        pointcloud_data: torch.Tensor
    ) -> torch.Tensor:
        """
        順伝播
        
        Args:
            skeleton_data: (B, C, T, H, W) - カメラ入力（3Dヒートマップ）
            pointcloud_data: (B, T, N, 3) - LiDAR入力（時系列点群）
            
        Returns:
            output: (B, num_classes)
        """
        # カメラ特徴を抽出
        camera_features = self.extract_camera_features(skeleton_data)  # (B, 512)
        
        # LiDAR特徴を抽出
        lidar_features = self.extract_lidar_features(pointcloud_data)  # (B, 512)
        
        # 融合して分類
        output = self.fusion_head(camera_features, lidar_features)  # (B, num_classes)
        
        return output
    
    def get_trainable_parameters(self):
        """
        学習可能なパラメータのみを返す（Fusion Headのみ）
        """
        return [p for p in self.fusion_head.parameters() if p.requires_grad]
    
    def get_num_trainable_params(self) -> int:
        """
        学習可能なパラメータ数を返す
        """
        return sum(p.numel() for p in self.get_trainable_parameters())


def create_fusion_model(
    camera_config_path: str,
    camera_checkpoint_path: str,
    lidar_checkpoint_path: str,
    num_classes: int = 2,
    freeze_backbones: bool = True,
    dropout: float = 0.5
) -> CameraLiDARFusionModel:
    """
    融合モデルを作成
    
    Args:
        camera_config_path: PoseC3Dの設定ファイルパス
        camera_checkpoint_path: PoseC3Dのチェックポイントパス
        lidar_checkpoint_path: LiDARモデルのチェックポイントパス
        num_classes: クラス数
        freeze_backbones: バックボーンを固定するか
        dropout: Fusion Headのドロップアウト率
        
    Returns:
        model: 融合モデル
    """
    model = CameraLiDARFusionModel(
        camera_config_path=camera_config_path,
        camera_checkpoint_path=camera_checkpoint_path,
        lidar_checkpoint_path=lidar_checkpoint_path,
        num_classes=num_classes,
        freeze_backbones=freeze_backbones,
        dropout=dropout
    )
    
    print(f"✓ Fusion model created")
    print(f"  - Camera: PoseC3D (ResNet3dSlowOnly-50)")
    print(f"  - LiDAR: PointNet++ + GRU (256 points)")
    print(f"  - Fusion: Late Fusion (MLP)")
    print(f"  - Trainable params: {model.get_num_trainable_params():,}")
    
    return model


if __name__ == '__main__':
    # テスト用コード
    print("Testing Fusion Model...")
    
    # ダミーデータ
    batch_size = 4
    seq_len = 48
    num_points = 256
    
    # カメラ入力（3Dヒートマップ）: (B, C, T, H, W)
    skeleton_data = torch.randn(batch_size, 17, seq_len, 56, 56)
    
    # LiDAR入力（時系列点群）: (B, T, N, 3)
    pointcloud_data = torch.randn(batch_size, seq_len, num_points, 3)
    
    # Fusion Headのみをテスト
    print("\n1. Testing Fusion Head...")
    fusion_head = FusionHead()
    camera_features = torch.randn(batch_size, 512)
    lidar_features = torch.randn(batch_size, 512)
    output = fusion_head(camera_features, lidar_features)
    print(f"   Input: camera={camera_features.shape}, lidar={lidar_features.shape}")
    print(f"   Output: {output.shape}")
    print(f"   ✓ Fusion Head OK")
    
    print("\nFusion Model test completed!")

