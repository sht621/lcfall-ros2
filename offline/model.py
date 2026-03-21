#!/usr/bin/env python3
"""
Pointnet++ + GRU 転倒検知モデル

アーキテクチャ:
1. Pointnet++: 各フレーム（点群）から空間的特徴を抽出
2. GRU: フレーム間の時系列特徴を学習
3. 分類器: 最終的な行動クラスを予測
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction Layer
    
    点群から階層的な特徴を抽出します。
    サンプリング → グルーピング → PointNet の3ステップで構成されます。
    """
    
    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: list,
        group_all: bool = False
    ):
        """
        Args:
            npoint (int): サンプリングする点の数
            radius (float): グルーピングの半径
            nsample (int): 各グループ内の点の数
            in_channel (int): 入力チャネル数
            mlp (list): MLPの各層のチャネル数
            group_all (bool): すべての点をグループ化するかどうか
        """
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        # MLP層を構築
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3) 点の座標
            points: (B, N, C) 点の特徴（Noneの場合は座標を使用）
            
        Returns:
            new_xyz: (B, npoint, 3) サンプリングされた点の座標
            new_points: (B, npoint, mlp[-1]) 抽出された特徴
        """
        xyz = xyz.permute(0, 2, 1)  # (B, 3, N)
        if points is not None:
            points = points.permute(0, 2, 1)  # (B, C, N)
        
        if self.group_all:
            # すべての点をグループ化（グローバル特徴抽出）
            new_xyz = xyz[:, :, 0:1]  # (B, 3, 1)
            if points is not None:
                new_points = torch.cat([xyz, points], dim=1)  # (B, 3+C, N)
            else:
                new_points = xyz  # (B, 3, N)
            new_points = new_points.unsqueeze(2)  # (B, 3+C, 1, N)
        else:
            # Farthest Point Sampling (FPS) でサンプリング
            new_xyz = self.farthest_point_sample(xyz, self.npoint)  # (B, 3, npoint)
            
            # グルーピング: 各サンプル点のK近傍点を集める（KNN）
            grouped_xyz, grouped_points = self.knn_group(
                self.nsample, xyz, new_xyz, points
            )
            # grouped_xyz: (B, 3, npoint, nsample)
            # grouped_points: (B, 3+C, npoint, nsample)
            
            new_points = grouped_points
        
        # PointNet: MLPで特徴抽出
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        # Max pooling: 各グループから最大値を取得
        new_points = torch.max(new_points, -1)[0]  # (B, mlp[-1], npoint)
        new_xyz = new_xyz.permute(0, 2, 1)  # (B, npoint, 3)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, mlp[-1])
        
        return new_xyz, new_points
    
    @staticmethod
    def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Farthest Point Sampling (FPS)
        
        点群から最も離れた点を順次サンプリングします。
        """
        device = xyz.device
        B, _, N = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, :, farthest].view(B, 3, 1)
            dist = torch.sum((xyz - centroid) ** 2, dim=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, dim=1)[1]
        
        # サンプリングされた点の座標を取得
        new_xyz = torch.gather(xyz, 2, centroids.unsqueeze(1).expand(-1, 3, -1))
        return new_xyz
    
    @staticmethod
    def knn_group(
        nsample: int,
        xyz: torch.Tensor,
        new_xyz: torch.Tensor,
        points: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        KNN Grouping: 各サンプル点のK近傍点をグルーピング（GPU最適化版）
        
        256点という低密度な点群において、常に一定の情報量（nsample個の近傍点）を
        抽出できるよう、Ball Queryの代わりにKNN（K近傍法）を使用します。
        """
        device = xyz.device
        B, _, N = xyz.shape
        _, _, S = new_xyz.shape
        
        # 距離計算（GPU並列化）
        # xyz: (B, 3, N) -> (B, N, 3)
        # new_xyz: (B, 3, S) -> (B, S, 3)
        xyz_t = xyz.transpose(1, 2)  # (B, N, 3)
        new_xyz_t = new_xyz.transpose(1, 2)  # (B, S, 3)
        
        # バッチごとに距離を計算（GPU並列化）
        # dists: (B, S, N) - 各サンプル点から全点への距離
        dists = torch.cdist(new_xyz_t, xyz_t)  # (B, S, N)
        
        # KNN: 最も近いnsample個の点を選択
        # (B, S, N) -> (B, S, nsample)
        _, group_idx = torch.topk(dists, k=nsample, dim=2, largest=False)  # (B, S, nsample)
        
        # グループ化（GPU並列化）
        # group_idx: (B, S, nsample)
        # xyz: (B, 3, N) -> grouped_xyz: (B, 3, S, nsample)
        
        # インデックスを展開
        idx_expanded = group_idx.unsqueeze(1).expand(B, 3, S, nsample)  # (B, 3, S, nsample)
        
        # xyzを展開してgather
        xyz_expanded = xyz.unsqueeze(2).expand(B, 3, S, N)  # (B, 3, S, N)
        grouped_xyz = torch.gather(xyz_expanded, 3, idx_expanded)  # (B, 3, S, nsample)
        
        # 相対座標に変換
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(3)  # (B, 3, S, nsample)
        
        if points is not None:
            # 特徴量もグループ化
            C = points.shape[1]
            idx_expanded_points = group_idx.unsqueeze(1).expand(B, C, S, nsample)
            points_expanded = points.unsqueeze(2).expand(B, C, S, N)
            grouped_points = torch.gather(points_expanded, 3, idx_expanded_points)
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=1)
        else:
            grouped_points = grouped_xyz
        
        return grouped_xyz, grouped_points


class PointNet2Encoder(nn.Module):
    """
    PointNet++ エンコーダー（256点用）
    
    点群から階層的な特徴を抽出します。
    低密度点群（256点）に最適化されています。
    """
    
    def __init__(self, num_points: int = 256):
        """
        Args:
            num_points (int): 入力点群の点数（デフォルト: 256）
        """
        super(PointNet2Encoder, self).__init__()
        
        # Set Abstraction層を3層構築
        # 第1層: 256点 → 128点
        self.sa1 = PointNetSetAbstraction(
            npoint=128,
            radius=0.2,  # 密度が下がったため半径を広げる
            nsample=32,
            in_channel=3,  # xyz座標のみ
            mlp=[64, 64, 128]
        )
        
        # 第2層: 128点 → 32点
        self.sa2 = PointNetSetAbstraction(
            npoint=32,
            radius=0.4,  # KNN使用のため参考値
            nsample=32,  # 256点用に最適化: 64→32
            in_channel=128 + 3,  # 前層の特徴 + xyz
            mlp=[128, 128, 256]
        )
        
        # 第3層: 32点 → グローバル特徴
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True
        )
    
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) 点群
            
        Returns:
            features: (B, 1024) グローバル特徴ベクトル
        """
        # 第1層
        xyz1, points1 = self.sa1(xyz, None)
        
        # 第2層
        xyz2, points2 = self.sa2(xyz1, points1)
        
        # 第3層（グローバル特徴）
        _, points3 = self.sa3(xyz2, points2)
        
        # (B, 1, 1024) → (B, 1024)
        features = points3.squeeze(1)
        
        return features


class PointNet2GRUModel(nn.Module):
    """
    Pointnet++ + GRU 転倒検知モデル
    
    時系列点群データから行動を分類します。
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        num_points: int = 256,
        hidden_size: int = 512,
        num_gru_layers: int = 2,
        dropout: float = 0.5
    ):
        """
        Args:
            num_classes (int): クラス数
            num_points (int): 各フレームの点数
            hidden_size (int): GRUの隠れ層のサイズ
            num_gru_layers (int): GRUの層数
            dropout (float): ドロップアウト率
        """
        super(PointNet2GRUModel, self).__init__()
        
        self.num_classes = num_classes
        self.num_points = num_points
        self.hidden_size = hidden_size
        
        # Pointnet++エンコーダー（各フレームの空間的特徴を抽出）
        self.pointnet_encoder = PointNet2Encoder(num_points)
        
        # GRU（時系列特徴を学習）
        # 入力: (sequence_length, batch_size, 1024)
        # 出力: (sequence_length, batch_size, hidden_size)
        self.gru = nn.GRU(
            input_size=1024,  # Pointnet++の出力次元
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,  # (batch, seq, feature)
            dropout=dropout if num_gru_layers > 1 else 0,
            bidirectional=False
        )
        
        # 分類器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, 3) 時系列点群
                B: バッチサイズ
                T: シーケンス長（フレーム数）
                N: 点数
                3: xyz座標
                
        Returns:
            output: (B, num_classes) クラスごとのスコア
        """
        batch_size, seq_len, num_points, _ = x.shape
        
        # 各フレームをPointnet++で処理
        # (B, T, N, 3) → (B*T, N, 3)
        x_reshaped = x.view(batch_size * seq_len, num_points, 3)
        
        # Pointnet++で空間的特徴を抽出
        # (B*T, N, 3) → (B*T, 1024)
        spatial_features = self.pointnet_encoder(x_reshaped)
        
        # (B*T, 1024) → (B, T, 1024)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)
        
        # GRUで時系列特徴を学習
        # (B, T, 1024) → (B, T, hidden_size)
        gru_out, _ = self.gru(spatial_features)
        
        # Global Max Pooling（時間軸方向）
        # シーケンス内の転倒のピーク情報を逃さず捉えるため、
        # 最終フレームのみではなく全フレームの出力からmax poolingを適用
        # (B, T, hidden_size) → (B, hidden_size)
        final_features = torch.max(gru_out, dim=1)[0]
        
        # 分類
        # (B, hidden_size) → (B, num_classes)
        output = self.classifier(final_features)
        
        return output
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        特徴ベクトルを抽出（分類前）
        
        Args:
            x: (B, T, N, 3) 時系列点群
            
        Returns:
            features: (B, hidden_size) 特徴ベクトル
        """
        batch_size, seq_len, num_points, _ = x.shape
        
        x_reshaped = x.view(batch_size * seq_len, num_points, 3)
        spatial_features = self.pointnet_encoder(x_reshaped)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)
        
        gru_out, _ = self.gru(spatial_features)
        # Global Max Pooling（時間軸方向）
        final_features = torch.max(gru_out, dim=1)[0]
        
        return final_features


def create_model(
    num_classes: int = 2,
    num_points: int = 1024,
    hidden_size: int = 512,
    num_gru_layers: int = 2,
    dropout: float = 0.5
) -> PointNet2GRUModel:
    """
    モデルを作成
    
    Args:
        num_classes (int): クラス数
        num_points (int): 各フレームの点数
        hidden_size (int): GRUの隠れ層のサイズ
        num_gru_layers (int): GRUの層数
        dropout (float): ドロップアウト率
        
    Returns:
        PointNet2GRUModel: モデル
    """
    model = PointNet2GRUModel(
        num_classes=num_classes,
        num_points=num_points,
        hidden_size=hidden_size,
        num_gru_layers=num_gru_layers,
        dropout=dropout
    )
    return model


if __name__ == '__main__':
    """
    モデルのテスト
    """
    # モデルを作成（2値分類）
    model = create_model(
        num_classes=2,
        num_points=1024,
        hidden_size=512,
        num_gru_layers=2,
        dropout=0.5
    )
    
    print("モデル構造:")
    print(model)
    print()
    
    # パラメータ数を計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"総パラメータ数: {total_params:,}")
    print(f"学習可能パラメータ数: {trainable_params:,}")
    print()
    
    # テスト入力
    batch_size = 4
    seq_len = 16
    num_points = 1024
    
    x = torch.randn(batch_size, seq_len, num_points, 3)
    print(f"入力形状: {x.shape}")
    
    # 順伝播
    output = model(x)
    print(f"出力形状: {output.shape}")
    print(f"出力: {output}")
    
    # 特徴抽出
    features = model.extract_features(x)
    print(f"\n特徴ベクトル形状: {features.shape}")
    
    print("\nモデルのテストが完了しました！")

