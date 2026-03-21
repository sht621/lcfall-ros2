#!/usr/bin/env python3
"""PointNet++ + GRU 転倒検知モデル (LiDAR Branch).

アーキテクチャ:
  1. PointNet++: 各フレーム（点群）から空間的特徴を抽出 (3層)
  2. GRU: フレーム間の時系列特徴を学習 (2層)
  3. Global Max Pooling (時間軸方向)
  4. 分類器: 最終的な行動クラスを予測

パラメータ:
  - num_classes=2, num_points=256, hidden_size=512, num_gru_layers=2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction Layer.

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
        group_all: bool = False,
    ):
        """
        Args:
            npoint: サンプリングする点の数
            radius: グルーピングの半径
            nsample: 各グループ内の点の数
            in_channel: 入力チャネル数
            mlp: MLPの各層のチャネル数
            group_all: すべての点をグループ化するかどうか
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

    def forward(
        self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            new_xyz = self.farthest_point_sample(
                xyz, self.npoint
            )  # (B, 3, npoint)

            # グルーピング: 各サンプル点のK近傍点を集める（KNN）
            grouped_xyz, grouped_points = self.knn_group(
                self.nsample, xyz, new_xyz, points
            )

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
    def farthest_point_sample(
        xyz: torch.Tensor, npoint: int
    ) -> torch.Tensor:
        """Farthest Point Sampling (FPS).

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
        new_xyz = torch.gather(
            xyz, 2, centroids.unsqueeze(1).expand(-1, 3, -1)
        )
        return new_xyz

    @staticmethod
    def knn_group(
        nsample: int,
        xyz: torch.Tensor,
        new_xyz: torch.Tensor,
        points: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """KNN Grouping: 各サンプル点のK近傍点をグルーピング.

        256点という低密度な点群において、常に一定の情報量（nsample個の近傍点）を
        抽出できるよう、Ball Queryの代わりにKNN（K近傍法）を使用します。
        """
        device = xyz.device
        B, _, N = xyz.shape
        _, _, S = new_xyz.shape

        # 距離計算（GPU並列化）
        xyz_t = xyz.transpose(1, 2)  # (B, N, 3)
        new_xyz_t = new_xyz.transpose(1, 2)  # (B, S, 3)

        # バッチごとに距離を計算
        dists = torch.cdist(new_xyz_t, xyz_t)  # (B, S, N)

        # KNN: 最も近いnsample個の点を選択
        _, group_idx = torch.topk(
            dists, k=nsample, dim=2, largest=False
        )  # (B, S, nsample)

        # グループ化
        idx_expanded = group_idx.unsqueeze(1).expand(
            B, 3, S, nsample
        )  # (B, 3, S, nsample)
        xyz_expanded = xyz.unsqueeze(2).expand(
            B, 3, S, N
        )  # (B, 3, S, N)
        grouped_xyz = torch.gather(
            xyz_expanded, 3, idx_expanded
        )  # (B, 3, S, nsample)

        # 相対座標に変換
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(3)  # (B, 3, S, nsample)

        if points is not None:
            C = points.shape[1]
            idx_expanded_points = group_idx.unsqueeze(1).expand(
                B, C, S, nsample
            )
            points_expanded = points.unsqueeze(2).expand(B, C, S, N)
            grouped_points = torch.gather(
                points_expanded, 3, idx_expanded_points
            )
            grouped_points = torch.cat(
                [grouped_xyz, grouped_points], dim=1
            )
        else:
            grouped_points = grouped_xyz

        return grouped_xyz, grouped_points


class PointNet2Encoder(nn.Module):
    """PointNet++ エンコーダー（256点用）.

    点群から階層的な特徴を抽出します。
    低密度点群（256点）に最適化されています。
    """

    def __init__(self, num_points: int = 256):
        super(PointNet2Encoder, self).__init__()

        # Set Abstraction層を3層構築
        # 第1層: 256点 → 128点
        self.sa1 = PointNetSetAbstraction(
            npoint=128,
            radius=0.2,
            nsample=32,
            in_channel=3,
            mlp=[64, 64, 128],
        )

        # 第2層: 128点 → 32点
        self.sa2 = PointNetSetAbstraction(
            npoint=32,
            radius=0.4,
            nsample=32,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
        )

        # 第3層: 32点 → グローバル特徴
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) 点群

        Returns:
            features: (B, 1024) グローバル特徴ベクトル
        """
        xyz1, points1 = self.sa1(xyz, None)
        xyz2, points2 = self.sa2(xyz1, points1)
        _, points3 = self.sa3(xyz2, points2)

        # (B, 1, 1024) → (B, 1024)
        features = points3.squeeze(1)
        return features


class PointNet2GRUModel(nn.Module):
    """PointNet++ + GRU 転倒検知モデル.

    時系列点群データから行動を分類します。
    入力: (B, 48, 256, 3)  各フレーム256点のXYZ座標
    出力: (B, 2)           2クラス分類スコア
    """

    def __init__(
        self,
        num_classes: int = 2,
        num_points: int = 256,
        hidden_size: int = 512,
        num_gru_layers: int = 2,
        dropout: float = 0.5,
    ):
        """
        Args:
            num_classes: クラス数
            num_points: 各フレームの点数
            hidden_size: GRUの隠れ層のサイズ
            num_gru_layers: GRUの層数
            dropout: ドロップアウト率
        """
        super(PointNet2GRUModel, self).__init__()

        self.num_classes = num_classes
        self.num_points = num_points
        self.hidden_size = hidden_size

        # PointNet++エンコーダー（各フレームの空間的特徴を抽出）
        self.pointnet_encoder = PointNet2Encoder(num_points)

        # GRU（時系列特徴を学習）
        self.gru = nn.GRU(
            input_size=1024,  # PointNet++の出力次元
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0,
            bidirectional=False,
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
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, 3) 時系列点群

        Returns:
            output: (B, num_classes) クラスごとのスコア
        """
        batch_size, seq_len, num_points, _ = x.shape

        # 各フレームをPointNet++で処理
        x_reshaped = x.view(batch_size * seq_len, num_points, 3)
        spatial_features = self.pointnet_encoder(x_reshaped)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)

        # GRUで時系列特徴を学習
        gru_out, _ = self.gru(spatial_features)

        # Global Max Pooling（時間軸方向）
        final_features = torch.max(gru_out, dim=1)[0]

        # 分類
        output = self.classifier(final_features)
        return output

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """特徴ベクトルを抽出（分類前）.

        Fusion モデルで使用。GRU → Global Max Pooling 後の
        hidden_size 次元の特徴ベクトルを返す。

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
