# LCFall ROS2 — Camera + LiDAR 転倒検知システム

ROS2 Humble 上で動作するリアルタイム転倒検知パイプライン。
カメラ（RealSense）と LiDAR（Livox MID-360）の Feature-Level-Fusion により、高精度な転倒判定を行う。

---

## システム概要

```
┌──────────────────┐   ┌──────────────────────┐
│  RealSense Camera │   │  Livox MID-360 LiDAR │
│  /camera/image_raw│   │  /livox/lidar         │
└────────┬─────────┘   └──────────┬───────────┘
         │                        │
         └───────────┬────────────┘
                     ▼
        ┌────────────────────────┐
        │  sync_preprocess_node  │  時刻同期 + 1 フレーム前処理
        │  - Skeleton抽出        │  (RTMDet-M + ViTPose-S)
        │  - 座標変換 + 背景差分 │  + 256点整形
        └────────────┬───────────┘
                     │ /preprocessed/frame
                     ▼
        ┌────────────────────────┐
        │    inference_node      │  48 フレーム蓄積
        │  - Heatmap生成         │  + 融合モデル推論
        │  - グローバル正規化     │
        └────────────┬───────────┘
                     │ /fall_detection/result
              ┌──────┴──────┐
              ▼             ▼
    ┌──────────────┐ ┌──────────────────┐
    │  alert_node   │ │ visualization_node│
    │  通知/ログ    │ │ OpenCV 可視化     │
    └──────────────┘ └──────────────────┘
```

## モデルアーキテクチャ

### Camera Branch — PoseC3D (ResNet3dSlowOnly-50)

| 項目 | 値 |
|---|---|
| backbone | `ResNet3dSlowOnly` (depth=50, in_channels=17) |
| cls_head | `I3DHead` (in_channels=512, num_classes=2) |
| 入力 | Heatmap `(B, 17, 48, 56, 56)` (NCTHW) |
| Heatmap sigma | 0.6 |
| キーポイント | 左 8 点 `[1,3,5,7,9,11,13,15]`, 右 8 点 `[2,4,6,8,10,12,14,16]` |

### LiDAR Branch — PointNet++ + GRU

| 項目 | 値 |
|---|---|
| 空間特徴抽出 | PointNet++ 3 層 (256→128→32→global) |
| 時系列学習 | GRU 2 層 (hidden_size=512) |
| 時間集約 | Global Max Pooling |
| 入力 | `(B, 48, 256, 3)` — 48 フレーム × 256 点 × XYZ |
| 出力特徴 | 512 次元 |

### Fusion — Late Fusion MLP

| 項目 | 値 |
|---|---|
| 入力 | Camera 512 次元 + LiDAR 512 次元 = 1024 次元 |
| 構造 | Linear(1024, 512) → BN → ReLU → Dropout(0.5) → Linear(512, 2) |
| 出力 | 2 クラス (class 0 = fall, class 1 = non-fall) |

## パッケージ構成

| パッケージ | 種類 | 概要 |
|---|---|---|
| `lcfall_msgs` | ament_cmake | カスタムメッセージ定義 (PreprocessedFrame, FallDetectionResult) |
| `lcfall_ros2` | ament_python | メインノード群 + ユーティリティ + launch/config |

## メッセージ定義

### PreprocessedFrame.msg
```
std_msgs/Header header
float32[] skeleton_2d      # 51 要素 (17 keypoints × 3)
float32[] pointcloud_frame # 768 要素 (256 points × 3)
```

### FallDetectionResult.msg
```
std_msgs/Header header
uint8 prediction     # 0: non-falling, 1: falling
float32 confidence   # [0.0, 1.0]
```

## 前提環境

- **OS**: Ubuntu 22.04
- **ROS2**: Humble
- **GPU**: NVIDIA CUDA 12.8 対応 GPU
- **カメラ**: Intel RealSense シリーズ
- **LiDAR**: Livox MID-360

## セットアップ

### Docker でのビルド（推奨）

```bash
# 1. Docker イメージをビルド
docker compose build

# 2. コンテナを起動
docker compose up -d

# 3. コンテナに入る
docker compose exec lcfall bash

# 4. ワークスペース内でビルド
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash    # livox_ros_driver2 用

# lcfall_msgs を先にビルド
colcon build --packages-select lcfall_msgs
source install/setup.bash

# lcfall_ros2 をビルド
colcon build --packages-select lcfall_ros2
source install/setup.bash
```

### メッセージ定義の確認

```bash
ros2 interface show lcfall_msgs/msg/PreprocessedFrame
ros2 interface show lcfall_msgs/msg/FallDetectionResult
```

## チェックポイントの配置

推論モデルは 3 つのチェックポイントで構成される。
コンテナ内の `/data/checkpoints/` 以下に配置すること。

```
/data/checkpoints/
├── camera/
│   └── best_model.pth    # PoseC3D (ResNet3dSlowOnly-50)
├── lidar/
│   └── best_model.pth    # PointNet++ + GRU
└── fusion/
    └── best_model.pth    # Fusion Head (Late Fusion MLP)
```

> **Note**: LCFall オフライン学習で生成される fold1 のチェックポイントを使用する場合:
> - Camera: `…/unified_comparison/camera/checkpoints/fold1/best_f1_f1_score_epoch_*.pth`
> - LiDAR: `…/lidar/checkpoints/fold1/best_model.pth`
> - Fusion: `…/fusion/checkpoints/fold1/best_model.pth`

## 実行方法

### システム起動（可視化あり = デフォルト）

センサドライバ＋全ノード＋可視化を一括起動：

```bash
ros2 launch lcfall_ros2 lcfall.launch.py
```

### 可視化なしで起動

```bash
ros2 launch lcfall_ros2 lcfall.launch.py enable_visualization:=false
```

## 事前準備: 背景モデルの取得

転倒検知システムを起動する **前に**、背景モデルを作成する必要がある。
**部屋に人がいない状態** で以下を実行する。

```bash
ros2 launch lcfall_ros2 capture_background.launch.py
```

LiDAR ドライバも自動で起動し、200 フレーム蓄積後に npz 保存 → 全ノード自動終了する。

### パラメータのカスタマイズ

```bash
ros2 launch lcfall_ros2 capture_background.launch.py \
    output_path:=/data/background/background_voxel_map.npz \
    capture_frames:=50 \
    voxel_size:=0.08 \
    min_hits:=15
```

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `output_path` | `/data/background/background_voxel_map.npz` | 保存先 |
| `capture_frames` | 200 | 蓄積フレーム数 |
| `voxel_size` | 0.05 | ボクセル 1 辺 [m] |
| `min_hits` | 5 | 背景とみなす最小出現回数 |
| `roi_*` | 各種 | ROI 範囲 [m]（本体と揃えること） |

### npz 保存内容

| キー | 型 | 説明 |
|---|---|---|
| `voxel_indices` | int (N, 3) | 背景ボクセルの index 群 |
| `voxel_size` | float | ボクセル 1 辺の長さ [m] |
| `roi_min` | float (3,) | ROI 最小値 [x, y, z] |
| `roi_max` | float (3,) | ROI 最大値 [x, y, z] |
| `metadata` | dict | センサ名, 作成日時, パラメータ |

## 主要パラメータ（config/params.yaml）

### sync_preprocess_node

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `sync_slop` | 0.05 | ApproximateTimeSynchronizer の slop [秒] |
| `sync_queue_size` | 10 | 同期キューサイズ |
| `roi_x_min` ~ `roi_z_max` | 各種 | ROI 範囲 [m] |
| `background_model_path` | `/data/background/background_voxel_map.npz` | 背景モデルパス |
| `skeleton_device` | `cuda:0` | skeleton 抽出デバイス |
| `apply_coordinate_transform` | `true` | LiDAR 座標変換の有効/無効 |
| `lidar_roll` | 1.1 | X 軸回転 [deg] |
| `lidar_pitch` | 27.8 | Y 軸回転 [deg]（カメラマウント角度由来） |
| `lidar_yaw` | 0.0 | Z 軸回転 [deg] |

### inference_node

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `inference_stride` | 10 | 推論間隔 (フレーム数) |
| `device` | `cuda:0` | 推論デバイス |
| `camera_config_path` | (install内) | PoseC3D config ファイルパス |
| `camera_checkpoint_path` | `/data/checkpoints/camera/best_model.pth` | Camera チェックポイント |
| `lidar_checkpoint_path` | `/data/checkpoints/lidar/best_model.pth` | LiDAR チェックポイント |
| `fusion_checkpoint_path` | `/data/checkpoints/fusion/best_model.pth` | Fusion チェックポイント |

## 座標変換（LiDAR）

LiDAR センサ座標から部屋座標系への変換は、以下の回転行列で行う:

```
R = Rz(yaw) · Ry(pitch) · Rx(roll)
= Rz(0.0°) · Ry(27.8°) · Rx(1.1°)
```

- **Roll (1.1°)**: X 軸回転
- **Pitch (27.8°)**: Y 軸回転（ZED カメラマウント角度由来）
- **Yaw (0.0°)**: Z 軸回転

`params.yaml` の `lidar_roll`, `lidar_pitch`, `lidar_yaw` で設定可能。
`apply_coordinate_transform: true` でシステム起動時に回転行列が生成され、全フレームに適用される。

## 可視化 UI

`visualization_node` は 1 ウィンドウ内に左右 2 画面を表示する：

| 画面 | 内容 |
|---|---|
| 左 | カメラ画像 + skeleton overlay（緑） |
| 右 | 前景点群の正面投影（緑） |
| 上部 | 転倒時のみ **FALLING** を赤文字表示 |
| 下部 | confidence 値 |

48 フレーム蓄積前は「Waiting for 48 frames」とバッファ数を表示する。

## ノード一覧

| ノード | 入力トピック | 出力トピック |
|---|---|---|
| `sync_preprocess_node` | `/camera/image_raw`, `/livox/lidar` | `/preprocessed/frame` |
| `inference_node` | `/preprocessed/frame` | `/fall_detection/result` |
| `alert_node` | `/fall_detection/result` | — |
| `visualization_node` | `/camera/image_raw`, `/livox/lidar`, `/preprocessed/frame`, `/fall_detection/result` | — |

## TODO / 未確定事項

- [ ] ROI の最終値 (実機環境で調整)

## ライセンス

Apache-2.0
