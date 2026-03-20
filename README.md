# LCFall ROS2 — Camera + LiDAR Late Fusion 転倒検知システム

ROS2 Humble 上で動作するリアルタイム転倒検知パイプライン。
カメラ（RealSense）と LiDAR（Livox MID-360）の Late Fusion により、高精度な転倒判定を行う。

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
        │  - 背景差分 + 256点整形 │
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
- **GPU**: NVIDIA CUDA 12.1 対応 GPU
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

### パラメータのカスタマイズ

```bash
ros2 launch lcfall_ros2 lcfall.launch.py params_file:=/path/to/custom_params.yaml
```

## 事前準備: 背景モデルの取得

転倒検知システムを起動する **前に**、背景モデルを作成する必要がある。
**部屋に人がいない状態** で以下を実行する。

```bash
ros2 launch lcfall_ros2 capture_background.launch.py
```

LiDAR ドライバも自動で起動し、30 フレーム蓄積後に npz 保存 → 全ノード自動終了する。

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
| `capture_frames` | 30 | 蓄積フレーム数 |
| `voxel_size` | 0.10 | ボクセル 1 辺 [m] |
| `min_hits` | 10 | 背景とみなす最小出現回数 |
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

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `sync_slop` | 0.05 | ApproximateTimeSynchronizer の slop [秒] |
| `sync_queue_size` | 10 | 同期キューサイズ |
| `inference_stride` | 10 | 推論間隔 (フレーム数) |
| `roi_x_min` ~ `roi_z_max` | 各種 | ROI 範囲 [m] |
| `background_model_path` | `/data/background/background_voxel_map.npz` | 背景モデルパス |
| `skeleton_device` | `cuda:0` | skeleton 抽出デバイス |

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

- [ ] 融合モデルの内部構造と学習済みモデルの配置
- [ ] 座標変換パラメータ (Livox MID-360 の設置角度に依存)
- [ ] ROI の最終値 (実機環境で調整)

## ライセンス

Apache-2.0
