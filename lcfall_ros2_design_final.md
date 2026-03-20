# オンライン転倒検知システム設計メモ 最終版
（ROS2 Humble / Camera + LiDAR Late Fusion）

## 1. 目的

既存のオフライン転倒検知パイプライン（カメラ + LiDAR の Late Fusion）を、ROS2 Humble 上で動作するリアルタイム推論システムとして再構成する。

本書は、これまでの議論で**確定した事項のみ**を残した最終版である。

---

## 2. システム前提

- OS / ROS2: Ubuntu 22.04 / ROS2 Humble
- PyTorch: 2.1.0 + CUDA 12.1
- コンテナ環境: `Dockerfile.ros` ベース
- 推論方式: カメラ + LiDAR のマルチモーダル Late Fusion
- センサドライバは自作せず、外部 ROS2 ドライバを使用する
- メインパッケージ名: **`lcfall_ros2`**
- メッセージパッケージ名: **`lcfall_msgs`**

---

## 3. センサ入出力方針

### 3.1 カメラ

- カメラドライバは外部 ROS2 ノードを使用する
- 入力トピック: `/camera/image_raw`
- 画像サイズ: **1280 x 720**
- カメラ publish rate: **30 FPS**

### 3.2 LiDAR

- LiDAR は **Livox MID-360** を使用する
- ドライバは **`livox_ros_driver2`** を使用する
- 入力形式は **`sensor_msgs/PointCloud2`** で固定する
- Livox カスタム形式は採用しない
- 点ごとの timestamp は扱わない
- 時刻同期は `header.stamp` ベースで行う
- 想定入力トピック: `/livox/lidar`

---

## 4. モデルと推論条件

- カメラ前処理モデル: **RTMDet-M + ViTPose-S**
- 時系列窓長: **48 フレーム**
- 推論出力頻度: **約 1 秒に 1 回**
- 初期値: `inference_stride = 10`
- 欠損時はゼロ埋めで推論継続
- Alert は推論結果に素直に従う
- `ApproximateTimeSynchronizer` の初期値: `queue_size = 10`, `slop = 0.05`

### 4.1 Alert 判定

- `prediction == falling` → `ALERT`
- `prediction == non-falling` → `NORMAL`

### 4.2 推論モデル内部構造の扱い

融合モデルの内部構造詳細は、後で実装・実験に合わせて調整する。
本書では、ノード間責務・入出力・前処理フローの確定を優先し、内部ネットワーク詳細は固定しない。

---

## 5. ノード構成

| # | ノード名 | 役割 | 入力 | 出力 |
|---|---|---|---|---|
| 1 | 外部カメラドライバ | カメラ画像取得 | — | `/camera/image_raw` |
| 2 | 外部 LiDAR ドライバ (`livox_ros_driver2`) | MID-360 点群取得 | — | `/livox/lidar` |
| 3 | `sync_preprocess_node` | 時刻同期、1フレーム単位前処理、前処理済みペア publish | `/camera/image_raw`, `/livox/lidar` | `/preprocessed/frame` |
| 4 | `inference_node` | 48フレーム蓄積、heatmap 生成、LiDAR グローバル正規化、融合推論 | `/preprocessed/frame` | `/fall_detection/result` |
| 5 | `alert_node` | Alert 判定と通知 | `/fall_detection/result` | — |
| 6 | `visualization_node` | デモ可視化 | `/camera/image_raw`, `/livox/lidar`, `/fall_detection/result` | — |

### 5.1 責務分担の確定

**48 フレームのリングバッファは `inference_node` 側で保持する。**
`sync_preprocess_node` は各同期ペアに対して 1 フレーム分の前処理だけを行い、前処理済み 1 フレームペアを publish する。

この構成にする理由は次のとおり。

- カメラ側は **skeleton 座標だけ送ればよく、heatmap 化は `inference_node` 側で行える**
- LiDAR のグローバル正規化は 48 フレーム窓がそろった時点で `inference_node` 側で実施できる
- 48 フレーム窓を丸ごと再送すると、重複フレームを毎回再送するため ROS 通信量が増える
- `sync_preprocess_node` を「同期 + 1フレーム前処理」に限定できる
- `inference_node` を「時系列窓の管理 + heatmap 化 + 推論」にまとめた方が自然である

---

## 6. カスタムメッセージ定義

### 6.1 採用方針

前処理済みフレームメッセージは、**必要最小限の情報だけ**を持つ。

- カメラ側は **1人分の 2D skeleton 座標**のみ送る
- LiDAR 側は **背景差分後の前景生点群 256 点**のみ送る
- heatmap は送らない
- 可変ではない固定寸法のメタデータは、メッセージから除外する

### 6.2 `lcfall_msgs/PreprocessedFrame.msg`

```msg
std_msgs/Header header

# 1人分の 2D skeleton: (17, 3) = (x_norm, y_norm, score) を flatten
# x_norm, y_norm は [0, 1] 正規化座標
float32[] skeleton_2d

# 背景差分後の前景生点群: (256, 3) を flatten
# この時点ではまだグローバル正規化しない
float32[] pointcloud_frame
```

#### 形状の約束

- `skeleton_2d`: 長さ **51** 固定（17 keypoints × 3）
- `pointcloud_frame`: 長さ **768** 固定（256 points × 3）

#### `PreprocessedFrame.msg` に入れないもの

以下は初版では不要なので送らない。

- `heatmap_channels`, `heatmap_height`, `heatmap_width`
- `pointcloud_num_points`, `pointcloud_dims`
- bbox 情報
- 元画像サイズ
- `camera_valid`, `lidar_valid`

元画像サイズを入れない理由は、skeleton 座標を **正規化座標**で送るためである。

### 6.3 `lcfall_msgs/FallDetectionResult.msg`

```msg
std_msgs/Header header

uint8 prediction
float32 confidence
```

#### `FallDetectionResult.msg` に入れないもの

以下は初版では不要なので送らない。

- `class_probs`
- `model_name`
- `camera_used`, `lidar_used`
- `inference_time_ms`

### 6.4 メッセージ定義の妥当性確認

#### `PreprocessedFrame.msg`

必要十分であり、冗長項目はない。

- **必要**: `header`, `skeleton_2d`, `pointcloud_frame`

#### `FallDetectionResult.msg`

初版用途としては最小構成で十分である。

- UI 表示: `prediction`, `confidence`
- alert 判定: `prediction`

---

## 7. 背景モデル

### 7.1 取得方針

背景モデルは、転倒検知システム本体とは独立した**別プログラム**で事前取得する。
本体起動中に背景取得や背景更新は行わない。

### 7.2 保存形式

背景モデルは **`npz`** 形式で保存する。

保存内容の初期案:

- `voxel_indices`: 背景ボクセル index 群
- `voxel_size`: ボクセルサイズ
- `roi_min`: ROI 最小値 `[x_min, y_min, z_min]`
- `roi_max`: ROI 最大値 `[x_max, y_max, z_max]`
- `metadata`: センサ名、作成日時、任意メモ

### 7.3 本体起動時の扱い

- `sync_preprocess_node` は起動時に背景モデルファイルを読み込む
- 背景モデルが存在しない場合はエラー終了とする
- 運用中に背景モデルは更新しない

---

## 8. アーキテクチャ採用方針

### 8.1 採用方式

**1 フレーム単位で前処理済みペアを publish し、`inference_node` で 48 フレームを束ねる方式を採用する。**

### 8.2 採用理由

- Pose 推定が支配的な計算負荷であり、skeleton 座標の整形や LiDAR 正規化の負荷は比較的小さい
- 48 フレーム窓を丸ごと再送すると、ROS 通信負荷が無視しづらい
- skeleton 座標を送る設計にすれば、heatmap 再送より通信量を大幅に削減できる
- `sync_preprocess_node` の責務を軽く保てる
- `inference_node` に時系列管理を集中させた方が構造が明快

---

## 9. `sync_preprocess_node` 詳細設計

### 9.1 役割

- カメラと LiDAR の時刻同期
- カメラ画像から 2D keypoint 抽出
- LiDAR 点群の 1 フレーム前処理
- `PreprocessedFrame` として publish

### 9.2 購読トピック

| トピック | 型 |
|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` |
| `/livox/lidar` | `sensor_msgs/PointCloud2` |

### 9.3 同期方針

```python
from message_filters import Subscriber, ApproximateTimeSynchronizer

image_sub = Subscriber(self, Image, '/camera/image_raw')
lidar_sub = Subscriber(self, PointCloud2, '/livox/lidar')

sync = ApproximateTimeSynchronizer(
    [image_sub, lidar_sub],
    queue_size=10,
    slop=0.05,
)
sync.registerCallback(self.sync_callback)
```

---

## 10. カメラ前処理

- Detector: **RTMDet-M**
- Pose: **ViTPose-S**
- 各同期ペアごとに 2D keypoint を抽出する
- 送信するのは **1人分の 2D skeleton 座標**のみ
- skeleton 座標は `(x_norm, y_norm, score)` で表し、`[0,1]` 正規化して送る
- 検出失敗フレームは全ゼロ skeleton を使う

### 初期化イメージ

```python
from mmpose.apis import MMPoseInferencer

self.pose_inferencer = MMPoseInferencer(
    pose2d='vitpose-s',
    det_model='rtmdet_m_8xb32-300e_coco',
    det_cat_ids=[0],
    device='cuda:0',
)
```

### 補足

heatmap 生成は `sync_preprocess_node` では行わない。
`inference_node` で 48 フレーム分の `skeleton_2d` を蓄積した後、`(17, 48, 56, 56)` の heatmap 列を構成する。

---

## 11. LiDAR 前処理

### 11.1 `sync_preprocess_node` で行う処理

```text
PointCloud2
  -> numpy (N, 3)
  -> 必要に応じた座標変換 / 回転補正
  -> ROI 範囲指定
  -> ボクセル背景差分で背景点を判定
  -> 背景に該当する点を元の生点群から除去
  -> 残った前景点群（人物点群）をそのまま使用
  -> 256 点へ整形
  -> 1 フレーム分として publish
```

### 11.2 背景差分の意味

背景差分ではボクセルを背景判定の内部表現として使うが、**出力として保持するのはボクセル中心点ではなく元の生点群**である。

つまり、各点について

1. その点が属するボクセルを求める
2. 背景 occupancy に含まれていれば除去する
3. 含まれていなければ、その元の点座標をそのまま残す

という処理を行う。

### 11.3 初期パラメータ案

```yaml
background_model_path: /data/background/background_voxel_map.npz
background_capture_frames: 30
background_voxel_size: 0.10
background_min_hits: 10
```

上記は一般的な初期値であり、実機で調整する。

### 11.4 点群整形ルール

前景点群の点数を `M` としたとき、整形ルールは以下で確定する。

- `M > 256`: ランダムサンプリングで 256 点に減らす
- `M = 256`: そのまま使う
- `0 < M < 256`: 重複サンプリングで 256 点に揃える
- `M = 0`: 全点 0 埋め

---

## 12. `inference_node`

### 12.1 役割

- `PreprocessedFrame` を受け取る
- 48 フレームのリングバッファを保持する
- `inference_stride` ごとに 48 フレーム窓を取り出す
- `skeleton_2d` 列から heatmap 列を生成する
- LiDAR 列に対してグローバル正規化を実施する
- 融合モデルで推論する
- `FallDetectionResult` を publish する

### 12.2 heatmap 生成

- 48 フレーム分の `skeleton_2d` を stack する
- `(17, T, 3)` から `(17, 48, 56, 56)` の heatmap 列を生成する
- 検出失敗フレームはゼロ skeleton なので、そのフレームの heatmap もゼロになる

### 12.3 グローバル正規化

LiDAR 正規化は、**48 フレーム窓の 1 フレーム目を基準としたグローバル正規化**で行う。

- 窓の 1 フレーム目が空点群 (`M = 0`) の場合は、次の非空フレームを基準フレームとする
- 窓全体が空点群の場合は、ゼロ埋め系列のまま扱う

### 12.4 処理負荷の考え方

- **重いのは RTMDet-M + ViTPose-S の姿勢推定**
- 48 フレーム分の skeleton 座標から heatmap を作る処理は比較的軽い
- 48 × 256 × 3 の点群列に対するグローバル正規化も軽い
- したがって、heatmap 生成とグローバル正規化を `inference_node` 側に持たせても、全体の支配的負荷にはなりにくい

---

## 13. `alert_node`

基本ロジック:

```python
if msg.prediction == 1:
    self.notify_fall()
else:
    pass
```

- 連続回数判定は使わない
- confidence 閾値による抑制も初版では行わない
- `alert_node` は `/fall_detection/result` を購読し、通知やブザー制御のみを担当する
- `alert_node` は新たなトピックを publish しない

---

## 14. `visualization_node`

### 14.1 UI 構成

- 1 ウィンドウ内の左右 2 画面構成
- 左: 画像 + スケルトン重畳
- 右: 点群の正面投影
- スケルトンは緑で描画
- 点群は緑で描画
- 転倒判定時のみ `FALLING` を文字表示する

### 14.2 待機中表示

48 フレーム蓄積前は以下のみ表示する。

- 左: カメラ画像 + スケルトン
- 右: 点群正面投影
- `Waiting for 48 frames`
- バッファ充填状況

### 14.3 推論開始後表示

- 左: カメラ画像 + スケルトン
- 右: 点群正面投影
- 転倒時のみ `FALLING`
- confidence

---

## 15. パッケージ構成

`bringup` パッケージは分離せず、launch / config は **`lcfall_ros2`** に含める。

```text
ros2_ws/src/
├── lcfall_msgs/
│   ├── msg/
│   │   ├── PreprocessedFrame.msg
│   │   └── FallDetectionResult.msg
│   ├── CMakeLists.txt
│   └── package.xml
│
└── lcfall_ros2/
    ├── lcfall_ros2/
    │   ├── __init__.py
    │   ├── sync_preprocess_node.py
    │   ├── inference_node.py
    │   ├── alert_node.py
    │   ├── visualization_node.py
    │   └── utils/
    │       ├── lidar_preprocessing.py
    │       ├── background_subtraction.py
    │       ├── skeleton_extraction.py
    │       ├── heatmap_generation.py
    │       ├── ring_buffer.py
    │       └── tensor_utils.py
    ├── launch/
    │   ├── lcfall.launch.py
    │   ├── demo.launch.py
    │   └── livox_mid360.launch.py
    ├── config/
    │   ├── params.yaml
    │   └── livox_mid360_config.json
    ├── resource/
    ├── setup.py
    ├── setup.cfg
    └── package.xml
```

---

## 16. 現時点で残る論点

- ボクセル背景差分の実機パラメータ調整
- ROI の最終値
- `skeleton_2d` から heatmap を生成する実装方法の詳細
- 融合モデル内部構造の最終調整
