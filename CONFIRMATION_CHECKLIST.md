# 動作確認チェックリスト

## 0. 事前確認

ワークスペースとROS環境を読み込む。

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

最低限の前提を確認する。

```bash
ros2 pkg executables lcfall_ros2
ls /data/checkpoints/camera/best_model.pth
ls /data/checkpoints/lidar/best_model.pth
ls /data/checkpoints/fusion/best_model.pth
ls /data/background/background_voxel_map.npz
```

期待値:

- `lcfall_ros2` の実行可能ノードが 5 つ見える
- 背景モデル 1 件と学習済みモデル 3 件が存在する

主なトピック:

- カメラ: `/camera/image_raw`
- LiDAR: `/livox/lidar`
- 前処理結果: `/preprocessed/frame`
- 転倒検知結果: `/fall_detection/result`

## 1. センサの動作確認

### 1-1. カメラ単体

```bash
ros2 run realsense2_camera realsense2_camera_node --ros-args \
  -p camera_name:=camera \
  -p rgb_camera.color_format:=RGB8 \
  -p rgb_camera.profile:=1280x720x30 \
  -p enable_color:=true \
  -p enable_depth:=false \
  -p enable_infra1:=false \
  -p enable_infra2:=false
```

別端末で確認する。

```bash
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
ros2 topic hz /camera/image_raw
ros2 topic echo /camera/image_raw --once
```

合格条件:

- `/camera/image_raw` が流れる
- 周波数が 30Hz 前後で安定する
- 画像サイズが 1280x720 で受信できる

### 1-2. LiDAR単体

```bash
ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args \
  -p xfer_format:=0 \
  -p multi_topic:=0 \
  -p data_src:=0 \
  -p publish_freq:=10.0 \
  -p output_data_type:=0
```

別端末で確認する。

```bash
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
ros2 topic hz /livox/lidar
ros2 topic echo /livox/lidar --once
```

合格条件:

- `/livox/lidar` が流れる
- 周波数が 10Hz 前後で安定する
- `sensor_msgs/msg/PointCloud2` として受信できる

## 2. 背景取得の動作確認

人がいない状態で実行する。

```bash
ros2 launch lcfall_ros2 capture_background.launch.py
```

確認ポイント:

- ログに `Captured frame n/200` が増えていく
- 完了後に `Background model saved!` が出る
- 自動でノード群が終了する

保存物の確認:

```bash
ls -l /data/background/background_voxel_map.npz
```

合格条件:

- `/data/background/background_voxel_map.npz` が更新される
- 途中で `No points within ROI` が連発しない
- 保存後に `You can now start the fall detection system.` が出る

注意:

- 背景取得の既定ROIと本体のROIは合わせる必要があるため、launch既定値は `params.yaml` に合わせてある
- 実機環境でROIを変える場合は、背景取得と本体の両方で同じ値を使う

## 3. 転倒検知システムの動作確認

システム全体を起動する。

```bash
ros2 launch lcfall_ros2 lcfall.launch.py
```

別端末で主要トピックを確認する。

```bash
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
ros2 topic list
ros2 topic hz /preprocessed/frame
ros2 topic echo /fall_detection/result --once
```

確認ポイント:

- `sync_preprocess_node` が背景モデルを正常に読み込む
- `Synchronized frame received. Processing...` が出る
- `/preprocessed/frame` が publish される
- `inference_node` が 48 フレーム蓄積後に推論を開始する
- ログに `Inference result:` が出る

合格条件:

- `/preprocessed/frame` が継続的に流れる
- 48 フレーム蓄積後に `/fall_detection/result` が流れる
- 通常姿勢で `NORMAL`、転倒姿勢で `FALLING` が出る
- `prediction == 1` のとき `alert_node` の警告ログが出る

補足:

- 実装上、推論は 48 フレーム蓄積後に開始される
- `inference_stride` は 10 なので、初回推論後は約 10 フレームごとに再推論される

## 4. 可視化の動作確認

`lcfall.launch.py` では可視化が既定で有効。

画面の見方:

- 左: カメラ画像と骨格オーバーレイ
- 右: 点群の正面投影
- 上部オーバーレイ: `Waiting for 48 frames` または `FALLING` / 通常状態

合格条件:

- OpenCVウィンドウ `LCFall ROS2 Demo` が開く
- 左側にカメラ映像が表示される
- 人物が写ると骨格が重畳される
- 右側に点群が描画される
- 推論開始前は待機表示、推論後は判定状態が更新される

可視化なしで本体だけ確認したい場合:

```bash
ros2 launch lcfall_ros2 lcfall.launch.py enable_visualization:=false
```

## 5. 進め方のおすすめ順

1. カメラ単体確認
2. LiDAR単体確認
3. 背景取得
4. システム全体起動
5. 可視化確認
6. 通常姿勢と転倒姿勢の両方で推論結果を記録

## 6. つまずきやすい点

- 背景モデルが古いままだと前景抽出が不安定になる
- ROI不一致だと背景差分結果が崩れる
- GPUや依存ライブラリ不足時は skeleton 抽出や推論モデル読み込みで失敗する
- センサ時刻が大きくずれると同期が成立せず `/preprocessed/frame` が出ない
