# D455画像未受信の調査とPi確認手順

## 最新の実機結果

ユーザーの追加許可後、こちらからPiを操作して画像受信・実画像姿勢推定・Livox・通常ROS全系を確認した。[実測記録](pi_validation_2026-11-10.md)を参照。以下の未実行手順・原因未確定の記述は初期調査時点の履歴。

- D455のuvcvideo interfacesをコンテナ停止中に一時unbindし、同設定で実画像受信に成功。根本的な競合発生条件と手動解除不要の再起動再現性は未解明。恒久設定変更なし。
- D455 1280×720/rgb8を維持。実画像60秒で597回推論、数値契約エラー0、中央値31.978ms。正確な人物数・関節位置の視覚評価は未実施。
- 全系で発見したConsoleのrclpy severity例外を最小修正し、通常ROS最終出力を215件受信。全件ATSの元stampと一致、重複0。約7.216Hzであり持続10Hzは未達。
- 設計・QoS・同期/点群処理の仕様変更なし。既存ローカル変更は保持。依存関係は再ビルドせず、修正common.pyだけのDocker追加層でイメージにも反映した。

2026-11-10 / feature/edge-pipeline。設計正本v2.6、RGB 1280×720・30fps・RGB8は維持する。
今回はソース閲覧と手順作成のみ。以下のコマンドはPi向けであり、今回実行していない。

## 確認済みの範囲

ユーザー報告: 通常ROSイメージのビルド、自作ノード/Python依存、HailoRT 4.23.0、Hailo-8認識、HEF SHA256/ロード、HailoBackend初期化、単色画像による推論48.2msが成功。D455はUSB 3.2、wrapper 4.56.4 / SDK 2.56.4、RGBプロファイル開始成功。実画像による推論・継続性能の確認とは区別する。

現在は `lcfall-d455-test` でD455単独検証中との報告。画像topic/Publisherは存在するが `topic hz` で未受信。現在のコンテナの起動引数・デバイス割当・完全なログは未提供。以前の全系停止ログは今回の単独起動ログと混同しない。

## ソースで分かることと分からないこと

### 追加ログの所見（ユーザー提供、2026-11-10）

- 今回の `lcfall-d455-test` は `interface 3` のclaimで `RS2_USB_STATUS_BUSY`、続いて `acquire_power failed` / `Error starting device: failed to set power state` を記録。今回のログではRGB開始・Node Is Upに達していない。以前の起動成功報告とは別の試行として扱う。
- `topic hz` も今回は `does not appear to be published yet`。まずUSB interface取得失敗を解消する必要があり、この試行の未受信をQoSやDDSだけでは説明しない。
- usbfs割当は `/dev/bus/usb/002/002`、host network、private IPC、追加GID44。色センサのパラメータ登録も完了していないように見えるが、param dumpの途中状態だけからRGB設定誤りとは断定しない。
- 固定SDKの `handle-libusb.h` は `libusb_claim_interface` のBUSYを短時間再試行し、それでも失敗するとエラーを返す。競合するユーザープロセス、kernel driverの保持など、占有者はまだ未特定。`failed to set power state` という文言だけで物理的な給電不足とは判断しない。
- 次のPi確認は `docker ps`、ホストの `sudo fuser -v /dev/bus/usb/002/002 /dev/video*`、`lsusb -t` と `/sys/bus/usb/devices/2-1:*/driver`。占有者が分かるまでkill/unbind/privileged化は行わない。今回もソース・設定変更や再ビルド、Pi操作は実施していない。

| 対象 | 確認した事実 | 判断 |
|---|---|---|
| Dockerfile（通常/EdgeRM） | librealsenseを `FORCE_RSUSB_BACKEND=ON`、CUDA無効、SDKサンプル/Python binding無効でソース構築 | RSUSBを意図した構成。`/dev/video*` の追加だけで解決するとは判断しない。既存イメージの実ビルド成果は別確認 |
| Compose | host network、D455の単一usbfsデバイスを同一パスへ渡し、USB数値GIDを追加。非root運用。IPC共有は明示なし | 通常のアクセスを意図した設定。再列挙でUSB番号が変わると古い割当が残る可能性。検証コンテナがこの設定を継承したかは未確認 |
| edge.launch.py | RGB8、1280×720×30、Depth/IR/IMU無効。画像QoSの上書きなし | リポジトリに今回の画像設定を変更すべき根拠はまだない。単独起動の実引数と比較が必要 |
| wrapperの起動完了 | `publishTopics()` はsetup後に `RealSense Node Is Up!` を表示 | フレームcallback実行や画像Publish成功の証明ではない |
| wrapperの画像Publish | `publishFrame()` は画像Publisherのsubscription countが非0の場合のみ画像を構築・Publish | グラフ上のPublisher数1だけでは配信を証明しない。購読側を起動したまま調査する |
| 画像QoS | 固定wrapperの初期値は `SYSTEM_DEFAULT`（intra-process等ではDEFAULT）。リポジトリでは変更していない | 報告のTRANSIENT_LOCALが何によるかは起動引数・パラメータ・DDSプロファイルの確認が必要 |
| hzのQoS | Jazzy上流はsensor_data（Best Effort / Volatile）で購読 | Reliable / Transient LocalのPublisherとは、この2ポリシーは互換。単純な不一致と断定して変更しない |
| USB警告 | SDKの `messenger-libusb.cpp` はlibusb制御転送の負の戻り値で警告を出す | 制御転送失敗の証拠。画像データ転送停止の証拠ではない。index 768だけでは対象操作を特定できない |
| 警告のnumber | 同ファイルはlibusb戻り値そのものではなく `errno` / `strerror(errno)` を表示 | `number: 11` 等だけからlibusbエラー種別・原因を決めない |
| DDS | Fast DDSは既定でUDPv4とSHMを利用。Composeはhost networkだがIPC共有は未指定 | 別コンテナ/ホストでのみ未受信ならSHM等を候補とする。同じコンテナ内の受信結果が先。Discovery成功はデータ配送成功を保証しない |

**原因は未確定。** 仮説に基づくQoS/USB/解像度/FPS/ファームウェア変更、privileged化、再ビルドは実施しない。

## Piで最初に行う確認（現設定のまま）

検証コンテナをそのまま使用し、別のD455ドライバ・全系コンテナは同時起動しない。
以下はPiホストから実行する。`docker exec` はentrypointを自動実行しないため、ROSコマンドには明示的に `/ros_entrypoint.sh` を使用する。

### 1. 実際の起動方法・デバイス・既存ログ

```bash
docker inspect lcfall-d455-test --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}} {{json .HostConfig.Devices}} network={{.HostConfig.NetworkMode}} ipc={{.HostConfig.IpcMode}} shm={{.HostConfig.ShmSize}} user={{.Config.User}} groups={{json .HostConfig.GroupAdd}}'
docker top lcfall-d455-test -eo pid,args
docker logs --tail 160 lcfall-d455-test
lsusb -t
ls -l /dev/bus/usb/*/*
docker exec lcfall-d455-test sh -c 'id; ls -l /dev/bus/usb/*/*; printenv | grep -E "^(ROS_|RMW_|FAST|CYCLONE|RCUTILS_)"'
```

シェルがPID 1でドライバを `docker exec` 起動している場合、そのドライバの出力は `docker logs` に含まれないことがある。元の起動ターミナルのログを確認する。ドライバ起動シェルだけでexportした変数も `docker exec ... printenv` には反映されないため、異なる場合は元の起動コマンドと環境も必要。

### 2. 同じコンテナ内のROS設定とグラフ

```bash
docker exec lcfall-d455-test /ros_entrypoint.sh ros2 node list --no-daemon
docker exec lcfall-d455-test /ros_entrypoint.sh ros2 param dump /camera/camera
docker exec lcfall-d455-test /ros_entrypoint.sh ros2 topic info /camera/camera/color/image_raw --verbose --no-daemon
```

ノード名が異なる場合は `node list` の結果を使う。画像profile/format、enable_color/depth、enable_sync、color_qos等を確認する。画像以外のtopicのQoSと取り違えない。

### 3. 同じコンテナ内で25秒、画像とCameraInfoを実購読

画像本体を表示・保存せず、到着数・受信FPS・stamp進行・サイズ等だけを表示する。Publisherの設定は変えない。Best Effort / Volatileを明示し、後発購読で過去の保持サンプルを要求しない。

```bash
docker exec -i lcfall-d455-test /ros_entrypoint.sh python3 - <<'PY'
import json
import time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo

rclpy.init()
node = rclpy.create_node('d455_receive_probe')
stats = {}
subs = {}

def receive(key, msg):
    now = time.monotonic()
    stamp = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
    s = stats.setdefault(key, dict(count=0, first=now, last=now,
                                  first_stamp=stamp, last_stamp=stamp,
                                  non_increasing_stamps=0, zero_stamps=0))
    if s['count'] and stamp <= s['last_stamp']:
        s['non_increasing_stamps'] += 1
    s['count'] += 1
    s['zero_stamps'] += int(stamp == 0)
    s['last'], s['last_stamp'] = now, stamp
    s['last_sample'] = dict(width=msg.width, height=msg.height,
                            frame_id=msg.header.frame_id, stamp_ns=stamp)
    if isinstance(msg, Image):
        s['last_sample'].update(encoding=msg.encoding, step=msg.step,
                               bytes=len(msg.data),
                               valid_data_length=len(msg.data) == msg.step * msg.height)

for key, cls, topic in (
    ('image', Image, '/camera/camera/color/image_raw'),
    ('camera_info', CameraInfo, '/camera/camera/color/camera_info'),
):
    subs[key] = node.create_subscription(
        cls, topic, lambda msg, key=key: receive(key, msg), qos_profile_sensor_data)

try:
    end = time.monotonic() + 25.0
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    result = {}
    for key, sub in subs.items():
        s = stats.get(key, {'count': 0})
        elapsed = s.get('last', 0) - s.get('first', 0)
        s['receive_hz'] = (s['count'] - 1) / elapsed if elapsed > 0 else None
        span = s.get('last_stamp', 0) - s.get('first_stamp', 0)
        s['stamp_span_hz'] = (s['count'] - 1) * 1e9 / span if span > 0 else None
        s['matched_publishers'] = sub.get_publisher_count()
        result[key] = s
    print(json.dumps(result, indent=2), flush=True)
finally:
    node.destroy_node()
    rclpy.shutdown()
PY
```

`receive_hz` は観測者が受信したレートで、ドライバ内部の取得FPSや全Publish数ではない。1280×720 RGB8は1枚約2.76MBのため、観測負荷やDDS転送損失も区別する。`stamp_span_hz` も欠落があれば低下する。

## 結果による次の分岐

- **画像が継続受信できる**: ドライバ取得・画像Publish・同コンテナ内DDS経路は成立。従来のhz実行場所/環境を比較する。別コンテナだけ失敗ならnetwork/IPC/SHMを調査し、両側をそろえたUDP限定等の一時比較を検討する。原因確認前にComposeへ `ipc: host` 等を追加しない。
- **画像0件、CameraInfoが継続受信できstampが進む**: wrapperの同じRGBフレーム処理からCameraInfoが出ているため、センサフレーム到着の証拠。画像側subscription認識・変換・Publish・大きなDDSデータ配送に絞る。単発のCameraInfoだけでは継続取得を証明しない。
- **両方0件、matched_publishersも0**: Domain、RMW、ノード名、実トピック、Discovery、QoSの他ポリシーを先に確認する。
- **両方0件でもPublisherがmatch**: センサ未到着かPublish前処理かDDS配送かはまだ未確定。既存debugログがあれば先に確認。なければ同じ起動引数を維持した短時間のDEBUG起動を次に提示する。固定wrapperの `Single video frame arrived` / `Frameset contain (... Color ...)`、`publishFrame(...)`、`Color stream published` の有無で分ける。最後のログはRMWへのPublish呼出しの証拠で、購読側到達の保証ではない。
- **RGB callbackが来ない**: USB権限/再列挙/他プロセス競合/カーネルUSBエラーを時刻対応で調べる。必要になった時点で `dmesg` の該当範囲や既存SDKツールでROSを介さないRGB取得を依頼する。現DockerfileではSDKサンプル/Python bindingを無効にしているため、`rs-*` や `pyrealsense2` が存在する前提の手順は出さない。

## 画像受信解決後の順序

1. 上の観測でRGB 1280×720/rgb8、step/data長、元stamp非0・進行、受信FPSを確認。長時間・欠落・遅延の測定は短時間の成立確認と区別する。
2. 既存HailoBackendとpersons_from_rowsで実画像の前処理→推論→デコード→元画素座標復元を個別確認。正常0人も有効だが、COCO17/BBoxの実データ確認には人物が写った条件が必要。元画像との整合・人物数・scoreを確認し、低scoreで欠損化しない。
3. 4要素のBBox、検出score、17要素ずつの関節XY/score、NaN三値のみの欠損表現、部分NaN/Inf拒否を確認。数値契約を満たすことと推定精度は別評価。
4. ウォームアップと定常を分離し、前処理＋Hailo＋デコード＋元座標復元の時間と継続性を計測。単色画像48.2msから実画像のNMS時間・10Hzを保証しない。
5. LiDARなしではATSの `/sync/image` は出ない。Poseノードへ生RGBをremapして30fps全処理する方法は採らず、実画像による個別診断として行う。本番トピック・同期構成は変更しない。具体的なコマンドは画像受信成立後に提示する。
6. その後にLivox MID-360の個別検証へ進み、両センサ確認後にATSと全系を確認する。

## 変更・再ビルド

今回の追加はこの調査文書と進捗の追記のみ。Dockerfile/Compose/Launch/ノード/設計書/モデルを変更しない。既存3ファイルのローカル変更は保持。現段階の再ビルドは不要。
後にパラメータだけで原因を直せる場合は起動設定変更で足りる。USB番号更新はコンテナ再作成が必要だがイメージ再ビルドは不要。ドライバ/SDKバイナリ自体の修正が必要と判明した場合だけ該当層を再ビルドする。

## 一次資料

- [固定wrapperのframe_callback / publishFrame / 起動表示](https://github.com/realsenseai/realsense-ros/blob/bafc21080c5c8e259dadbb309797949aee0dd950/realsense2_camera/src/base_realsense_node.cpp)
- [固定wrapperのprofile/QoS登録](https://github.com/realsenseai/realsense-ros/blob/bafc21080c5c8e259dadbb309797949aee0dd950/realsense2_camera/src/profile_manager.cpp)、[QoS初期値](https://github.com/realsenseai/realsense-ros/blob/bafc21080c5c8e259dadbb309797949aee0dd950/realsense2_camera/include/constants.h)
- [固定SDKのlibusb制御転送警告](https://github.com/realsenseai/librealsense/blob/5af97a58423ec11822e8d951d37d9c6945a09a24/src/libusb/messenger-libusb.cpp)
- [Jazzy topic hz実装](https://github.com/ros2/ros2cli/blob/jazzy/ros2topic/ros2topic/verb/hz.py)、[Jazzy RMW QoS定義](https://github.com/ros2/rmw/blob/jazzy/rmw/include/rmw/qos_profiles.h)
- [Fast DDS 2.14 transport](https://fast-dds.docs.eprosima.com/en/v2.14.5/fastdds/transport/transport.html)
