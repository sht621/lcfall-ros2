# 通常ROS版のPi起動確認（2026-11-10）

## 最新: 10Hz未達のDDS原因調査

以下の7Hz前後の結果の後、Fast DDS共有メモリ領域不足を比較実験で切り分けた。16MiB/参加者とコンテナshm_size256MiBへ変更後、一時計測コードなしの通常ROS全系で約60秒・最終600件/9.999Hz、600/600件の元stamp一致、重複0を観測。[詳細・限界・ソース反映](pipeline_10hz_investigation.md)を参照。これ以前の未達結果は比較履歴として保持する。

## 追試: RGBを640×480へ変更（ユーザー指示、21:27 JST以降）

- `src/fall_detection_edge/config/edge.yaml` のimage_width/image_heightを640/480へ変更。Launchが同じ値からカメラprofileを作るため、D455とPoseの入力検査が一致する。30fps・RGB8、HEFの640×640入力、letterboxと元画素座標への復元方式は維持。設計正本・msg/srv・同期・ROI・QoSは未変更。サーバ側の元画像設定も今後640×480へ合わせる必要がある。
- 通常ROSコンテナを停止、D455の既知のuvcvideo競合を対象interfaceのunbindで回避し、`ENABLE_BRIDGE=false docker compose up -d --no-build edge` で再起動。設定ディレクトリはbind mountのため今回イメージのビルドは不要。前回のcommon.py修正層を含むイメージで再作成した。
- 起動ログで実profileがColor/RGB8/640×480/30fpsであることを確認。Poseパラメータ640/480、カメラprofile `640x480x30`、format RGB8を照会して確認。最初のwidth照会は一時的にNode not foundとなり、再照会で640を取得。
- 画像＋PairInfo＋最終出力をReliable購読し、62.021秒（先頭2秒除外）の観測で画像1798枚、29.943Hz。元stampの逆行・重複0。640×480/rgb8/step1920/921600 bytes、frame_id camera_color_optical_frame。
- 同じ約60秒で最終349件、5.823Hz。349/349件がPairInfoの元stampと完全一致、重複0、全件のstamp差が16.7ms以内。COCO17/BBox/scoreの数値形式エラー0。ROI出力はfall_detection_frameのXYZ float32＋tag uint8。正常出力例2595点。
- 人物候補数は4–10件で変動。これはモデルが返した候補数であり、実人物数・精度の証明ではない。しきい値・NMS・低score保持の規則は変更していない。
- 診断結果: `/tmp/lcfall-d455-investigation/640x480-result.txt`。画像追加購読のない前回7.216Hzとは観測負荷条件が異なるため、直接の性能改善/悪化判定には使わない。持続10Hz達成とは判断しない。
- 続けて前回と同じPairInfo＋最終出力だけのReliable購読で再測定。32.200秒、先頭2秒除外で205件 / **7.005Hz**、205/205件の元stamp一致、重複0、全件slop内。前回1280×720の7.216Hzに対して明確な改善は確認できず、解像度変更だけで持続10Hzは達成していない。厳密な性能比較試験ではない。
- 最後まで通常ROSコンテナは640×480設定・ブリッジ無効で稼働。ログのWARN/INFO切替も例外なく継続。`640x480-narrow-result.txt` と起動ログも同一の一時ディレクトリへ保存。


## 追試: D455・Hailo実画像・Livox・通常ROS全系（21時台 JST）

ユーザーから権限が足りればこちらで完結する指示を受け、Pi `TB-40` でsudo/Docker権限を確認して実行。以下は今回の測定結果。後掲のHailo認識失敗は過去の試行として保持する。

### D455 USB競合と復旧

- `lcfall-d455-test` はinterface 3の `RS2_USB_STATUS_BUSY` で起動失敗。別ユーザープロセスによるusbfs/videoデバイス占有は `sudo fuser` で検出されなかった。ホストのuvcvideoがD455のvideo interfacesへbindしていた。
- 検証コンテナを停止し、D455の実USBパス `2-1` のうちuvcvideoに結び付いたinterfacesだけをsysfs unbindして、同じ設定・同じコンテナを起動。RGB開始と実画像受信に成功。以後の全系起動でも同じ事前処理を行った。グローバルなモジュール削除・blacklist・privileged化・恒久udev変更はしていない。
- これはuvcvideoとのinterface取得競合を回避した実機証拠。libusb自動detachが失敗する詳細条件と、手動解除なしの再起動再現性は未解明。通常運用の自動復旧が完成したとは扱わない。
- 20秒観測: CameraInfo 599件 / 29.996Hz、画像548件 / 27.434Hz（Best Effortの診断購読）。画像1280×720/rgb8/2,764,800 bytes。`control_transfer index:768` 警告は開始時に残ったが、その後も実画像を受信できたため、警告だけで画像停止とは判断できない。

### 実RGB画像によるHailo姿勢推定

- カメラ単独コンテナ内で既存HailoBackend、CvBridge、persons_from_rowsを利用。最新画像1枚だけを保持し、0.1秒周期で診断推論。ROSの本番経路やATSは変更しない。
- 60.044秒、画像受信1711件、推論597回（約9.94回/秒）、全597回で2件の人物検出。推論/変換/数値契約のエラー0、受信stamp逆行・重複0。
- 元画像1280×720/rgb8、step3840、frame_id=`camera_color_optical_frame`。BBox4値、COCO17のXY/score各17値、有限値または欠損NaN三値の契約を全検出について確認。実人物数や関節位置の視覚的な精度評価は未実施。
- CvBridge変換＋前処理＋Hailo推論＋デコード＋元座標復元の時間: 最小30.765ms、中央値31.978ms、p95 36.083ms、p99 40.370ms、最大49.484ms。初回43.022ms。DDS待機・センサ取得からの遅延は含まない。単体測定を全系10Hzの証拠としない。
- 最初の60秒診断は集計JSONのnumpy.float32変換漏れで最終表示に失敗した。出力処理を修正して再計測し、上記は成功した再計測の値。

### Livox個別確認

- `192.168.1.198` へeth0からping 2/2成功。driverをPointCloud2/10Hz/livox_frameで起動し、実点群91件を受信。起動途中からの短区間で受信11.69Hzとなったため、定常10Hzの証拠とはしない。
- XYZ float32、tag uint8、frame_id livox_frame、約2万点/フレーム。read_xyz_tagによる型・データ形式確認エラー0、stamp逆行なし。tagの観測範囲0–192。後述の全系観測では生点群の受信約9.66Hz。
- 最初のCLIは空文字列パラメータの指定形式不備で起動前に失敗。デフォルト空文字列の指定を省いて起動した。ノード実装の変更はない。

### 全系のログ出力不具合と最小修正

- `ENABLE_BRIDGE=false docker compose up -d --no-build edge` で実センサ＋静的TF＋4処理ノードを起動。同期・Pose・ROI・最終Publishまで到達したが、集約警告時に `Logger severity cannot be changed between calls.` で終了。
- common.Console.showの1つの呼出し位置からINFO/WARNを切替える式を、異なるソース行のif/else呼出しへ変更。処理ロジック、QoS、msg/srv、ATS/ROI/TF/bridgeパラメータは変更していない。
- 修正common.pyを既存コンテナへコピー。実rclpyでINFO→WARN→INFO→WARNを呼び、例外なしを確認。全系を再起動した。
- 再作成でも修正を維持するため、元イメージを `fall-detection-edge:jazzy-arm64-before-console-fix`（e87b1e93cf88）として保存し、FROM元イメージ＋common.py COPYのみの小さなDocker層を作成。apt/pip/colcon/SDK/ROSの再ビルドなし。現在の `fall-detection-edge:jazzy-arm64` は `ecbb90085b207408cea08d30a2ded21d07e77859d79d03951cad5f3ee23c1866`。
- 実行中コンテナは元イメージからのコンテナに同じファイルをコピーしたもの。ソースと実行中ファイルのSHA256は `8ce76fd802f5d51df466205d2a4ab08e3533f00fd39d0ad852b019623e97fd8c` で一致。通常Dockerfileから今後ビルドする場合もソース修正が取り込まれる。

### 全系の受信結果と限界

- 8トピック同時購読の30秒観測では最終107件 / 3.547Hz。観測者自体が画像・点群を追加購読しており、Best Effortで観測取りこぼしもある（最終107件に対し観測PairInfoは105件）。
- 負荷を減らしPairInfoと最終出力のみReliable/Volatile/Depth10で購読。32.057秒中、先頭2秒を除いた約30秒で最終215件 / **7.216Hz**。215/215件が観測したPairInfoの元stampペアと完全一致、重複0、全件のRGB–LiDAR差が16.7ms以内。
- 最終サンプル: 2人物、ROI2611点、frame_id=`fall_detection_frame`、XYZ float32＋tag uint8の4フィールド、stamp差16.038ms。
- 結論: 実RGB＋実LiDAR→ATS→実姿勢推定/TF・ROI→厳密集約→`/synchronized_frame` の通常ROS経路を受信確認。**持続10Hzは未達**。FPS低下の内訳・同期率・長時間負荷・精度・移動中性能は未評価。観測負荷で出力レートが変わるため、7.216Hzを無観測時の最大性能とは扱わない。
- 起動直後のTF未到着によるROI失敗と未完成データの期限破棄も観測。その後は正常出力が継続。Livox終了時exit -11とHailoログ書込警告は別途未解決。
- 終了時: 通常ROSの `lcfall-ros2-edge-1` は稼働中、サービスブリッジ無効。旧 `lcfall-d455-test` は停止して保持。既存Compose/.envの値は変更していない。EdgeRM/通常ROSサービス受信サーバは未検証。
- 診断ログ・測定結果の一時保存先 `/tmp/lcfall-d455-investigation/`。再起動で失われ得るため主要数値は本書へ転記。

---

ユーザーがDockerビルド完了を報告し、実コンテナで通常ROSパイプラインの確認を明示許可した後の実測記録。過去の実装段階の未実行記録とは区別する。

## 条件と実施内容

- Piホスト: Debian 13.4、Hailo PCIe driver 4.23.0。
- 使用イメージ: `fall-detection-edge:jazzy-arm64`、ID `e87b1e93cf88`。今回ビルドは実施していない。
- `ENABLE_BRIDGE=false docker compose up -d --no-build edge` で起動。受信サーバ／サービス通信は今回の対象外。
- eth0: `192.168.1.5/24`、Livox JSONのLiDAR: `192.168.1.198`。直前にユーザーのping 3/3成功を確認。
- 自作4処理ノード、RealSense、Livox、静的TFの各プロセスが起動した。

## 観測結果

- RealSense D455をUSB 3.2として認識。RGB8、1280×720、30fpsのプロファイル開始と `RealSense Node Is Up!` をログで確認。画像トピックの受信・レートは未測定。
- Livox driver 1.2.6が初期化成功を報告し、PointCloud2出力を使用する旨を表示。実点群トピックの受信・フィールド・レートは未測定。
- 静的TF: `fall_detection_frame` → `livox_frame`、高さ0.85m、Pitch +25°の発行開始。実測校正の確認ではない。
- Pose初期化が `/dev/hailo0` のopenでerrno 6、`HAILO_DRIVER_OPERATION_FAILED(36)` により失敗。カーネルは `Device disconnected while opening device` を記録。
- ホストの `hailortcli fw-control identify` と、同じイメージの単発CLIでも同じ失敗を再現。コンテナ固有の問題とは判断できない。
- 必須プロセス終了を受け、Launchが全系を停止。Livoxは終了処理中にexit -11を記録したため、停止処理の問題も未解決。
- トピック観測用プロセスを実行しようとした時点でコンテナは停止済み。同期ペア、PoseFrame、ROI点群、`/synchronized_frame` の受信件数は取得できていない。

## 復旧試行と残る事項

- 使用中プロセスがない状態で `hailo_pci` を再読込。再probeは `Failed reading device BARs, device may be disconnected` で失敗し、`/dev/hailo0` は生成されなかった。
- Hailoデバイス `0001:01:00.0` のsysfs resetを試行したが、カーネルが `not ready 65535ms after FLR; giving up` を記録。続く再probeもBAR読出し失敗。最終scanは `Hailo devices not found`、復旧しなかった。
- `vcgencmd get_throttled` は `0x0`。電源・物理接続・PCIe不調の原因を特定したものではない。
- Piの再起動、電源断、配線変更、ドライバ更新、起動設定変更は実施していない。
- 追加警告: 非root Poseプロセスのカレントディレクトリに `hailort.log` を作成できない。初期化失敗の直接原因とは区別する。

**結果: 通常ROSパイプラインの成立は未確認。** Hailoのホスト側アクセス復旧後、全系を再起動し、実センサの両入力・同期・実推論・ROI・最終メッセージと元stamp対応を測定する必要がある。持続10Hz、推論精度、空結果、サービス通信は未検証。

起動時ログとカーネルログの一時保存先: `/tmp/edge-validation-20261110/`（再起動で失われる可能性あり）。
