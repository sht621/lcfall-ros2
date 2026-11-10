# Raspberry Pi 5 エッジ観測パイプライン

通常ROS 2 Jazzy版とEdgeRM対応Jazzy版で、同じRGB–LiDARパイプラインを使用します。対象はUbuntu 24.04 ARM64 / Raspberry Pi 5 / Hailo-8 / RealSense D455 RGB / Livox MID-360です。正本の[設計v2.6](docs/edge_pipeline_design_v2.6.md)は変更していません。

**今回行ったのはソース・設定の修正と閲覧による差分確認のみです。ビルド・テスト・lint・構文チェック・ROS/Docker実行・実機検証は未実施です。以下の操作は、ユーザーがPi段階を開始して明示的に許可した後にのみ行います。**

## 2種類のDocker環境

| サービス | Dockerfile / イメージ | ROS環境 | ブリッジ |
|---|---|---|---|
| `edge` | `Dockerfile` / `fall-detection-edge:jazzy-arm64` | 公式`ros:jazzy-ros-base` | `use_edgerm=false`、通常ROSサービス |
| `edge-edgerm` | `Dockerfile.edgerm` / `fall-detection-edge:edgerm-jazzy-arm64` | Ubuntu 24.04からEdgeRM改変版Jazzyをソースビルド | `use_edgerm=true`、公式Python Middleware |

Composeは[compose.yaml](compose.yaml)の1ファイルです。`edge-edgerm`には`edgerm` profileを付け、サービス無指定で両方が起動する構成にはしていません。明示的にサービスを指定するとprofile名を追加せず選択できます。

**2サービスは排他的に運用してください。同じHailo・D455・MID-360を同時に使用する構成ではありません。** 切替時は先に起動済みサービスを停止します。

両環境ともROSは`/opt/ros/jazzy`、ワークスペースは`/opt/edge_ws`、installは`/opt/edge_ws/install`です。1回の運用につき、新規の専用イメージ1つ・コンテナ1つでドライバと自作ノードを動かします。GPUサーバやEdgeRMサービス基盤自体はこのComposeには含みません。

## 共用する処理

```text
D455 RGB ─┐            ┌─ /sync/image ─ pose_estimation_node ─ /pose_frame ─┐
          ├─ sync_node ├─ /sync/pair_info ────────────────────────────────┤
MID-360 ──┘   ATS      └─ /sync/points ─ lidar_preprocess_node ─ /points_roi┤
                                                                        ▼
                                                             frame_publisher_node
                                                                        │
                                                             /synchronized_frame
                                                                        │
                                                      topic_to_service_bridge_node
                                                                        │
                                            SendSynchronizedFrame (ROS / EdgeRM)
```

`fall_detection_msgs`の4 msg＋1 srv、`fall_detection_edge`の4処理ノード、内部トピックは共通です。サーバ送信には生RGBを含めません。追跡・転倒判定・256点化・床除去・録画は対象外です。

- ATSは元stampでqueue 10 / slop 0.0167秒。内部5経路と最終トピックはReliable / Keep Last / Depth 10 / Volatileです。
- Poseは元RGB画素座標の全人物COCO17/BBox/score。正常0人は空配列、処理失敗は未出力。明示的欠損のみNaN三値です。
- LiDARは静的TFで`fall_detection_frame`へ変換し、前方ROIを適用。出力は可変点数のXYZ float32＋元点に対応するtag uint8のみ。ROI 0点は正常な空点群です。
- 集約は整数nsの完全一致、完成順Publish、各入力10件・単調時計1秒TTL。再ATSは行いません。
- ブリッジは非同期`call_async()`、最大2要求、最新1待機、期限0.5秒、自動再送なし。Futureの回収・解放、`accepted`判定は今回変更していません。

処理設定は[edge.yaml](src/fall_detection_edge/config/edge.yaml)、取付TFは[mount.yaml](src/fall_detection_edge/config/mount.yaml)です。取付の0.85m / Pitch +25°、ROI X=1–8 / Y=−2–2 / Z=0–2mは暫定初期値であり未校正です。設定は起動時に適用します。既存TFがある場合は`PUBLISH_MOUNT_TF=false`で重複発行を避けます。

## Pi段階の準備

[.env.example](.env.example)をもとに`.env`を作り、以下を実機に合わせて設定します。既存`.env`がある場合は変数名を移行してください。従来の`*_REF`や`WITH_EDGERM`、`EDGE_ROS_BASE_IMAGE`は使用しません。

| 設定 | 内容 |
|---|---|
| `LIBREALSENSE_COMMIT`, `REALSENSE_ROS_COMMIT`, `LIVOX_SDK2_COMMIT`, `LIVOX_DRIVER_COMMIT` | 完全な小文字40桁コミットSHA。[環境準備文書](docs/pi_environment_preparation.md)に根拠を記録したリリース候補を設定済み。Piでのビルド・実機互換は未確認 |
| `HAILORT_DEB`, `HAILORT_WHEEL` | `artifacts/`内の実ファイル名。HailoホストPCIeドライバと一致するARM64 runtime deb、CPython 3.12/Linux ARM64 wheel |
| `BUILD_JOBS` | 初期1。CMake/Makeの並列数。colconはパッケージを逐次ビルドし、Piのメモリ使用を抑える |
| `HAILO_DEVICE`, `REALSENSE_USB_DEVICE`, `HAILO_GID`, `USB_GID` | 実Hailoデバイス、D455のUSBパス、アクセスを許可する数値GID |
| `ROS_DOMAIN_ID` | PiとGPUサーバで合わせるDomain |
| `ENABLE_BRIDGE` | サービスを使う場合true。サーバがない場合や最終トピックのみ利用する場合false |
| `FRAME_SERVICE_NAME` | 通常ROS/EdgeRM共通のサービス名。受信サーバと合わせる |

Hailoカーネルドライバはホスト側に導入します。この作業ではドライバをインストールしていません。実HEFと対応manifestは[models/](models/README.md)へ置きます。[manifest例](src/fall_detection_edge/config/pose_model.example.yaml)は未設定値を含み、意図して起動を拒否します。入力形状・量子化・出力名・モデル本来の検出/NMS条件・欠損定義の確認が必要で、実HEFによって専用decoderの追加が必要です。

両DockerfileのRealSense取得先は正式な`realsenseai/librealsense` / `realsenseai/realsense-ros`です。選定済みの固定SHAは維持しています。RealSenseのCMake必須依存`rclcpp_action`とLivoxの宣言済み実行時依存`rosbag2`は、通常版では公式ROS deb、EdgeRM版では選択した改変ROSのソースから準備する構成です。録画処理を起動する変更ではありません。

D455はRGBのみ1280×720×30の初期設定です。RSUSB版librealsenseとrealsense-rosをソースから構築します。USB再接続でデバイス番号が変わった場合は`.env`を更新してください。LivoxはPointCloud2 / 10Hz / `livox_frame`を初期値とし、[Livox JSON](config/livox/MID360_config.json)の旧IP（ホスト192.168.1.50 / LiDAR .198）を実環境へ合わせます。JSON内の外部変換はidentityを維持し、TFとの二重変換を避けます。

## EdgeRMソースビルドの追加準備

参考はローカルの`robot-agent-edgerm-main.zip`（アーカイブコメントのcommit: `be22bb43865e3aea8185929a00ef8800f35bfd93`）内のrobot/server Dockerfile、補助関数、entrypoint、Compose、`.gitmodules`、README、Pythonノードです。古いローカル`robot-agent-edgerm`は今回の根拠にしていません。

`EDGERM_ROS2_REF`と`EDGERM_ROS2_COMMIT`に、`enjoydolylab/edgerm-ros2`の取得refと、そのrefが指す完全なコミットSHAを設定します。両方を同じ完全SHAにすることもできます。ZIP記載の`i047840/2026011600`は参考値であり、最新版や互換版として採用していません。refの指す先がlockと違う場合はビルドを止め、無断で追従しません。

EdgeRM版は次の順で構築する記述になっています。

1. Ubuntu 24.04 ARM64に基本ビルド依存とROS apt-sourceを設定。`ros-jazzy-*`のapt導入を禁止するpinを作成。
2. BuildKit Secretで認証し、EdgeRMソースの固定commitと`src`配下のsubmoduleを取得。
3. `rclpy`、通常DDS用`rmw_fastrtps_cpp`、`rmw_edgerm_cpp` / `rmw_edgerm_py`、Launch/CLI、アプリとドライバに必要な画像/TF/点群依存を`/opt/ros/jazzy`へソースビルド。
4. librealsense RSUSB / Livox SDK2、両ROSドライバ、HailoRT、自作2パッケージを順に構築。

`cv_bridge`、`image_transport`、`camera_info_manager`、`diagnostic_updater`、`pcl_conversions`等も改変版ROSに対して構築します。通常ROSのaptバイナリを混ぜる方式にはしていません。選択したEdgeRMソースに不足する場合だけ、次の公式ソースを固定commitで追加します。

| 不足する主なパッケージ | 追加ソース | `.env`のlock |
|---|---|---|
| image_transport / camera_info_manager | `ros-perception/image_common` | `IMAGE_COMMON_COMMIT` |
| cv_bridge | `ros-perception/vision_opencv` | `VISION_OPENCV_COMMIT` |
| diagnostic_updater | `ros/diagnostics` | `DIAGNOSTICS_COMMIT` |
| pcl_conversions | `ros-perception/perception_pcl` | `PERCEPTION_PCL_COMMIT` |
| pcl_msgs | `ros-perception/pcl_msgs` | `PCL_MSGS_COMMIT` |

`image_common`は`image_transport`と`camera_info_manager`の両方、および後者の依存`camera_calibration_parsers`を確認し、不足があれば固定commitから補完します。追加側の既存同名パッケージを先に`COLCON_IGNORE`で除外し、colconの依存集合に含まれる不足パッケージと推移依存の兄弟パッケージを保持します。不要な追加パッケージは除外します。補完後に一覧を更新し、必要なパッケージが揃わない場合は停止します。

`rclcpp_action`と`rosbag2`もEdgeRM側の必須ソースとして存在確認します。rosdep helperへは`--packages-up-to`で列挙した依存一式を渡し、rosbag2メタパッケージの実行時依存も対象にする記述です。非公開helperの引数仕様・依存解決と`COLCON_IGNORE`の扱いは未確認であり、構築可能と判定したものではありません。

これらの取得元はZIPの`.gitmodules`と公式の[RealSense依存](https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_camera/package.xml)、[Livox依存](https://github.com/Livox-SDK/livox_ros_driver2/blob/master/package_ROS2.xml)、[PCL変換依存](https://github.com/ros-perception/perception_pcl/blob/ros2/pcl_conversions/package.xml)をソース閲覧して整理しました。閲覧したブランチをそのままJazzy互換の固定版と扱っていません。実際に選択するSHAと、その時点のmanifestの照合が必要です。Nav2/TurtleBot3/音声/可視化アプリ/GPU推論サーバは導入・ビルド対象にしていません。

**未確認点:** EdgeRM非公開リポジトリ内の`src/rosdep-install.py`はZIPに含まれず、接続先からの読出しも404でした。その内部実装と、選択予定commitの実パッケージ一覧は未確認です。DockerfileはZIPと同じ呼出し形式でこのファイルを使用し、存在しない場合や必要なソースパッケージが不足する場合は理由を表示して停止します。apt pinで依存導入が止まった場合は不足ソースを確認する必要があり、pinを外して公式ROSへ置換する方針ではありません。APIが適合すること、全依存が充足することはまだ立証していません。

### GitHub認証（Piで許可後のみ）

非公開ソースと必要なsubmoduleを読み取れる権限を用意し、GitHub CLIなどで認証します。ホストのシェルでのみトークンを環境変数へ設定します。`.env`へ保存しません。

```bash
gh auth login
export GITHUB_ACCESS_TOKEN="$(gh auth token)"
```

Composeはこの変数を`github-access-token`というBuildKit SecretとしてEdgeRMビルドだけに渡します。通常ROS版にGitHub認証は不要です。DockerfileのARG/ENVやruntime設定にはトークンを入れません。認証中はxtraceを無効にし、tmpfs上のaskpassがSecretファイルを読みます。remote URLは常に認証情報なしのHTTPS URLです。認証ヘルパーをレイヤーへCOPYしません。

Secret自体の変更はDockerのビルドキャッシュ無効化キーではありません。取得commitを更新する場合はlockを明示して更新してください。認証確認を伴う再ビルドやキャッシュの扱いは、Pi検証の許可後に行います。

## 起動・切替（Piで許可後のみ）

通常ROS版:

```bash
docker compose up --build edge
```

通常ROSの受信サーバがなく、`/synchronized_frame`まで利用する場合:

```bash
ENABLE_BRIDGE=false docker compose up --build edge
```

EdgeRM版:

```bash
docker compose up --build edge-edgerm
```

2回目以降、ソースや設定に応じて再ビルドが不要なら`--build`を省略できます。切替時の例:

```bash
docker compose stop edge
docker compose up edge-edgerm
```

逆方向も先に`edge-edgerm`を停止します。両サービスを指定する起動や、両方を有効にする全体起動は行いません。

EdgeRM版では`.env`の`EDGERM_URL`、`EDGERM_RUNTIME_NAME`、`EDGERM_VERBOSE`を受信環境に合わせます。参考ZIPによればservice名だけではEdgeRMの配送を分離しないため、`SendSynchronizedFrame`専用のruntime名をサーバと共有します。Composeの値はLaunch引数から**ブリッジだけ**のROSパラメータとしてYAMLより優先して渡します。通常ROS版でEdgeRMモジュールのimportは不要です。EdgeRM失敗時の通常ROSへの自動fallbackはありません。

受信サーバは同じ`fall_detection_msgs`を使用します。サーバ実装とEdgeRMサービス基盤の配備はこの作業の対象外です。将来トピック通信へ移る場合はブリッジを無効化し、サーバが`/synchronized_frame`だけを購読します。画像トピックの購読は作りません。

## 共通補助スクリプトと実行設定

[build-tools.sh](build-tools.sh)は両DockerfileのRUNからsourceして使用します。

- `git-clone(repository, ref, commit)`は専用BuildKitソースキャッシュ内でだけ動作し、既存checkoutの更新、固定commitへの切替、submodule同期・取得を行います。reset/cleanは実パスが`/var/cache/edge-build/`配下の場合だけ許可します。通常ROS版ではSDK/ドライバ取得、EdgeRM版ではそれに加えて改変ROSと不足依存の取得に使用します。取得結果はイメージ内`/opt/edge-source-locks/`へ記録します。
- `colcon-build(base_path, install_base, targets...)`はテスト無効化、共通CMake設定、逐次パッケージビルドと`BUILD_JOBS`管理をまとめます。`--packages-up-to`でソース依存も対象にし、通常ROS版のドライバ/アプリ、EdgeRM版のROS underlay・ドライバ・アプリで使用します。
- `install-hailort()`は将来のイメージビルド時に、実debのARM64/版、wheelのCPython 3.12対応タグ・版・依存metadataを照合します。既知のPython依存はaptで用意し、pipは`--no-deps --no-index`で指定wheelだけを導入するため、依存解決でaptのNumPyを置換しません。実配布物がないため、この処理とnative ABI適合は未検証です。metadata不整合時は停止し、未確認のHailoRT/NumPy版へ自動変更しません。

[ros_entrypoint.sh](ros_entrypoint.sh)は両イメージで共通です。必須のROS setupとworkspace setupの存在を確認し、順に読み込んで`exec "$@"`します。欠けている場合は理由を示して起動を失敗させます。旧`docker/entrypoint.sh`と二重の生成処理は残していません。

ComposeはARM64、host network、Hailo/D455実デバイス、数値GID、モデル・Livox・パラメータのread-onlyマウント、Domain、tmpfs、一時ログ、停止猶予をYAMLアンカーで共用します。`privileged`やGPU runtimeは使用しません。ROS/Launch既定ログは`/tmp` tmpfsです。両サービスのDockerログは`local`ドライバに保存し、`max-size`・`max-file`を指定せず標準のローテーション設定を使用します。

Pi段階で起動後、使用しているサービスのログを追跡できます（今回未実行）。

```bash
docker compose logs -f edge
docker compose logs -f edge-edgerm
```

## 未解決事項とPiでの確認

- EdgeRM非公開ソースの実レイアウトとrosdep helper API、改変版`rclpy`/MiddlewareのARM64適合、各依存の固定SHA、ROS apt置換禁止のもとでのビルド依存充足。
- Ubuntu 24.04 / Python 3.12 / HailoRT deb・wheel・ホストPCIeドライバの整合。aptのNumPyを保持した状態での実wheelとcv_bridge/OpenCVのnative ABI適合。metadata一致だけでは動作を保証しません。
- librealsense RSUSB/realsense-ros/Livoxの固定版、USB認識・非root権限、実topic/QoS/tag型/frame_id、元stampの時計整合。
- TFの取付実測、床基準とROI、正常0人/空点群の伝送、起動順と全系停止処理、ブリッジ設定の実反映。
- 通常ROSサービスとEdgeRM実通信、最大2要求・最新1待機・0.5秒期限、Future回収/解放・late response、再送なし。
- 全系持続10Hz、CPU・メモリ・温度・同期率・遅延・欠落。Hailo単体のFPSから達成を判断しません。

コミットとsubmoduleは固定しますが、apt/pipや上流CMakeが取得する推移依存まで固定したhermetic buildではありません。ビルド可能・ハードウェア互換・動作済みとは主張しません。過去の[Pi引継ぎ](docs/pi_validation_handoff.md)と[設計対照](docs/design_to_code_review.md)の旧バインディング/単一Docker切替の記述は履歴です。今回の配備方法はこのREADMEと現在のDocker/Composeを参照してください。[実装進捗](docs/implementation_progress.md)にも今回の差分と未検証事項を記録しています。
