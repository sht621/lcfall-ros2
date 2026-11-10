# Raspberry Pi 5 実機検証前の環境・依存準備

調査日: 2026-11-10（JST）。対象: `feature/edge-pipeline`。

**調査と、その結果に基づくDockerfile・共通補助スクリプトの静的修正のみ。Docker/colconビルド、テスト、lint、構文チェック、ROS実行、ハードウェア照会、インストール、コミット、Pushは行っていない。** 設計v2.6、処理アルゴリズム、Composeの2サービス構成は変更していない。

「確認済み」は公開ソースやメタデータで確認できた内容。「候補」は次のPi段階で評価する入力。「未確認」は配布物・非公開ソース・実機が必要な事項である。SHAが確定していることは、その組合せのビルド成功やハードウェア互換を意味しない。

## 1. 調査開始時の状態と必要ソフトウェア

ローカルは作業ツリーがクリーンだった。HEADは`442477075621f0c5dc780a830b01e9f1947e318c`。GitHubの[`feature/edge-pipeline`先頭](https://api.github.com/repos/sht621/lcfall-ros2/commits?sha=feature%2Fedge-pipeline&per_page=1)も同じSHA（commit時刻2026-11-10 04:59:56 UTC）だった。pull/fetch/resetは行わず、既存コードを保持した。

| 要素 | 準備条件・候補 | 状態 |
|---|---|---|
| PiホストOS | Raspberry Pi 5向けUbuntu 24.04 LTS、64-bit ARM64 | 設計条件。実機OS/kernel/page sizeは未確認 |
| コンテナ | 通常版`ros:jazzy-ros-base`、EdgeRM版`ubuntu:24.04`から構築 | 現行Dockerfile。ベースdigest、aptパッケージ版は未固定 |
| ROS | Jazzy、インストール先`/opt/ros/jazzy` | Nobleのamd64/arm64向けdebがあることは[公式インストールソース](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Installation/Ubuntu-Install-Debs.rst)で確認 |
| Python | コンテナのCPython 3.12系、ROSとHailo wheelで共用 | 実イメージのpatch版とNumPy ABIは未確認。Pythonを3.11へ差し替える方針ではない |
| SDK/ROSドライバ | 下表の4つのリリースcommit | 公開ソースに基づく候補。実機未検証 |
| HailoRT/PCIe driver | Hailo-8向け4.24.0を整合させる候補 | ソースtag確認済み。ホスト既存版・deb/wheel未提供のため採用版は未確定 |
| Docker/Compose | ARM64、BuildKit、Dockerfile heredoc/cache/secret mounts、Compose v2のbuild secrets（environment）を扱える環境 | Piの導入版は未確認。現在Docker操作なし |
| メモリ/並列数 | `BUILD_JOBS=1`から開始、colconは逐次 | Pi RAM/空き容量/冷却/必要swapは未確認 |

### 最新状態の観測と候補選定は分ける

GitHub APIで観測した各default branchの最新commitは以下。これは採用lockではない。RealSenseは既知の非betaリリースの対、LivoxはJazzy対応が明記されたリリースを候補にした。

| repository | 観測したdefault branch先頭SHA | commit日（UTC） |
|---|---|---|
| [realsenseai/librealsense](https://api.github.com/repos/realsenseai/librealsense/commits?per_page=1) | `c84fe6438bd14a4cc69f46deae92c993268232c6` | 2026-10-11 |
| [realsenseai/realsense-ros](https://api.github.com/repos/realsenseai/realsense-ros/commits?per_page=1) | `7a87097c499e3b0b69518d54a8e7a9ac1924e87c` | 2026-10-11 |
| [Livox-SDK/Livox-SDK2](https://api.github.com/repos/Livox-SDK/Livox-SDK2/commits?per_page=1) | `c0796f04c143143899c87a773d9f6b7136453c0b` | 2026-09-21 |
| [Livox-SDK/livox_ros_driver2](https://api.github.com/repos/Livox-SDK/livox_ros_driver2/commits?per_page=1) | `21445540f0d100dc86a7e6df312dd70bbdb4afdf` | 2026-09-21 |

## 2. ROSドライバ・SDKの固定commit候補

以下はGitHub `git/ref/tags/...`で解決した**commit object**の40文字SHA。annotated tag objectを誤ってlockへ入れたものではない。`.env.example`へはこの4件だけを「ソース確認済み・実機未検証の候補」として反映した。

| 変数 | 正確な取得先 | tag | 固定候補SHA |
|---|---|---|---|
| `LIBREALSENSE_COMMIT` | `https://github.com/realsenseai/librealsense.git` | `v2.56.4` | `5af97a58423ec11822e8d951d37d9c6945a09a24` |
| `REALSENSE_ROS_COMMIT` | `https://github.com/realsenseai/realsense-ros.git` | `4.56.4` | `bafc21080c5c8e259dadbb309797949aee0dd950` |
| `LIVOX_SDK2_COMMIT` | `https://github.com/Livox-SDK/Livox-SDK2.git` | `v1.3.1` | `f5d9375f84efe2b15bc0a052d3e18482ed13adf4` |
| `LIVOX_DRIVER_COMMIT` | `https://github.com/Livox-SDK/livox_ros_driver2.git` | `1.2.6` | `13eb05e4e6dd7a765b934d0c5fd6236676a57b49` |

タグ解決根拠: [SDK tag](https://api.github.com/repos/realsenseai/librealsense/git/ref/tags/v2.56.4)、[ROS wrapper tag](https://api.github.com/repos/realsenseai/realsense-ros/git/ref/tags/4.56.4)、[Livox SDK tag](https://api.github.com/repos/Livox-SDK/Livox-SDK2/git/ref/tags/v1.3.1)、[Livox driver tag](https://api.github.com/repos/Livox-SDK/livox_ros_driver2/git/ref/tags/1.2.6)。

### librealsense v2.56.4

- [公式リリース](https://github.com/realsenseai/librealsense/releases/tag/v2.56.4)はUbuntu 24.04を対象に挙げ、D400系のFW条件を掲載し、D455を含む修正を記録している。FW表ではD400は5.17.0.9以降。実D455の版を確認してから判断し、自動更新はしない。
- [CMake option定義](https://github.com/realsenseai/librealsense/blob/5af97a58423ec11822e8d951d37d9c6945a09a24/CMake/lrs_options.cmake)に`FORCE_RSUSB_BACKEND`が存在。現在のRSUSB、CUDA無効、Python binding無効の構築方針と対応する。Jazzyそのものの対応は次のROS wrapperとの組合せで判断する。
- 基本依存はCMake、C/C++ compiler、pkg-config、libusb-1.0、libudev、OpenSSL。現行DockerはこれらとGL/GLFWを明示導入する。[Ubuntu向けソース構築説明](https://github.com/realsenseai/librealsense/blob/5af97a58423ec11822e8d951d37d9c6945a09a24/doc/installation.md)も参照。
- Pi 5 ARM64 + Ubuntu kernel + RSUSB + D455 RGBのみの連続取得は未検証。古いRaspbian/ARMv7手順をPi 5 Noble向け検証済み手順と同一視しない。ホストUVCパッチを今回導入する指示ではない。

### realsense-ros 4.56.4

- [公式リリース](https://github.com/realsenseai/realsense-ros/releases/tag/4.56.4)にUbuntu 24.04/Jazzy対応の変更がある。[固定ソースCMake](https://github.com/realsenseai/realsense-ros/blob/bafc21080c5c8e259dadbb309797949aee0dd950/realsense2_camera/CMakeLists.txt)はJazzy分岐と`find_package(realsense2 2.56)`を持つため、SDK 2.56.4を同時候補とした。
- [manifest](https://github.com/realsenseai/realsense-ros/blob/bafc21080c5c8e259dadbb309797949aee0dd950/realsense2_camera/package.xml)とCMakeに基づく主要依存は、ament_cmake、OpenCV/cv_bridge、image_transport、rclcpp/components、camera_msgs、sensor/std/geometry/nav系interfaces、tf2/TF、diagnostic_updater、Eigen、ROS environment。CMakeには`rclcpp_action`の必須要求もある（後述）。
- `/opt/edge_ws/src`へ配置し、camera_msgsも依存としてビルドする現行`--packages-up-to`と整合する。自作LaunchがD455のRGBのみを選択するが、ドライバ自体は汎用実装としてビルドされる。
- Pi ARM64上の実動作、ROS parameter名・出力topic・時計、USB権限、改変版ROSに対するABI適合は未確認。

### Livox SDK2 v1.3.1 + Driver 1.2.6

- [Driver 1.2.6リリース](https://github.com/Livox-SDK/livox_ros_driver2/releases/tag/1.2.6)はUbuntu 24.04/Jazzy対応を明記する。SDKはROS非依存であり「SDK自身がJazzy認定」という意味ではない。[SDK v1.3.1](https://github.com/Livox-SDK/Livox-SDK2/releases/tag/v1.3.1)と同時期のMID-360系対応リリースを組み合わせる候補とした。Pi上の組合せ認定は未確認。
- SDKの[CMake](https://github.com/Livox-SDK/Livox-SDK2/blob/f5d9375f84efe2b15bc0a052d3e18482ed13adf4/CMakeLists.txt)はC++11/pthreadベース。[sdk_core](https://github.com/Livox-SDK/Livox-SDK2/blob/f5d9375f84efe2b15bc0a052d3e18482ed13adf4/sdk_core/CMakeLists.txt)は静的/共有ライブラリとheaderをinstallする。現行Dockerの先行SDK installとldconfig、driverが要求する`/usr/local/lib/liblivox_lidar_sdk_shared.so`・headerはソース上対応する。
- [Driver ROS2 manifest](https://github.com/Livox-SDK/livox_ros_driver2/blob/13eb05e4e6dd7a765b934d0c5fd6236676a57b49/package_ROS2.xml)の主依存はament_cmake_auto、rosidl生成/runtime、rclcpp/components、std/sensor/rcl/rcutils系、pcl_conversions、PCL、APR、git。[CMake](https://github.com/Livox-SDK/livox_ros_driver2/blob/13eb05e4e6dd7a765b934d0c5fd6236676a57b49/CMakeLists.txt)がPCL/APRとSDK共有ライブラリを参照する。現行Dockerのlibpcl-dev/libapr1-dev/rapidjson-devはこれらの準備に対応する。
- [build.sh](https://github.com/Livox-SDK/livox_ros_driver2/blob/13eb05e4e6dd7a765b934d0c5fd6236676a57b49/build.sh)のJazzy指定は`ROS_EDITION=ROS2` / `DISTRO_ROS=jazzy`。DockerがmanifestをROS2版へ置換して同じCMake値を指定することを確認した。
- [Driver README](https://github.com/Livox-SDK/livox_ros_driver2/blob/13eb05e4e6dd7a765b934d0c5fd6236676a57b49/README.md)上は`xfer_format=0`のXYZ float32 / tag uint8を含む形式が記載されている。これは実Publisherフィールドを確認した証拠ではない。
- SDKの[config parser](https://github.com/Livox-SDK/Livox-SDK2/blob/f5d9375f84efe2b15bc0a052d3e18482ed13adf4/sdk_core/parse_cfg_file.cpp)は`host_net_info`のobject旧形式とarray新形式を分岐する。保持中のJSONを新形式へ機械的に書き換える必要はソース上確認されなかった。実IP/ポートと通信・時刻同期は未確認。

## 3. 通常ROS版のビルド前提とDockerfile照合

通常版はEdgeRMの認証・モジュール・ソースビルドを要求しない。`.env`には候補4 SHA、実在するHailo配布物名、実デバイスパス/数値GID、Domainを準備する。`.env.example`をそのまま実機環境とみなさない。

- `git-clone()`は完全SHAをrefとしてfetchし、一致確認後にcheckoutする。SDK/driverは専用キャッシュ内、source dependencyは`--packages-up-to`で扱う。ソースSHA固定とapt/pip/ベースdigest/上流CMakeのネット取得まで固定した再現環境とは別である。
- SDKをROS driverより先にinstallし、ROSをsourceして自作msgとdriver依存を含めて構築する流れは、選定したソースの要求とおおむね対応する。
- Dockerは非root UID/GID 1000、HailoとD455の実デバイス/数値グループのみを渡す。ホストnetworkを使用。`ENABLE_BRIDGE=false`なら受信サーバがなくても最終トピックまでの検証対象を選べる。

### 調査結果に基づくDocker依存修正と残る確認点

| 指摘 | ソースから分かる事実 | 今回の修正・残る条件 |
|---|---|---|
| RealSense取得URLの移転 | [公式README](https://github.com/realsenseai/realsense-ros)も新URLへの更新を案内 | 両DockerfileのSDK/ROS取得先をrealsenseaiへ変更。第2節と.env.exampleの固定4 SHAは維持 |
| RealSenseのrclcpp_action | 4.56.4 CMakeが必須要求する一方、そのversionのcamera manifestには明示dependがない | 通常版へros-jazzy-rclcpp-action、EdgeRM underlayのtargetsへrclcpp_actionを明示。private source内に存在しなければ停止し、stock coreの追加で代替しない |
| Livoxのrosbag2 | 1.2.6 ROS2 manifestのexec_depend。ROS2 CMakeではrosbag2を直接find/linkしない | 通常版へros-jazzy-rosbag2、EdgeRM targetsへrosbag2を追加し宣言依存を満たす方針。録画の起動や処理ノードの変更はしない。private sourceの実依存は未確認 |
| SDK追加ビルド | librealsense 2.56.4ではBUILD_TOOLS/BUILD_GLSL_EXTENSIONSが既定ON、Livox SDK2 rootもsamplesを追加する。現行Dockerはこれら全部を明示OFFにしてはいない | 容量・依存削減の余地。ドライバが動かないと断定する不整合ではなく、最小構成化の後続課題 |
| Hailo pip依存 | wheel setupがNumPy等を要求する | 共通install-hailortで実配布物のmetadataを照合し、pipは--no-deps --no-index。NumPy等はapt版を保持。実wheelのnative ABIは未確認 |

上記の修正はソース上だけで実施した。**EdgeRMの依存閉包とHailo配布物が未確認のまま、両イメージを構築可能とは報告しない。**

`rclcpp_action`の公式取得元は[ros2/rclcppのJazzy manifest](https://github.com/ros2/rclcpp/blob/jazzy/rclcpp_action/package.xml)。action_msgs、rclcpp、rcl_action、rcl、rcpputils、rosidl_runtime_c等を要求する。EdgeRMでは選択したfork/submoduleにある対応ソースを使い、独立した上流rclcppのSHAは追加しない。

`rosbag2`は[ros2/rosbag2のJazzy manifest](https://github.com/ros2/rosbag2/blob/jazzy/rosbag2/package.xml)で、ros2bag、rosbag2_compression/cpp/py/storage/transport、compression_zstd、storage_default_pluginsをexec_dependに持つメタパッケージ。名前だけ存在しても依存充足とはしない。`--packages-up-to`のソース依存一式を列挙してrosdep helperへ渡し、同じroot targetsで構築する記述にした。これら公開Jazzyソースは依存関係の根拠であり、非公開forkの実構成や固定版の証明ではない。

## 4. EdgeRM版固有の前提

`https://github.com/enjoydolylab/edgerm-ros2.git`について、GitHub connectorのrepository/branch/source読出しはNot Found/404だった。非公開repoの存在・権限・refのどれが原因かは区別できない。従って次は**未確定のまま空欄を維持**する。

- `EDGERM_ROS2_REF` / `EDGERM_ROS2_COMMIT`。参考ZIPの`i047840/2026011600`を採用確定や最新版としない。
- `src/rosdep-install.py`の存在、引数の解釈、対象packageの列挙方法、COLCON_IGNOREの扱い。現行Dockerは参考ZIPの呼出し形式を記述しているが、private helper自身を読めていない。
- `rmw_edgerm_cpp` / `rmw_edgerm_py`の実package名・依存・ARM64対応と、改変rclpyの`create_client(..., middleware=...)`、Future/remove_pending_requestとの互換。
- 現行targets（rclpy、rclcpp_components/lifecycle/action、rosbag2、rmw_fastrtps_cpp、rmw_edgerm_cpp/py、ros2launch/run/service、ament_cmake_auto、rosidl_default_generators、sensor/std/geometry/nav msgs、tf2_ros/py/eigen/geometry_msgs、message_filters、cv_bridge、camera_info_manager、image_transport、diagnostic_updater、pcl_conversions）が指定commitで揃うか。

コード側は提示済みの公式Middleware APIを維持し、通常ROSへの自動fallbackは行わない。サービス種別ごとのruntime名、URL、サーバ受付型と同時要求管理は受信側と整合させる。アプリ固有C++ bindingは不要であり復活させない。

### 追加source lockが必要になる条件

| 変数 | fork内で存在確認するpackage | 不足時の取得先と扱い |
|---|---|---|
| IMAGE_COMMON_COMMIT | image_transport、camera_info_manager、camera_calibration_parsers | `https://github.com/ros-perception/image_common.git`。どれか不足した場合だけ固定commitから不足分を有効化 |
| VISION_OPENCV_COMMIT | cv_bridge | `https://github.com/ros-perception/vision_opencv.git`。不要なimage_geometry等は除外 |
| DIAGNOSTICS_COMMIT | diagnostic_updater | `https://github.com/ros/diagnostics.git`。不要なaggregator等は除外 |
| PERCEPTION_PCL_COMMIT | pcl_conversions | `https://github.com/ros-perception/perception_pcl.git`。不要なpcl_ros等は除外 |
| PCL_MSGS_COMMIT | pcl_msgs | `https://github.com/ros-perception/pcl_msgs.git` |

公開repoの存在と関連依存は過去のソース調査で確認しているが、**今回のforkに対して必要か、どの版がABI互換かは未確定**。全5変数を空欄に維持する。将来forkを閲覧できた段階で、そのcommitのpackage.xml/CMakeとJazzy releaseを照合して選ぶ。

補完処理は追加repo中の既存同名packageを先にCOLCON_IGNOREで除外し、全sourceの`colcon list --packages-up-to`から不足packageの依存集合を得る。その集合内の追加packageは推移依存の兄弟packageも保持し、集合外だけを除外する。従来の名前だけの許可リストによる依存の除外を避ける。根拠はcolconの[list verb](https://github.com/colcon/colcon-package-information/blob/master/colcon_package_information/verb/list.py)と[依存選択処理](https://github.com/colcon/colcon-package-selection/blob/master/colcon_package_selection/package_selection/dependencies.py)。既存fork側にはCOLCON_IGNOREを追加しない。

標準colconの除外と、非公開rosdep helperが同じ除外集合を使うことは別の確認事項である。helperへ渡す名前はcolconの依存集合に揃えたが、helperが直接package.xmlを全走査する方式なら重複や不要依存を再び見てしまう可能性がある。**helperの引数仕様・ソース閲覧と、fork側に元からあるCOLCON_IGNOREの確認をEdgeRMビルド前の条件**にする。公式apt ROSで埋めるためにpinを外す方針ではない。

認証は既存どおりBuildKit Secretを使う。権限あるGitHub認証をホストに用意し、トークンを.env/ARG/remote/レイヤーに保存しない。今回は認証情報の取得・変更やprivate checkoutを行っていない。

## 5. Hailo-8 runtime / HEFの準備

### HailoRTの候補と未確認点

[現行HailoRTのREADME](https://github.com/hailo-ai/hailort/blob/master/README.md)はmasterをHailo-10/15向けとし、Hailo-8/8Lにはhailo8 branchを指定する。単に最新の5.xを選ばない。

| 要素 | 確認できた候補 | 判断 |
|---|---|---|
| HailoRT | v4.24.0 / `63adffec12da88f1e7cfdc07f2e07681671e83e8` | [tag](https://api.github.com/repos/hailo-ai/hailort/git/ref/tags/v4.24.0)、Hailo-8 branch系。runtimeソースの候補 |
| PCIe driver | v4.24.0 / `5d9ce886bd1c91451e3e0a9c2e2e5ee07e5bcc63` | [tag](https://api.github.com/repos/hailo-ai/hailort-drivers/git/ref/tags/v4.24.0)。ホストのkernel headersと一致して構築/導入する版は後で判断 |
| Model Zoo | v2.19.0 / `3f19ede8673c0d050bca24ea02dd6606d52f816c` | [tag](https://api.github.com/repos/hailo-ai/hailo_model_zoo/git/ref/tags/v2.19.0)。[公式設定](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/config/config.yaml)は4.24.xとv2.19.0の対応を記載 |

**Pi 5なら必ず4.24.0という要件ではない。** 実ホストの既存driver/FW、入手できるUbuntu向けruntime/wheel、実HEFのcompiler世代を合わせて版を決める。runtime/deb/wheel/driverを同一のリリース系列で揃えることを初期条件とし、例外的な組合せはvendorの互換表なしに採用しない。

4.24.0の[Python setup](https://github.com/hailo-ai/hailort/blob/63adffec12da88f1e7cfdc07f2e07681671e83e8/hailort/libhailort/bindings/python/platform/setup.py)はlinux_aarch64を含む。Python拡張の[CMake](https://github.com/hailo-ai/hailort/blob/63adffec12da88f1e7cfdc07f2e07681671e83e8/hailort/libhailort/bindings/python/src/CMakeLists.txt)は通常経路でHailoRT 4.24.0 EXACTを要求する。既存backendのVStream系API名は同版の[Python bindingソース](https://github.com/hailo-ai/hailort/blob/63adffec12da88f1e7cfdc07f2e07681671e83e8/hailort/libhailort/bindings/python/platform/hailo_platform/pyhailort/pyhailort.py)に存在するが、実行適合は未検証。

ただし、これだけで**公式のUbuntu 24.04 ARM64 / CPython 3.12用wheelが配布されていると断定しない**。公開GitHub releaseの4.24.0 assets一覧は空だった。Developer Zone配布カタログの詳細は今回取得できず、実deb/wheelの名称・hash・Requires-Python・共有ライブラリ/NumPy依存は未確認。

Raspberry Pi向け[公式AI手順](https://www.raspberrypi.com/documentation/computers/ai.html)やHailo Appsのhailo-all推奨はRaspberry Pi OS向けの導入文脈であり、Nobleコンテナへ無条件に導入する根拠にしない。[CanonicalのUbuntu 24.04記事](https://ubuntu.com/blog/hackers-guide-to-the-raspberry-pi-ai-kit-on-ubuntu)も古いHailo-8Lの開発例で、今回のHailo-8・4.24.0・Python3.12の検証結果ではない。記事内のPython差替えをJazzy環境へ転用しない。

### DockerfileのHailoRT導入方針（今回記述のみ）

両Dockerfileはbuild-tools.shの`install-hailort`を呼ぶ。将来のビルドで、実debのPackage=hailort/Architecture=arm64、CPython 3.12、wheelの対応タグ・Name/Version/Requires-Python/Requires-Dist、debのupstream版とwheel版の一致を照合する。Debian epoch/revisionを除いた版を比較し、未対応の版表記や要件は無理に補正せず停止させる。今回この処理を実行したわけではない。

- NumPy・OpenCVはUbuntu aptから導入。Hailo v4.24.0の公開setup.pyが列挙するargcomplete/contextlib2/future/netaddr/netifacesもaptで準備し、metadata読出し用にpython3-packagingを追加する。4.24.0を配布物の採用版として固定する変更ではない。
- 実wheelの依存がapt導入済みのPython distributionと合わなければ停止し、追加依存や版を別途判断する。pipは`--no-deps --no-index`で指定wheelだけを導入し、依存解決でシステムNumPyを置換しない。
- metadata一致はnative ABIの証明ではない。実wheelのELF/libhailort/glibc/libstdc++/NumPyビルドABIと、aptのcv_bridge/OpenCV/NumPyの組合せは未確認。wheelがNumPy 2等を必要とする場合も、その場でNumPyを更新せず対応配布物を再検討する。
- debのDepends/maintainer script、Ubuntu 24.04適合、ホストPCIe driver/FWとの整合は実配布物とPiが必要。ホストdriverをコンテナ内へ導入しない。ヘルパーはHailo・NumPy・cv_bridgeのnative moduleをimportして実動作を判定するものではない。

### 入手・配置するファイル

現時点で`artifacts/`と`models/`にはREADMEしかない。以下の名前のうち、`<version>`は**必要形式の説明**であって、配布実在を確認したファイル名ではない。未提供の値は.envへ埋めない。

| 用途 | 入手先・必要形式 | 配置・未確認事項 |
|---|---|---|
| ユーザー空間runtime deb | [Hailo Developer Zone](https://hailo.ai/developer-zone/software-downloads/)のHailo-8 / Linux ARM64。例示形式`hailort_<version>_arm64.deb` | `artifacts/`へ実ファイル、HAILORT_DEBにそのbasename。Package/Version/Architecture=arm64、Depends、配布hashを確認 |
| Python wheel | 同じreleaseのPyHailoRT。必要ABI形式`hailort-<version>-cp312-cp312-linux_aarch64.whl`相当 | artifactsへ実wheel、HAILORT_WHEELにbasename。cp311/x86_64用は不可。実metadata/ELFとlibhailort版を確認 |
| ホストPCIe driver/FW | [公式driver](https://github.com/hailo-ai/hailort-drivers/tree/v4.24.0)とvendor手順。既存host版を先に確認 | コンテナにkernel driverを導入しない。Ubuntu Pi kernel headers/ページサイズ、FW、udev、再起動要否を後の許可段階で確認 |
| YOLOv8m-Pose HEF | 次項の公式Hailo-8モデル配布表、または提供済みモデルの正しい出所 | `models/yolov8m_pose.hef`。実バイナリ/hash/target/FW互換は未確認 |
| manifest | このリポジトリの`pose_model.example.yaml`と実HEF/元モデルの照合から作成 | `models/pose_model.yaml`。単なる例のコピーやverified_for_hefだけの変更は不可 |

wheelが入手できずソースビルドが必要なら、それは別途許可が必要な準備作業であり、今回は代替ビルドを行わない。GStreamer/TAPPASのPython `hailo`モジュールと、今回の`hailo_platform`を混同しない。TAPPAS導入はこの直接HailoRT経路の必須条件として追加しない。

### HEFとmanifestの照合

[Model Zoo Hailo-8 pose表](https://github.com/hailo-ai/hailo_model_zoo/blob/3f19ede8673c0d050bca24ea02dd6606d52f816c/docs/public_models/HAILO8/HAILO8_pose_estimation.rst)には次のHEFリンクがある（**リンクをソースで確認しただけで、ダウンロード・読出しはしていない**）。

[YOLOv8m-Pose / Hailo-8 / v2.19.0 HEF候補](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/yolov8m_pose.hef)

同版の[network設定](https://github.com/hailo-ai/hailo_model_zoo/blob/3f19ede8673c0d050bca24ea02dd6606d52f816c/hailo_model_zoo/cfg/networks/yolov8m_pose.yaml)は640×640×3入力、20/40/80の3スケールごとに64/1/51 channelの出力情報を持つ。既存raw decoderの想定形状に対応する候補だが、**HEFの実VStream名、最終出力形式、値の意味は確定していない**。[alls script](https://github.com/hailo-ai/hailo_model_zoo/blob/3f19ede8673c0d050bca24ea02dd6606d52f816c/hailo_model_zoo/cfg/alls/generic/yolov8m_pose.alls)には入力正規化と一部出力のsigmoid指定がある。二重正規化や二重sigmoidを避ける。

| 現在の例の項目 | 実HEF・元モデルで照合する内容 |
|---|---|
| verified_for_hef=false | 実ファイル出所・hash・対象Hailo-8・compiler/runtime適合を記録してから変更 |
| input_shape=[640,640,3] | 実入力数（backendは1入力のみ）、HWC順序、batch=1、RGB/BGR |
| pad_value=114 | letterbox、丸め、padding位置・色、補間、元1280×720への逆変換が元モデルと一致するか |
| input_dtype / input_quantized / scale / offset | host UINT8/FLOAT32、HEF内正規化と量子化、二重処理の有無 |
| decoder.type / heads | raw 9出力か組込NMS等か、出力名/形状/順序を実HEFから取得しREPLACE値を置換 |
| reg_max=16 / stride=8,16,32 | DFL bins、anchor/grid、BBox距離とkeypointの式・単位が一致するか |
| scores / keypoint_scores | 各出力がlogitsか確率か、どの出力へsigmoidが含まれるか |
| missing_joints=none | 0値や低scoreを勝手に欠損化しない。明示的欠損がある場合だけNaN三値へ変換し、必要ならHEF固有decoderを用意 |
| person_score_threshold / nms_iou_threshold=null | 元モデルの選択/NMS条件を照合する。未確認の数値を埋めない |
| decoderの最終契約 | 全人物、入力画素座標のxyxy＋det_score＋COCO17の(N,56)、正常0人は(0,56)、不整合は例外 |

元RGBサイズはメッセージへ追加せず、カメラ・Pose設定・GPUサーバの事前設定を一致させる。公式ベンチマークは別ホスト・PCIe条件の値であり、Pi全系10Hzの根拠には使わない。モデル/重みのライセンスも配布元の条件を確認してから利用する。

## 6. Pi本体で後から確認する項目

以下は未実施であり、今回は問い合わせコマンドも実行していない。

- Ubuntuリリース/arm64、Pi kernelの完全版・headers・ページサイズ、RAM・空きディスク・電源・冷却。Hailo driver構築対象は**ホストkernel**で、コンテナのNoble userspaceとは別。
- 実アクセラレータがHailo-8（26TOPS）であること。PCIe/HAT/FPC接続、既存driver/runtime/FWの版、`/dev/hailo0`等の実パスと所有GID。
- D455実機/FW、USB3接続、`/dev/bus/usb/BBB/DDD`の実パスと所有GID/udev。再接続時の番号変更。コンテナの非root UID1000とgroup_addの実効権限。
- MID-360への実NIC/IP/UDPポート、JSON中の旧参照IP、host networkと競合。元stampの基準がRGBと比較可能か。TFとSDK内extrinsicの二重変換がないか。
- ROS_DOMAIN_IDとサーバの型package、サービス名、EdgeRM runtime/URLの一致。EdgeRMトークンはビルド用Secretでありruntimeへ渡さない。
- Docker localログの取得と標準ローテーション、tmpfs上のROS/Launch一時ログ、停止猶予と全系再起動時の残留。

## 7. 未解決事項・開始条件

1. **EdgeRMソースへのアクセス**。ref/commit/helper API/実依存閉包/COLCON_IGNORE整合を未確認のまま確定しない。
2. **Hailo配布物とホスト版**。4.24.0はソース候補。実ARM64 deb・cp312 wheel・host driver/FWとHEFが揃うまでHAILORT_DEB/HAILORT_WHEELは空欄。
3. **修正したDocker依存の実確認**。RealSense移転URL、rclcpp_action/rosbag2の明示要求、依存列挙・除外、Hailoのpip依存解決禁止をソース上修正済み。private forkの依存充足と実配布物/native ABIは未確認。
4. **追加source lock**。private forkに不足するpackageと互換版が判明してから5変数を埋める。公開repoが存在するだけでは採用SHAを決定しない。
5. **実機設定**。デバイス/GID、Livox IP、取付TF校正、カメラサイズ、時計、モデルmanifestは未確定。

## 8. 次の実機検証の順序（今回実行しない）

ユーザーがPi段階と個々の導入・ビルド・検証を明示的に許可した後に進める。依存不足のままビルドに突入せず、各段階の結果を次の条件にする。

1. PiのOS/arch/kernel/headers/RAM/電源/冷却を記録。既存Hailo driver/FWとD455/USB、MID-360 NICの状態を確認する。
2. Hailoの同一系列driver/runtime/wheelとHEFの出所・hash・ABIを確定。ホストdriverの導入/変更が必要なら別途許可のもと実施し、再起動後にdeviceアクセスを確認する。
3. 通常版向け`.env`を候補4 SHAと実ファイル名/デバイス/GID/Domainで準備。今回のDocker修正内容と実配布物のmetadata/依存/ABIを照合し、未解決のHailo条件を確定する。
4. まず通常ROS版だけをビルドする。依存解決、msg/srv生成、driver/自作ノードのビルド結果を記録し、失敗時はここで止める。
5. D455 RGBのみ、Livox PointCloud2の実topic/QoS/frame_id/tag型、元stampの時計基準を確認。TF/ROIと正常空点群、人物0人/推論エラーの区別、manifest復元を確認する。
6. `ENABLE_BRIDGE=false`で全処理経路と元stampの厳密集約を確認。到着順・欠落・期限・重複・再起動/空結果について、別途許可された検証を行う。
7. 通常ROSサービス受付を用意してブリッジを検証。最大2要求、最新1待機、accepted、拒否/例外/0.5秒期限、late response、Future解放と再送なしを確認する。
8. 権限あるEdgeRMソースでhelper/改変rclpy/全targetsと除外処理を閲覧し、ref+40桁lockと必要な追加lockを確定。通常ROSサービス基準の結果を得てから、EdgeRM版をビルドする。
9. 通常版を停止してEdgeRM版だけを起動。サーバと同じ型/runtimeを用い、新Middleware経路とFuture管理が通常ROSと同じ契約を守るか確認する。
10. それぞれ独立に全系持続10Hz・CPU/メモリ/温度・同期率・滞留・通信量・遅延/欠落を測定。ログと測定条件を残し、実測に基づいて暫定値を調整する。

この文書の調査完了は実機検証開始の許可を意味しない。ここで停止し、次の指示を待つ。
