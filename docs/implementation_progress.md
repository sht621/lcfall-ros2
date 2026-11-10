# エッジ実装進捗

## オフロード通信量の簡易比較（2026-11-10）

- 実装/設計変更なしでCLI hz/bwを各約20秒測定。圧縮プラグイン未導入のため既存OpenCVの一時Publisher（JPEG品質90）を使用し、終了済み。
- 平均ROSサイズ: RGB921672 bytes、JPEG83982.68 bytes、Pose521.76 bytes。Pose測定窓は検出候補2–3件。共通10Hzの生RGB比でJPEG90.888%、Pose99.943%削減。
- 比較中の本番最終出力は約9.993Hzで継続。実回線性能・詳細最適化は対象外。[条件・コマンド・実測と換算表](offload_bandwidth_comparison_2026-11-10.md)へ記録。既存ローカル変更を保持。

## 10Hz未達の原因調査とDDS共有メモリ設定（2026-11-10）

- ノード内部の一時計測でPose約33–35ms、ROI約4–6ms、ATS後Publish約2–3msを確認。計算処理より前のRGB配送で件数減少が起きていた。
- 既定SHM512KiBはVGA RGB8 payload921600 bytesより小さい。4MiBへ拡大で最終約9.67Hzへ改善、同じtransport構成で512KiBへ戻すとATS約5–6Hzへ再低下。共有メモリ容量不足が主因と判断。
- Fast DDS XMLを追加、参加者SHM16MiB・コンテナ/dev/shm256MiBをComposeへ設定。QoS・ATS・集約・モデル仕様は維持。イメージ再ビルドなし。既存ローカル差分を保持。
- 一時計測コードを外した通常ノードで約60秒、ATS/Pose/ROI/最終それぞれ600件、最終9.999Hz、600/600元stamp一致、重複0。通常ROSは稼働継続。長時間安定性・精度・EdgeRM通信は未確認。
- [調査と比較条件](pipeline_10hz_investigation.md)。ログ例外修正はPythonソースの問題であり、Dockerビルド処理の不具合ではないことも説明。

## RGB解像度640×480への変更と再確認（2026-11-10）

- ユーザー指示でedge.yamlを640×480へ変更しREADMEを更新。D455/Poseの入力寸法を統一、30fps/RGB8とHEF640×640は維持。既存ローカル変更を保持。
- bind mountの設定を使って通常ROSコンテナを再起動。今回ビルドなし。実profileとPoseパラメータを照会確認。
- 約60秒で実画像1798枚/29.943Hz、最終349件/5.823Hz。全件PairInfoと元stamp一致、重複なし、COCO17/BBox/score数値形式エラー0。画像追加購読ありの測定であり、持続10Hzは未達。
- 画像購読なしの再測定は約30秒205件/7.005Hz、全件stamp一致・重複なし。前回と比較して明確な性能改善は未確認。640×480の通常ROS全系を稼働継続。
- [詳細記録](pi_validation_2026-11-10.md)。ログ例外修正の所属はfall_detection_edgeのcommon.py（Console.show）。

## Pi実機でのD455復旧・実画像推論・通常ROS全系確認（2026-11-10）

- ユーザーの明示許可とsudo/Docker権限を確認して実機作業。D455のuvcvideo interfaces一時unbind後、同じ1280×720/30fps/RGB8設定で画像受信成功。競合の詳細原因と再起動時の手動操作不要化は未解決。
- 実画像60秒で597回のHailo推論、COCO17/BBox/scoreの数値契約エラー0、中央値31.978ms。Livox実点群・tag uint8も確認。
- 全系のConsole INFO/WARN切替でrclpy例外が再現。common.pyの呼出し位置を分離する最小修正を行い、実rclpy回帰確認と全系受信を実施。元イメージを別タグ保存し、common.py COPYだけの追加層で通常ROSタグを更新。依存・SDK・ROSの再ビルドなし。
- 約30秒で最終215件、7.216Hz、全件元stampペア一致、重複なし、slop範囲内。通常ROSパイプライン成立を確認したが持続10Hzは未達。サービス通信・精度・長時間負荷は未確認。
- [測定条件と結果](pi_validation_2026-11-10.md)を追記。旧検証コンテナは停止保持し、通常ROS全系はブリッジ無効で稼働中。既存Compose差分などを保持。今回は実機検証と修正ファイルのみのDocker層作成を実施したため、下記の未実施記述は過去段階の履歴。

## D455画像配信のソース調査（2026-11-10）

- ユーザー報告のHailo認識/HEFロード/単色実推論48.2ms成功を受領。過去のHailo初期化失敗記録は履歴として保持し、実RGB推論/全系成立とは区別する。
- 固定RealSense wrapper/SDKとJazzy hzのソースを閲覧。起動完了・Publisher存在は実画像配信の証拠ではなく、画像Publishに購読者数の条件がある。報告のReliable/Transient LocalとhzのBest Effort/Volatileは、この2項目では非互換でない。USB制御転送警告だけから画像停止の原因を断定しない。
- [D455調査手順](d455_image_delivery_investigation.md)を追加。現在のlcfall-d455-testの割当/起動引数/既存ログと、同コンテナ内画像・CameraInfo受信を最初のPi確認として分離。現設定は変更せず、結果に応じてUSB・画像Publish・DDSを切り分ける。
- 原因未確定。既存ローカル差分を保持し、進捗への追記と新規調査文書だけを作成。今回はビルド・テスト・ROS/Docker実行・Piハードウェア操作を実施していない。再ビルド不要。

## Pi実機の通常ROS起動確認（2026-11-10、明示許可後）

- ユーザーのビルド完了報告と通常ROS実コンテナ検証の許可を受け、既存イメージをブリッジ無効で起動。
- D455のRGBプロファイル開始、Livox初期化、静的TF発行をログ確認。PoseはHailo open errno 6で失敗し、全系停止。ホストCLIでも再現。
- Hailoドライバ再読込と対象PCIe resetを試行したが復旧未確認。最終メッセージ受信・実推論・全系10Hzは未確認。Livox終了時exit -11も記録。
- 今回の実測条件・結果は [Pi起動確認記録](pi_validation_2026-11-10.md)。以下の「未実施」は各過去段階の履歴として保持する。

2026-11-10 / `feature/edge-pipeline`。正本: [設計v2.6](edge_pipeline_design_v2.6.md)（変更なし）。

**Implementation complete; tests/build/hardware validation NOT RUN**

## 完了した実装段階

1. AGENTS、設計、実装計画を読了。`oc-demo`の同期ノード、LiDARユーティリティ、Launch、Dockerをソース参照。旧来の1人選択・256点化・GPU分類は移植しないと整理。
2. `fall_detection_msgs`に確定どおり4 msg＋SendSynchronizedFrame.srv、ament_cmake定義を追加。
3. `sync_node`にraw RGB/LiDAR ATS（10件/0.0167秒）、元stamp維持、3経路出力、driver別QoS、入力不正/同期停止のコンソール表示を実装。
4. `lidar_preprocess_node`に元時刻の非待機TF照会、前方ROI、XYZ float32＋対応するtag uint8のみの出力を実装。行padding/エンディアン対応、0点正常出力、TF/契約失敗の未出力を区別。
5. `pose_estimation_node`にROS Image→RGB→letterbox→Python/HailoRT vstreams→元画素座標の全人物COCO17/BBox/scoreを実装。欠損三値NaN/部分NaN・Inf拒否、正常0人/推論失敗未出力を区別。HEF manifest、raw-head decoderと実HEF専用decoderの境界を用意。
6. `frame_publisher_node`に3入力任意順、整数ns完全一致、完成即Publish、各入力10件・単調時計1秒TTL、容量超過最古破棄、重複と退役stampの有界管理を実装。
7. `topic_to_service_bridge_node`に非同期call_async、最大2要求・最新1待機、0.5秒単調期限、非再送、rclpy pending解放を実装。任意EdgeRM bindingはこの1サービス型だけを扱い、通常ROSへの暗黙fallbackなし。
8. 5ノード/ドライバ/静的TFのLaunch、ROS設定、Pi ARM64単一Dockerfile/Compose、配布物とmodelの入力仕様、READMEを追加。旧GPUアプリ・型・可視化・ROIツール・検証コンテナ・自動セットアップ/実行スクリプトをfeatureブランチ内で削除。Livox JSONと既存data除外は保持。
9. [設計対照レビュー](design_to_code_review.md)をソース閲覧のみで実施。[Pi引継ぎ](pi_validation_handoff.md)に依存/不足情報/将来の検証項目を分離。設計・計画・AGENTSと保護ブランチは未変更。

## 実装判断・参照

- パラメータは起動時適用。ROI/TFの設定変更は全系再起動で反映。既存TFがある場合は専用publisherを無効化する。
- 内部QoSは確定値。driver購読Best Effort、実入力トピック、サービス名、重複履歴256キー、表示周期は実装仮値。設計の承認値として追加決定していない。
- 待機時間は単調時計。センサstampとPi壁時計の同一エポックを仮定した破棄条件は採用しない。起動時は空のVolatile状態、異常終了時は全系停止。完全な再起動epoch識別は現行メッセージにないため、残留挙動はPi確認事項。
- 隣接に`fall-collection`というディレクトリはなかった。`/Users/sunsun/LCFall/Dockerfile.sensor`も閲覧しJazzy/Livox構築の参考とした。同ファイルはapt RealSenseの参照であり、RSUSB設定がそこにあったとは扱わない。新DockerfileでRSUSB版ソース構築を記述。
- `/Users/sunsun/robot-agent-edgerm/src/multimodal_robot_middleware/`と`robot_agent/agent_node.py`の型別EdgeRM binding/APIをソース参照した。今回のoptional bindingは同じ形だが、ARM64で利用可能/動作済みという証拠ではない。
- HailoRT接続は[公式Python bindingソース](https://github.com/hailo-ai/hailort/blob/master/hailort/libhailort/bindings/python/platform/hailo_platform/pyhailort/pyhailort.py)のvstream APIを参照。バージョン/ABI適合は未確認。
- 旧ファイル削除を含む一括編集コマンドは自動承認レビューで拒否され、実行されなかった。新規編集と対象を明示した`git rm`に分け、許可されたソース整理を完了した。

## 不足情報・Piでのみ解消する項目

**requires Pi verification**:

- HailoホストPCIe driver/runtime deb/Python3.12 ARM64 wheelの一致。バイナリ配布物は未提供。
- YOLOv8m-Pose HEF/manifestの実入出力、人物選択/NMS、欠損表現、量子化、座標復元。実HEF次第で専用decoderの追加が必要。
- EdgeRM改変版Jazzy ARM64ベース、開発ライブラリ、サービス登録、URL/runtime名、同時要求とpending解放。標準JazzyにEdgeRMは含まれず、実環境の準備は未実施。
- 各ドライバの固定リビジョン、Dockerビルド依存、D455 RGBのみ1280×720×30、tag uint8、実topic/QoS/frame_id、元stampの時計整合。
- デバイス/GID、Livox IP、静的TF(0,0,0.85)m / RPY(0,+25°,0)の校正、床基準・前方ROI、空結果伝送。
- 全系持続10Hz、CPU/温度/遅延/滞留、サービス期限境界と遅延応答、再起動残留、移動中の性能。

ビルド・テスト・構文検査・実行型lint・ノード起動・疑似データ・ベンチマーク・ハードウェア照会・インストール・配備は一切実施していない。コンパイル可能、10Hz達成、ハードウェア互換、production-readyとは主張しない。ユーザーがPi段階を開始し、明示的に検証を許可するまで停止する。

## Docker両環境対応のソース更新（2026-11-10）

- 前回の未コミット公式Middleware API移行（型別binding削除）を維持。今回の根拠は`Downloads/robot-agent-edgerm-main.zip`（commitコメント`be22bb43865e3aea8185929a00ef8800f35bfd93`）であり、旧ローカルリポジトリは再参照していない。
- 通常ROS専用DockerfileとUbuntu 24.04から改変Jazzyを構築するDockerfile.edgermを分離。Composeはedge/edge-edgermの1ファイル。EdgeRMのみprofileとBuildKit Secretを使用。
- build-tools.shのgit-cloneをSDK/driver/EdgeRMソース取得から実使用し、固定commit確認・submodule・専用cache内限定のreset/clean・tmpfs askpassを実装。認証情報をremote/ARG/ENVへ保存しない。colcon-buildはpackages-up-to、テスト無効、逐次パッケージと並列ジョブ数を管理。
- ros_entrypoint.shをルートへ統一し、旧docker/entrypoint.shと空になったdockerディレクトリを削除。両setup必須・読込失敗時停止・execを共通化。
- Launchに通信パラメータの明示上書きを追加し、Composeからmode/URL/runtime/verbose/service名をブリッジだけへ渡す。4ノードの処理、msg/srv、TF/ROI、bridge送信制御は変更していない。
- `.env.example`、`.dockerignore`、README、AGENTSの日本語完了報告を更新。以前のソースのみレビュー/Pi引継ぎにある旧bindingオプションと単一Docker切替の説明は過去時点の履歴であり、現行の構築説明はREADMEを優先する。
- 未解消: ZIPにEdgeRM本体とoptサブモジュールの中身はなく、GitHubの非公開ソース読出しも404。`src/rosdep-install.py`の内部実装/選択commitのパッケージ一覧は確認できない。参考と同じ呼出し方と存在guardを記述。公開manifestから画像・診断・PCL依存を整理し、不足時の固定ソース追加を記述したが、固定SHAとJazzy/ARM64適合は未選定。requires Pi verification。
- 後のPi項目: 両Dockerビルド、apt pinでの依存充足、Hailo/CPython ABI、センサ権限・時計・tag、Launch設定反映、通常ROS/EdgeRM通信とFuture解放、全系10Hzとメモリ/温度。今回もビルド・テスト・lint・構文チェック・ROS/Docker実行・疑似データ・ハードウェア検証・コミット・Pushは行っていない。ソース差分確認後に停止する。

## Docker静的レビュー指摘の修正（2026-11-10）

- 共通ログ設定をlocalドライバへ変更。max-size/max-fileは指定せず、両サービスに適用。READMEへlogs -fの操作例を追記。ROSノードのログ出力は未変更。
- image_commonはimage_transport/camera_info_managerと必要なcamera_calibration_parsersを個別確認。不足時に固定commitを取得し、追加側の既存同名パッケージ・不要パッケージをCOLCON_IGNOREで除外する。その他の補完リポジトリも同じ除外処理を適用し、追加後に一覧を更新する。
- 固定SHAは推測せず未設定を維持。非公開EdgeRM helperの実装・COLCON_IGNOREの扱い、選択commitの依存充足、両サービスのログ取得/ローテーションは requires Pi verification。ソース閲覧と編集・Git差分確認のみで、ビルド/テスト/lint/構文チェック/実行/コミット/Pushは未実施。

## Pi環境準備の調査（2026-11-10）

- ローカルとGitHubのfeature先頭が442477075621f0c5dc780a830b01e9f1947e318cで一致、開始時の作業ツリーはクリーン。pull/fetchや既存ソースの変更はしていない。
- docs/pi_environment_preparation.mdを作成。公式release/sourceとtagのcommit objectから、RealSense SDK v2.56.4/ROS wrapper 4.56.4、Livox SDK v1.3.1/driver 1.2.6の4 SHAを候補として.env.exampleへ反映。これは実機適合の確定ではない。
- EdgeRM非公開repoとhelperは今回もNot Found/404。ref/commitと追加5 source lockは未確定のまま。rclcpp_actionのCMake必須要求とEdgeRM targetsの照合、Livoxのrosbag2宣言依存、RealSense URL移転、NumPy ABIの確認点を報告し、Dockerfile自体は改修していない。
- Hailo-8用4.24.0と同版PCIe driver、Model Zoo v2.19.0をソース候補として整理。cp312 ARM64 wheel/debの実ファイルは未確認。公式HEF配布リンクとmanifest照合項目を記録したが、配布物取得・HEF読出し・Hailo/センサ照会はしていない。
- ビルド・テスト・lint・構文チェック・ROS/Docker実行・実機検証・インストール・コミット・Pushは未実施。調査・設定差分の閲覧後に停止する。

## Pi Dockerビルド前の依存修正（2026-11-10）

- 作業ブランチfeature/edge-pipelineを確認。前回調査の未コミット変更を保持し、両DockerfileのRealSense URLをrealsenseaiへ更新。選定済み4 SHAは変更していない。
- 固定RealSense CMakeのrclcpp_action必須要求とLivox ROS2 manifestのrosbag2実行時依存を公開ソースで確認。通常版は対応ROS deb、EdgeRM版はfork内の必須source targetsへ明示。rclcpp/rosbag2を別のstock coreから補完する変更はしていない。
- EdgeRMのrosdep helperへ渡す名前をcolconのpackages-up-to依存集合へ統一。追加sourceの重複を先に除外し、必要な推移依存の兄弟packageを残してから不要分へCOLCON_IGNOREを付ける。非公開helperの引数・除外挙動、forkの実依存と既存COLCON_IGNOREは未確認。
- 共通install-hailortを記述。将来のDockerビルドでdeb/wheelの版・ARM64/CPythonタグ・依存metadataを照合し、apt側Python依存を保持してpipをno-deps/no-indexに制限する。実deb/wheelが未提供なので対応版は固定せず、Hailo/NumPy/cv_bridge/OpenCVのnative ABIはrequires Pi verification。
- READMEと環境準備文書を更新。4処理ノード、ブリッジ送信制御、msg/srv、設計v2.6、Composeの2サービス構成は変更していない。
- ソースとGit差分の閲覧のみ。ビルド・テスト・lint・構文チェック・ROS/Docker実行・インストール・実機検証・コミット・Pushは未実施。Implementation complete; tests/build/hardware validation NOT RUN。次の指示まで停止する。

## HailoRT 4.23.0のビルド時自動取得（2026-11-10）

- ユーザー確認済みのPi 5 / Debian 13 ARM64 / HAILO8 / ホストPCIe driver・FW 4.23.0に合わせ、コンテナruntimeとcp312 ARM64 wheelを4.23.0固定。ホストは変更しない。
- Hailo公式コミュニティの配布案内と、指定dev-public.hailo.aiの2 URLへのHTTPS HEAD 200を確認。公開SHA256は投稿・HTTPヘッダから確認できず、固定値を推測していない。本体ダウンロード・metadata読出し・インストールは今回未実施。
- 共通install-hailortを固定HTTPS自動取得へ改修。取得失敗時は停止、リダイレクトもHTTPS限定、別版・ミラーへのfallbackなし。deb/wheelの版がともに4.23.0であること、ARM64/CPythonタグ・依存metadataの照合とpipのno-deps/no-indexを維持。終了trapで一時ファイルを削除する。
- 両DockerfileからartifactsのCOPYとHAILORT_DEB/HAILORT_WHEELのARGを削除。通常版にcurlを明示追加。Composeと.env.exampleから旧2変数を撤去。README・環境準備文書・artifactsの説明を更新し、HEF/manifestは引き続き別途準備と明記。
- 4処理ノード、EdgeRM API、ブリッジ送信制御、msg/srv、設計v2.6、2サービス構成は未変更。native ABI、実HEF互換、実通信・Future管理・全系10Hzはrequires Pi verification。
- 今回は公開配布先の調査・ソース編集・静的差分確認のみ。ビルド・テスト・lint・構文チェック・ROS実行・Pi操作は未実施。コミット・Pushは今回のユーザー指示に従って行い、その後停止する。

## YOLOv8m-Pose HEFの自動配置とmanifest調査（2026-11-10）

- feature/edge-pipeline、開始時クリーンを確認。HailoRT 4.23.0とDebian 13 ARM64ホストのdriver/FW 4.23.0は維持。
- 公式Hailo Apps対応表でRT 4.23.x→Model Zoo v2.18.0を確認。公式ZooのGit tagはv2.18、commit dc402c7ae70149c516324ef9ddff32d27609ae06。Hailo-8 pose表のv2.18.0配布リンクとHTTPS HEAD 200（31,609,032 bytes）を確認。HEF本体の取得・解析・実行、SHA256確認は未実施。
- prepare-pose-modelを両Dockerfileから呼び、固定HTTPSのHEFを/models/yolov8m_pose.hefへ取得する記述を追加。失敗・空ファイルで停止し、別版fallbackなし、一時ファイルは削除。models/pose_model.yamlをイメージへCOPY。Git/Dockerの除外設定はmanifestだけ許可し、Composeのmodels bind mountは撤去。既存hef_path/model_manifest/PYTHONPATHは維持。
- 公開ネットワーク/base/ALLS/CPU後処理を静的照合。640×640 RGB、9 raw heads、ネット内正規化、DFL 16 bins、COCO17の復元式を確認。既存rawは公式のstrict score>0.001/max=300を再現しておらず、実HEFのVStream/量子化/NMS有無も未確認。互換とは判定しない。
- manifestはテンプレートを検証済みとしてコピーせず、公開ソースの事実と未確定値を記録した未承認設定として管理。verified_for_hef=falseを維持し、デバイス設定前に起動拒否。自動配置は実装したが、手動配置不要での推論開始はmanifest/decoder未確定のため未完了。必要なら既存custom decoder APIで対応する。
- README、models/README、環境準備文書、artifactsの旧案内、.env.exampleを更新。設計v2.6、4処理ノード、msg/srv、通信・同期・点群処理は未変更。
- requires Pi verification: 実HEF I/O/量子化/NMS、前後処理・欠損意味、HailoRT ABI、推論と全系10Hz。今回は静的差分確認のみで、ビルド・テスト・lint・構文チェック・ROS実行・Pi操作・コミット・Pushは行わず停止する。

## 実HEF解析結果に基づくmanifest・rawデコーダ修正（2026-11-10）

- feature/edge-pipeline、開始時クリーンを確認。ユーザー提供のPi/HailoRT 4.23.0解析結果を根拠に、HAILO8、UINT8 NHWC入力/scale=1/zero point=0、9 raw VStreamの名前・形状、NMS出力なしをmanifestへ反映。MacでHEF解析を再実行していない。
- 指定SHA256 54f22e23ee5e161e3b5d8507add82035122f72fa6357b453eb8773c63c0eb4b7をmanifestとprepare-pose-modelへ固定。公式公開checksumではなくPi実測の提供値。将来のDockerビルド時に一致しなければ停止。URL・Model Zoo版・HailoRT版は変更していない。
- 固定commit dc402c7ae70149c516324ef9ddff32d27609ae06のpose前処理、Python後処理、Cython NMSを閲覧。DFL 16 bins/XY復元を維持し、strict score>0.001、NMSの+1面積・IoU>=0.7抑制・二段argsort/返却順、NMS後最大300件を実装。sigmoidはlogitをclipしない安定形へ。正常0人はfloat32 (0,56)。関節閾値や人物の事前top-kは追加しない。
- 前処理の静的照合でINTER_LINEAR縮小/寸法roundとの差が判明したため、pose_processing.prepareのみ公式の寸法切捨て・縮小INTER_AREA・中央paddingへ修正。元座標復元と出力契約は維持。同期・点群・通信・msg/srv・設計v2.6は未変更。HailoBackendのquantized=false/FLOAT32逆量子化も未変更。
- manifestはHEFメタデータ・公式前後処理・実装を静的照合した根拠でverified_for_hef=true。verification.inference_validated/performance_validated=falseを別記。実推論成功・精度・性能の検証結果ではない。README/モデルREADME/環境準備文書と旧案内を更新。raw設定例に必須max_detections項目を追加。
- requires Pi verification: ビルドとSHA256照合、実推論・数値比較・元画素座標復元、正常0人/複数人・境界・同score時のNMS、NumPy/OpenCV/HailoRT ABI、Python NMS最大8400候補の最悪O(K²)時間・メモリ、Pose/全系p95/p99遅延と持続10Hz。
- ソース・設定編集と静的差分確認のみ。ビルド・テスト・lint・構文チェック・ROS実行・Pi操作・コミット・Pushは未実施。検証は自動開始せず、次の指示まで停止する。

## Livox SDK2の上流ビルド修正commit採用（2026-11-10）

- feature/edge-pipeline、開始時クリーンを確認。公式compareで旧固定f5d9375f84efe2b15bc0a052d3e18482ed13adf4から68ae1e1dc77f61f03c95d7c2809831e198d0aeddが直後の1 commitであること、Ubuntu 24.04修正とcomm/define.h・logger_handler/file_manager.hへのcstdint追加を確認。
- Driver 1.2.6固定13eb05e4e6dd7a765b934d0c5fd6236676a57b49のROS2 CMake、初期化/設定callbackとSDK公開API差分を静的照合。使用中のAPI・callback・点群構造体、共有ライブラリ名・header install先、C++11/pthreadの構築条件に非互換は見つからず採用。
- .env.exampleのLIVOX_SDK2_COMMITを更新。SDK内version macroは1.3.3のため、環境準備文書・READMEもv1.3.1後の修正commitとして更新。Driver SHAは維持。SDKソースへのパッチ・強制include・Dockerfile変更は行っていない。
- 上流差分にはMID-360S API追加、DirectLidarStateInfo末尾拡張、ログ/debug記録変更も含む。既存バイナリのABI互換を保証するものではなく、SDKとDriverを共にソース再ビルドする構成。Piでのビルド・実通信・ログ/停止挙動はrequires Pi verification。
- 無関係な処理・設計v2.6は未変更。公式ソース閲覧・設定編集・静的差分確認のみ。ビルド・テスト・lint・構文チェック・ROS実行・Pi操作・コミット・Pushは行わず停止する。

## HailoRT debのsystemd失敗修正とキャッシュ境界分離（2026-11-10）

- ユーザーのPi Dockerビルドでhailort postinstのsystemd接続失敗を受領。公式4.23.0 ARM64 deb/cp312 wheelをMacの一時領域へダウンロードし、ar/bsdtar/objdumpによる静的閲覧のみ実施。deb control、preinst/postinst/postrm、payload、wheel metadata、CLI/libhailort/Python拡張のDT_NEEDEDを確認。配布物・maintainer scriptは実行していない。
- debにDepends宣言はなく、postinstはlibhailort.soリンク作成と/.dockerenv条件下のsystemctlを含む。aptでdebをconfigureする処理を撤去し、dpkg-deb --extract後に直接推論用ライブラリ・CLI・header/CMake・copyrightだけを配置。必要リンクとldconfigを補完し、ELFのC/C++依存はaptのlibc6/libstdc++6/libgcc-s1で明示準備する記述へ。
- 公式4.23.0 get_vdevice_paramsのmulti_process_service=falseと現行hp.VDevice()をソース照合。サービスunit/実行ファイル・GStreamer plugin・kernel driverは導入せず、サービス専用preinstディレクトリも作らない。runtimeはdpkg登録しない。動的リンク・非rootログ権限・サービスなし実推論はrequires Pi verification。
- install-hailort/prepare-pose-modelを新hailo-runtime-tools.shへ移し、両DockerfileのSDK/driverソース工程後（EdgeRMはdriver構築後）でCOPY/sourceする。分離時はbuild-tools.sh変更により前段キャッシュが一度無効になるが、その後の新ファイルだけの変更では前段を維持できる配置。通常版の後段driver/app colconは再実行対象。実キャッシュ利用は未確認。
- HailoRT/ホストdriver 4.23.0、HEF URL/版/SHA、pip no-deps/no-index・NumPy保持は維持。ROSノード・msg/srv・設計v2.6は未変更。README、環境準備文書、配布物案内、.env.exampleの参照を更新し、docs/hailort_deb_review.mdへ確認根拠と省略の影響を記録。
- MacでのDocker/ARM64ビルド、テスト、lint、構文チェック、CLI/Python拡張実行、ROS実行、Pi操作、コミット・Pushは未実施。静的差分確認後に停止する。
