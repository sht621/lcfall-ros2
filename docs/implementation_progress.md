# エッジ実装進捗

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
