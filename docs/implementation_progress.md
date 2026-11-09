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
