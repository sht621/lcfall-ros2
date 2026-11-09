# エッジパイプライン実装計画（MacBook / Codex）

## 前提
- ベース：`oc-demo`。新規実装ブランチ：`feature/edge-pipeline`。
- 設計の正本：[`edge_pipeline_design_v2.6.md`](edge_pipeline_design_v2.6.md)。
- 既存コードは同期・LiDAR処理・センサ設定等の参考とする。新機能と責務が異なるGPU推論・可視化・旧型等は新ブランチ内で整理・削除してよい。
- **MacBookでの実装完了まではビルド、テスト、実行、疑似データ検証、ベンチマークを一切行わない。** 実装後もユーザーの明示指示なしに検証を開始しない。
- 検証時のみRaspberry Pi 5（Linux ARM64＋Hailo-8＋RealSense＋Livox）で環境構築・動作確認を行う。

## 実装順序
1. **既存ソースの静的調査と整理**：`oc-demo`の設計関連コード／設定のみ参照し、新旧構成の責務を整理する。既存ブランチは変更しない。
2. **ROS 2メッセージとサービス**：`fall_detection_msgs`に`PersonPose`、`PoseFrame`、`SyncPairInfo`、`SynchronizedFrame`、`SendSynchronizedFrame.srv`を設計書どおり定義する。
3. **センサ同期**：`sync_node`でRGB（30Hz）とPointCloud2（10Hz）を元タイムスタンプでATS同期し、`/sync/image`、`/sync/points`、`/sync/pair_info`をPublish。
4. **LiDAR前処理**：`lidar_preprocess_node`で静的TF相当の座標変換と前方ROI、XYZ＋tagのPointCloud2生成、0点正常系を実装。
5. **姿勢推定**：`pose_estimation_node`で画像のHailoRT推論を統合し、全人物COCO17・BBox・信頼度・欠損表現を`PoseFrame`に変換。HEF依存情報とPi検証事項を明示。
6. **同期結果集約**：`frame_publisher_node`で元スタンプの厳密一致・3入力待機管理・完成順Publish・バッファ/TTL/異常系を実装。
7. **一時的なEdgeRM向けブリッジ**：`topic_to_service_bridge_node`で非同期サービス送信（同時2件、待機最新1件、タイムアウト0.5秒、再送なし）を実装。将来ブリッジなしでサーバがトピック受信できる分離を守る。
8. **統合設定と配布**：ROS 2 Jazzy用Launch・パラメータ・Dockerfile・Compose・起動手順・README。ターゲットはPi 5 ARM64。不要な旧構成を新ブランチ内で整理する。
9. **実装完了時の静的レビューのみ**：設計対照表、未確定依存、Piでの実機検証項目、完了/未完了事項を文書化して停止。テスト・ビルドは禁止を継続する。

## Piへ移した後にのみ行う作業（現段階では実行禁止）
- Docker/ROS 2 Jazzy/各センサ/HailoRTの実際の導入とビルド
- HEFモデル入出力、点群`tag`、driver topic/QoS、TF/ROI、通信方式の確認
- 初回起動、機能テスト、10Hz持続性能、4コアCPU負荷と温度、EdgeRM応答遅延計測
- 測定結果に基づくパラメータ調整

## Codexへの最初の依頼
`AGENTS.md`と`docs/edge_pipeline_design_v2.6.md`を最初に読み、設計書に従って`docs/edge_implementation_plan.md`の順序で新規エッジ処理を実装してください。既存コードは参照にとどめ、旧GPU推論構成を維持する必要はありません。**MacBookでは実装完了までビルド・テスト・実行・疑似データ検証を絶対に行わず、実装完了後も自動で検証を開始しないでください。** 不明な実機依存事項はPi検証項目として明記し、実装可能なコードを先に完成させてください。
