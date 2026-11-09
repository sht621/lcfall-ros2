# 設計v2.6と実装ソースの対照レビュー

実施日: 2026-11-10。対象: `feature/edge-pipeline`。
**ソース閲覧のみのレビューです。構文検査・lint実行・テスト・ビルド・動作確認を含みません。**

正本の設計書、AGENTS、実装計画は変更していません。設計合意とソース実装完了と実機確認を区別します。

| 設計条件 | 実装箇所 | ソースで確認した内容 |
|---|---|---|
| §5.2/§10.2–10.6 共通型 | `src/fall_detection_msgs/msg/`, `srv/` | 4 msg + 1 srv。フィールド追加なし。COCO17固定3配列、Pose header、2 stamp、pose+points、frame/accepted |
| §2–4 先ATSは1回のみ | `sync_node.py` | 生Image/PointCloud2にmessage_filters ATS、queue=10、slop=0.0167秒。選択済みデータだけを後段へ渡す |
| §4 元時刻 | `sync_node.py`, `pose_estimation_node.py`, `pointcloud.py` | Image/Poseは元RGB header、cloudは元LiDAR stamp、PairInfoは両stampの複写。合成stamp・pair_idなし |
| §5/§10.7 topics | 各nodeと`config/edge.yaml` | 内部5トピック＋`/synchronized_frame`を確定名で使用。driver入力のみ設定可能 |
| §5.3 Python/HailoRT | `hailo_backend.py`, `pose_processing.py`, `pose_decoder.py` | ROS RGB変換→letterbox→Hailo vstreams→明示デコーダ→元画素座標。カメラ直接取得・GStreamer・GPU推論なし |
| §10.3–10.4 姿勢契約 | `persons_from_rows()`, `pose_estimation_node.py` | 全検出を出力。有限の低score保持、3値NaNだけ欠損許可、部分NaN/Inf拒否。正常0人と失敗未出力を区別 |
| §5.5 静的TF/ROI | `launch/edge.launch.py`, `config/mount.yaml`, `lidar_preprocess_node.py` | 親fall_detection_frame・子livox_frame、0.85m/+25°初期値。元時刻でTF照会、追加傾斜補正・動的地図ROIなし。ROI6境界パラメータ |
| §10.5 点群 | `pointcloud.py`, `lidar_preprocess_node.py` | XYZ float32＋tag uint8を要求し、元点との対応保持。4フィールドのみ。行padding/エンディアンを考慮。ROI空でも有効形式を出力。失敗未出力 |
| §4.3/§10.8 集約 | `collector.py`, `frame_publisher_node.py` | 整数nsの完全一致、3入力任意順、完成即Publish、消費済み結果削除。0人/0点をtruthinessで除外しない |
| §4.3 有界性 | `Collector.add()/expire()` | 各入力10件、単調時計1秒TTL。先着結果も対象。重複が受信時刻を更新しない。容量超過時は入力内で受信が最古の未完成項目から破棄 |
| §10.8 重複/残留 | `Collector.seen/floors`, Launch終了管理 | 有界キー履歴で重複/退役stampを拒否し、stamp再使用のPairInfo競合を診断。起動時に状態を空にしVolatile購読。異常終了時は全系を止める |
| §5.1 bridge | `topic_to_service_bridge_node.py` | call_async、同時2・未送信最新1、0.5秒単調期限、最大20ms周期の完了回収。受付拒否/例外/期限超過の警告、再送なし。pending request削除とFuture cancel、終了時解放 |
| §5.1 EdgeRM分離 | `fall_detection_msgs/src/edgerm_service.cpp`, `fall_detection_edgerm/` | 今回の1サービス型だけの任意バインディング。4処理ノードにEdgeRM依存なし。enable_bridge=falseで将来撤去。汎用通信adapterなし |
| §6.2 QoS | `common.internal_qos()`, `sync_node.driver_qos()` | 内部5経路/最終topicはReliable/Keep Last/10/Volatile。driverは別指定の購読QoS。service区間は別扱い |
| §5.4 配備 | Dockerfile/Compose/Launch | Jazzy ARM64単一イメージ/コンテナ。RSUSBソース構築、Livox SDK2/driver、Hailo配布物、自作ノード。ホストnetwork・必要デバイス/数値GID。privileged/CUDA/PyTorchなし |
| §6.3 表示 | Console、Composeログ設定 | 簡素な5秒間隔の成功/警告表示。独自ファイルログ/集計なし。ROS/Launch既定ログは/tmp tmpfs、Dockerログ永続化なし |
| §1.2 スコープ | src構成、旧コード削除 | Pi上の分類/追跡/警報/可視化/収集/256点化を除去。driver参照JSONを保持。保護ブランチ変更なし |

## レビューで整理した点

- `rcl_interfaces`の明示依存を追加。Dockerで既存UID1000と衝突しない実行ユーザー指定へ調整。旧`.gitignore`の`data/`除外を保持し、既存データを新実装の成果物として扱わない。
- 元stampとPiの壁時計が同じエポックとは設計で確定していないため、起動時壁時計との比較によるデータ破棄は採用しない。待機期限は設計どおり単調時計。起動時の状態初期化、Volatile、重複/退役キー、全系をまとめた再起動で残留を管理する。
- `PoseFrame`未検出とLiDAR ROI 0点は正常出力、例外は未出力で統一。tagがなければ捏造しない。
- HEFのscore条件を設計に新設せず、manifestの必須値として未設定のまま提示。サンプルHEF寸法/出力名を実機確認済みと記載しない。
- EdgeRM対応は別リポジトリの型別middlewareパターンを参照。stock ROSでは存在しないAPIなので、明示選択・任意ビルドとし、暗黙の通常ROSへのfallbackを除外。

## 未解消の外部依存と実機リスク

**requires Pi verification**: HailoRT/ホストdriver/Python ABI、実HEFとmanifest、HEFに応じた追加decoder、EdgeRM改変版Jazzy ARM64ベースとサービス登録/ABI、driver互換リビジョン、必要なビルド依存の実際の充足、デバイス権限、Livox IP/tag/時計、TF/ROI校正、QoS適合、空結果通信、全系10Hz性能。

Epoch識別子を持たない契約なので、全ノードが新プロセスに入れ替わった後の古い通信と新しい通信をstamp以外で完全に区別することはできません。初期化とVolatile/期限管理を実装し、再起動時の残留とセンサ時計巻戻りの実際の挙動はPi段階で確認します。サーバ側の時系列並べ替え・欠落処理と転倒判定は対象外です。

既知のソース実装範囲は完了。配布物/実HEF/EdgeRM環境に依存する追加設定・適合修正は[Pi引継ぎ](pi_validation_handoff.md)へ分離しています。コンパイル可能・動作済み・10Hz達成・ハードウェア互換とは主張しません。

**Implementation complete; tests/build/hardware validation NOT RUN**
