# Raspberry Pi 5 エッジ観測パイプライン

**Implementation complete; tests/build/hardware validation NOT RUN**

このブランチは Raspberry Pi 5 / Linux ARM64 / ROS 2 Jazzy / Hailo-8 / RealSense D455 RGB / Livox MID-360 向けの実装です。正本は [設計v2.6](docs/edge_pipeline_design_v2.6.md)。MacBookではソース閲覧・編集と設計対照レビューのみを行い、ビルド・テスト・コード実行・疑似データ検証・機器照会は行っていません。実機適合や10Hz性能は未確認です。

```text
D455 RGB ─┐            ┌─ /sync/image ── pose_estimation_node ── /pose_frame ─┐
          ├─ sync_node ├─ /sync/pair_info ───────────────────────────────────┤
MID-360 ──┘   ATS      └─ /sync/points ─ lidar_preprocess_node ─ /points_roi ┤
                                                                         ▼
                                                               frame_publisher_node
                                                                         │
                                                              /synchronized_frame
                                                                         │
                                                        topic_to_service_bridge_node
                                                                         │
                                                   SendSynchronizedFrame service / EdgeRM
```

`fall_detection_msgs`は4 msg＋1 srvの共通インターフェース、`fall_detection_edge`は4処理ノード＋一時ブリッジです。PoseとLiDARは別プロセスで処理します。GPUサーバには元RGBを含まない骨格とROI点群だけを送ります。追跡・転倒判定・256点化・床除去・録画機能は含みません。

## 設定と処理規約

- [edge.yaml](src/fall_detection_edge/config/edge.yaml): ATSはqueue 10 / slop 0.0167秒。ドライバトピックと購読QoSは設定可能。初期Best Effort購読はBest Effort/ Reliable両方のPublisherと互換にするための実装初期値です。内部5経路と最終トピックはReliable / Keep Last / depth 10 / Volatile。
- [mount.yaml](src/fall_detection_edge/config/mount.yaml): 親`fall_detection_frame`、子`livox_frame`。並進(0,0,0.85)m、RPY(0,+25°,0)。暫定取付値であり未校正です。点群は元時刻のTFで変換し、X=1–8、Y=−2–2、Z=0–2mの包含境界ROIを適用します。6境界は変更可能です。
- 入力点群のXYZはfloat32、tagはuint8を要求し、行padding・エンディアンを考慮します。出力はXYZ＋元点に対応するtagのみ（point_step=13）、可変点数です。ROI 0点は正常な空点群を出力します。TF/入力契約失敗では出力しません。
- Poseは1280×720のRGBを初期固定値とし、カメラ設定にも同じ値を使います。GPUサーバでもこの元解像度を設定してください。推論時のletterboxと丸めた実リサイズ比を戻し、全人物のBBox・検出スコア・COCO17を元画素座標で出します。低い関節scoreも保持し、明示的欠損だけ3値NaNを許容します。部分NaN・Infは拒否します。正常0人は`persons=[]`、推論失敗は未出力です。
- 集約は元stampの整数ns完全一致のみ。3入力は任意着順を許容し、完成したペアを即Publishします。各入力10件・単調時計1秒TTL、容量超過は古い未完成入力から破棄します。PairInfo前の先着結果も対象です。
- 重複stampは先着を保持し、期限を延長しません。各入力256キーの有界履歴と、それ以前の退役下限で再受付を防ぎます。同一センサstampを別ペアへ割り当てる競合を警告・破棄します。256件は**実装仮値**です。起動時は空のVolatile状態から始めます。長い順序逆転、時計巻戻り、センサ再起動はコンソール確認後に全系を再起動してください。独自IDで曖昧さを隠しません。
- ブリッジは`call_async()`、同時2要求、最新1待機、0.5秒の単調期限です。応答回収・期限切れの管理解放は20ms以下のタイマ周期で行います。拒否・例外・期限超過で再送しません。タイムアウトはサーバの取消を意味しません。

パラメータは起動時に適用します。編集後に全系を再起動してください。Launch管理下のプロセスが終了したら全系を停止し、状態を空にしてから再起動する方針です。TF既設ロボットでは`publish_mount_tf:=false`を指定し、重複TFを避けます。将来トピック転送に移行するときは`enable_bridge:=false`だけでブリッジを外せます。

## 外部依存：requires Pi verification

現段階のファイルは配備用のソースであり、次を埋めた動作済み環境ではありません。

1. **HailoRTとHEF**: [artifacts](artifacts/README.md)にホストPCIeドライバと一致するARM64 runtime debとCPython 3.12用wheel、[models](models/README.md)に実HEFとmanifestが必要です。[manifest例](src/fall_detection_edge/config/pose_model.example.yaml)は未確定値を含み、意図して起動を拒否します。HEF確認後に入力形式、量子化、出力名、後処理、人物選択/NMS条件を記載します。新たな人物/関節しきい値はここで決定していません。
2. **HEF後処理**: 3スケール別のDFL boxes・person score・COCO17 raw headに対するデコーダを用意しています。これが実HEFに適合することは未確認です。組み込みNMS等で出力が異なる場合は、明示したHEF専用decoder契約へ接続する実装が必要です。非互換を空人物配列で偽装しません。
3. **EdgeRM**: ブリッジの`use_edgerm: true`には改変版Jazzyのrclcpp/rclpy、`rmw_edgerm_cpp`、URL、runtime名が必要です。共通メッセージパッケージの任意オプション`FALL_DETECTION_WITH_EDGERM=ON`で今回のサービス型だけのPython bindingを生成します。通常ROSへの暗黙フォールバックはありません。参考にしたローカル別リポジトリと同じAPIパターンですが、ARM64 ABI、サービス登録、同時要求、pending解放は未確認です。サーバ側でも同じ型のbindingと受付サービスが必要です。サーバ実装は本スコープ外です。
4. **Docker入力**: [.env.example](.env.example)は全ドライバの正確なリビジョン、Hailo配布物名、デバイス/グループ、Domainを埋めるテンプレートです。現行EdgeRM運用では`EDGE_ROS_BASE_IMAGE`に必要な改変版Jazzy・開発ライブラリを含むARM64ベース、`WITH_EDGERM=ON`を指定します。例のstock ROS / OFFは通常ROSの構築用であり、既定のEdgeRM設定をそのまま動かせません。ベースの準備とaptが改変版ROSを上書きしない依存固定もPi側で必要です。
5. **センサ・ネットワーク**: Livox JSONの192.168.1.50 / .198は旧参照値です。Pi/NIC/LiDARの実アドレスへ変更し、ドライバ内の外部変換は単位変換（identity）を維持してTFとの二重変換を避けます。実tag型・topic・frame_id・QoS・RGB/LiDAR時計の比較可能性を確認してください。

DockerfileはRSUSB版librealsense、realsense-ros、Livox SDK2/driver、HailoRT、自作2パッケージを新しい単一イメージへまとめるソースです。`network_mode: host`、HailoとD455の実デバイスと数値グループのみを渡し、privilegedやGPU runtimeは使いません。USB再接続時はパス・権限の再確認が必要です。Hailoカーネルドライバはホスト側に置きます。ComposeはDockerのログ永続化を無効にし、ROS/Launch標準の一時ログは`/tmp` tmpfsに収めます。運用時はアタッチしたコンソールで確認します。

## Pi段階の手順（現在は実行禁止）

**ユーザーがPi段階を開始して明示的に許可するまで、下記も含め一切実行しません。**

1. 実機OS/ドライバ/ABIとEdgeRMベースを選び、Hailo配布物・HEF・manifest、`.env`、edge/mount/Livox設定を準備します。
2. ARM64 Pi上でのみ専用イメージを構築し、インターフェースと5ノードの依存を確認します。
3. センサ・TF・単体経路を確認してから全系を起動し、[Pi検証項目](docs/pi_validation_handoff.md)に従います。

許可後の想定コマンド（未実行）:

```bash
docker compose build edge
docker compose up edge
```

通常ROSサービスを明示的に選ぶ場合だけ`use_edgerm: false`とします。Piとサーバで同じ`ROS_DOMAIN_ID`と`fall_detection_msgs`を使用してください。現行はサービスのみ受信し、将来は`/synchronized_frame`のみ購読します。RGBをサーバで購読しません。

## 旧構成と引継ぎ

旧`lcfall_ros2`/`lcfall_msgs`、CUDA/PyTorch/OpenMMLab、転倒分類・警報・可視化・ROIツール・旧検証コンテナ・起動/自動セットアップスクリプトは、このfeatureブランチから削除しました。参照したコードは`oc-demo`に残っており、`oc-demo`、`tpr-demo`、`main`は変更していません。Livox接続JSONは参照として保持しました。

[実装進捗](docs/implementation_progress.md) / [設計とソースの対照](docs/design_to_code_review.md) / [Piへの引継ぎ](docs/pi_validation_handoff.md)。実装完了後も自動検証は開始しません。
