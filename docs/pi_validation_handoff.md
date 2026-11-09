# Pi検証への引継ぎ（未実行）

**Implementation complete; tests/build/hardware validation NOT RUN**

この文書は将来の検証作業リストです。ユーザーがRaspberry Pi段階を開始し、明示的に許可するまで実行しません。MacBookではソース編集・閲覧のみ行いました。

## 最初に解消する依存

| 依存 | 必要な作業 | 現在の状態 |
|---|---|---|
| Hailo-8 runtime | ホストPCIeドライバ、ARM64 deb、Python3.12 wheelの版を合わせる | 配布物なし、requires Pi verification |
| HEF入出力 | 実YOLOv8m-Pose HEF、HWC寸法、入力量子化/スケーリング、出力名・形状、欠損定義を照合 | HEFなし。raw-head decoderの適合は未確認 |
| 人物選択 | 元モデルの検出条件とNMS値をmanifestに記載 | 未指定。新たな閾値を決定していない |
| 別HEF出力 | 組込NMS等ならHEF固有decoderを実装・接続 | 選択するHEF次第で追加実装が必要 |
| EdgeRM | 改変版Jazzy ARM64ベース、開発用rmw_edgerm_cpp、optional bindingを構築。URL/runtime/サービス名を設定 | バインディングのソースのみ。ABI/型登録/通信は未確認 |
| Driver revisions | librealsense/realsense-ros/SDK2/driverの互換リビジョンを固定 | `.env.example`の空欄をPi段階で決定 |
| USB/Hailo権限 | 実デバイスパスと所有数値GID、非rootアクセス | 未確認。USB再接続でパス変更に注意 |
| Livox IP | JSONのPi NIC/LiDAR IPを実機へ合わせる | 旧参照値を保持、identity外部変換 |

EdgeRMベースの構築・配布はここでは行っていません。標準の`ros:jazzy-ros-base`にはEdgeRM APIがないため、WITH_EDGERM=ONでのビルドにはそのまま使えません。ベースに必要な改変版ROSと開発ライブラリを用意し、apt導入時に改変版が公式版に置換されないよう依存を固定する必要があります。Optional bindingのCMake依存は `ament_cmake_python`, `rclcpp`, `rmw_edgerm_cpp`, Python development, `pybind11_vendor`/`pybind11` です。通常の共通msg/srvのみのビルドはこのオプションをOFFにします。

## 許可後の確認順序

1. 単一DockerのARM64ビルド、Jazzyのインターフェース生成、各entry point、Hailo/RealSense/Livoxの依存と同時アクセス。RSUSBとUSB再接続、非root権限も確認。
2. D455のRGBのみ1280×720×30、Livox PointCloud2 10Hz、実topic/QoS/frame_id。tagがuint8で存在するか。元stampの時計・エポックが揃うか（受信時刻で代用しない）。
3. 静的TFの親子・回転方向と床高さ、前方/左方/上方軸、ROI内の人物点、学習時の床基準との整合。既存TFの重複とドライバ内二重変換を避ける。設計初期値を実測校正値と扱わない。
4. Hailoの全人物COCO17/BBox/score、letterbox逆変換、低scoreの保持、明示的欠損のNaN三値、正常0人と推論エラーの区別。HEFと異なるレイアウトを拒否すること。
5. 集約の3入力全到着順、完全一致と近接不一致、正常0人/ROI0点、PairInfo欠落、片側欠落、期限境界、満杯時の最古破棄、重複・異なるペアによるstamp再使用、完成順、旧起動データと時計巻戻り。空点群のフィールド/step/header、行padding・big-endian入力も確認。
6. サービス受付true/false/例外/サービス未提供/遅延/late response。最大2件・待機最新1件の上書き、0.5秒後のpending解放、再送なし、シャットダウン時の解放。EdgeRMが同じrclpyクライアントのFutureとremove_pending_requestを実際に扱えるか確認する。
7. 同一Domainのサーバで型を共有し、現在はSendSynchronizedFrame受付のみ、将来は最終トピックのみを受信。RGBが意図せず配送されないこと。サーバ側の時刻並べ替え/欠落管理は別途設計する。
8. Pi 5の4コアで全系持続10Hz、CPU・温度・メモリ・通信量・同期率・欠落・遅延・ROSキューと集約/送信滞留。単体Hailo FPSから全系性能を判断しない。移動時の歪みとモデル性能は別に測定する。

## 実装上の運用仮値と制限

- ドライバ入力トピック、購読Best Effort、サービス名 `/send_synchronized_frame`、重複履歴256キー、表示5秒/bridge timer最大20msは実装仮値です。設計の合意値を追加で確定したものではありません。
- 待機TTLは到着からの単調時計。Pi壁時計とセンサstampの同一エポックは仮定しません。RGBとLiDARの時計は比較可能である必要があります。センサstampの大幅な巻戻りや極端な順序逆転は退役下限で拒否されます。全系再起動で状態を初期化します。識別子を追加しない契約上、全系再起動をまたいだ古い通信の完全な識別は保証できず、Volatileと期限管理の実機挙動を確認します。
- 10件/1秒を超える遅延・欠落は破棄され、別ペアで補完しません。ブリッジの受付成功は転倒判定成功を意味しません。応答期限超過でもサーバ側の受信は取り消せません。
- ROS/Launchの標準ログはComposeの`/tmp` tmpfsに置き、Dockerログ永続化は無効です。アタッチしたコンソールで状況確認します。自作ログ保存・詳細収集機構はありません。
- HailoRTの推論はPose専用プロセス内で同期呼出しします。異常なドライバ停止時の推論呼出し復帰、ROS reliable画像キューの滞留、プロセス終了時の資源解放も実機で確認してください。

検証結果はこの文書の未実行状態と混ぜず、Pi段階で測定条件・実測値・失敗内容を別記してください。
