# 通常ROSパイプラインの10Hz未達調査

2026-11-10、Pi TB-40、Debian 13 ARM64、通常ROS 2 Jazzy、640×480 RGB8/30fps。
ユーザーから原因調査と実機操作を許可された範囲で実施。実機結果とソース閲覧の推定を区別する。

## ログ例外修正とDockerの関係

ログの例外は `fall_detection_edge/fall_detection_edge/common.py` のConsole.showにあるPythonソースの問題だった。rclpyが呼出し位置ごとのseverityを保持するため、同一行でINFO/WARNを切り替える式を別行のif/elseへ変更した。

コンパイルやROS/ドライバの再ビルドを必要とする修正ではない。ただしDockerfileはCOPYとament_python installでイメージ内にPythonファイルを配置し、通常Composeのbind mountは設定ファイルだけでソースは含まない。したがってホストのソース編集だけではコンテナ内の実行ファイルは変わらない。docker cpや開発用bind mountで反映でき、前回はさらに再作成時にも修正を保持するためCOPYだけの追加イメージ層を作った。Dockerのビルド処理そのものの不具合ではない。

## 変更前の切り分け

- 生LiDARの観測は10.001Hz、ATSペア約5.525Hz、Pose約5.561Hz、ROI約5.524Hz、最終約5.154Hz（約30秒、5トピックReliable購読）。ペアの両処理結果は全164件について観測できたが、一部は集約期限等で出力に至らなかった。観測順序の逆転があり、観測者上の到着差は真のノード処理時間と同一ではない。
- 実行中コンテナだけに一時計測コードを追加して再起動。約5秒窓でノード内処理時間を計測。Pose callback中央値約33–35ms、backend推論等約31–33ms、ROI約4–6ms、ATS後3 Publish合計約2–3ms。100msを超える計算時間が定常的な律速という証拠はなかった。
- 同じ内部計測でLiDARは約10Hz届くが、RGBの受信が30Hzを下回り、ATSペアも約5–8Hzに変動。例: 5秒の受信増分RGB128件/LiDAR50件/ペア35件、別窓ではRGB101件/LiDAR48件/ペア24件。
- Fast DDSのSHM実ファイルは約549408 bytes（領域設定512KiBに管理領域を含む）。640×480 RGB8のpayloadだけで921600 bytes。公式資料は領域サイズがメッセージサイズに近い/小さい場合の損失を警告する。
- UDP受信エラーの既存カウンタも確認したが、このカウンタだけを今回の欠落原因とは断定せず、sysctlは変更しなかった。

## 領域サイズの比較

| 条件 | 実測・結果 |
|---|---|
| 元の既定SHM 512KiB | ATS約5–8Hz、RGB配送欠落を内部受信件数で確認 |
| 明示UDPv4＋SHM 4MiB | 内部計測で5秒50ペアの窓を確認。約30秒の最終出力291件/9.666Hz、全件PairInfo一致、重複0 |
| 同じ明示UDPv4＋SHMを512KiBへ戻す | 再起動後の5秒窓で26ペア/5.20Hz、31ペア/6.20Hzへ再低下。RGB受信増分117件、LiDAR50件の窓もあり。計算処理の時間は小さいまま |
| SHM 16MiB、コンテナ/dev/shm 256MiB | 一時計測コードを外した通常ノードで、後述の約60秒全系10Hzを観測 |

主因は画像サイズに対するFast DDS共有メモリ領域不足と判断する。再起動だけの効果と区別するため、同じ明示transport構成で512KiBへの復帰比較を行った。4MiBでは改善したが一部欠落が残ったため、画像キューを保持する余裕を設けて16MiBへ拡大した。長時間・あらゆる負荷に対する最適値の証明ではない。

## 反映した修正

- `config/fastdds/edge.xml`: UDPv4とSHMを明示し、参加者ごとのsegment_sizeを16777216 bytesとする。Discoveryと同一ホスト外のUDP通信を維持。独自Publisher/Subscriber QoSを定義しない。
- `compose.yaml`: FASTRTPS_DEFAULT_PROFILES_FILEを指定しXMLをreadonly bind mount。参加者ごとに領域が作られるため、コンテナのshm_sizeを256mへ拡大。7参加者の実ファイルと/dev/shm使用量約119MiB/256MiBを確認。
- 640×480/30fps/RGB8、ATS queue10/slop0.0167秒、内部Reliable/Depth10/Volatile、集約各10件/1秒、HEF、ROI/TF、msg/srvは変更していない。
- Compose共通runtimeに適用する記述なのでEdgeRMサービスにも同設定が渡るが、EdgeRM版の起動・性能は今回未検証。
- イメージ再ビルドなし、コンテナ再作成のみ。一時計測コードは再作成で除去し、common.pyはリポジトリ版と同じSHA256へ戻ったことを確認。リポジトリの通常ログへ性能計測機能を追加していない。
- 既存のComposeローカル変更（HAILO_GIDの削除）と以前の未コミット作業を保持。

## 最終確認（計測コードなし）

通常ROS全系、ブリッジ無効。8トピックを1つの診断プロセスでBest Effort購読。起動から十分経った状態で62.001秒実行し、先頭2秒を除いた約60秒を集計。

| トピック | 件数 | 受信Hz |
|---|---:|---:|
| /camera/camera/color/image_raw | 1769 | 29.944 |
| /livox/lidar | 594 | 10.000 |
| /sync/pair_info | 600 | 10.000 |
| /sync/image | 600 | 9.999 |
| /sync/points | 600 | 9.999 |
| /pose_frame | 600 | 9.999 |
| /points_roi | 600 | 10.000 |
| /synchronized_frame | 600 | 9.999 |

生センサ2トピックの件数が少ないのは観測開始時のDiscovery到達時刻が他経路より遅いことと区別する。各Hzはそれぞれの初回～最終受信時刻から計算。生センサの全送信数や無欠落を保証する表ではない。

- 最終600/600件が観測したPairInfoの元stampペアと完全一致、重複0。
- 画像640×480/rgb8。最終サンプルは7件の人物候補、ROI2386点、fall_detection_frame、XYZ float32＋tag uint8。最終サンプルのstamp差5.119ms。
- **今回の約60秒条件で全系10Hz相当を確認した。** 長時間の安定性、p95/p99のセンサからの遅延、移動中性能、姿勢精度、EdgeRM/サービス通信は未検証。起動直後のTF/Discoveryに伴う欠落、既知のD455 USB競合とLivox終了処理の問題も別課題。
- 終了時も通常ROSコンテナを640×480・ブリッジ無効で稼働。最終起動はD455の手動unbindなしで成功したが、過去のUSB競合が恒久解消したとは判断しない。

一時ログ・比較計測: `/tmp/lcfall-perf/`（baseline.log、shm4m.log、shm512k-repeat.log、final-result.txt）。再起動で失われ得るため主要値を本書へ記録した。

## 根拠

[Fast DDS 2.14 Shared Memory Transport](https://fast-dds.docs.eprosima.com/en/v2.14.5/fastdds/transport/shared_memory/shared_memory.html): 既定segment_size=512×1024、メッセージより小さい領域の損失リスク、同一ホストでのSHM利用とUDP Discoveryの説明。これに実機の領域サイズ、内部受信件数、容量を戻す比較結果を合わせて判断した。
