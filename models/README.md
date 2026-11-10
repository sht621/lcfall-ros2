# YOLOv8m-Poseの自動配置とmanifestの確認状況

両Dockerfileは共通の`prepare-pose-model()`から、Hailo-8用Model Zoo
v2.18.0の[公式HEF](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/yolov8m_pose.hef)を
ビルド中にHTTPSで取得し、`/models/yolov8m_pose.hef`へ配置します。
取得失敗・空ファイルではビルドを停止し、別版やミラーへ切り替えません。
HailoRTは4.23.0固定。公式[対応表](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/config/config.yaml)は4.23.xとv2.18.0を対応付けています。

HEFのリンクは[公式モデル表](https://github.com/hailo-ai/hailo_model_zoo/blob/dc402c7ae70149c516324ef9ddff32d27609ae06/docs/public_models/HAILO8/HAILO8_pose_estimation.rst)で確認。
2026-11-10 07:52:00 UTCのHTTPS HEADは200、Content-Length=31,609,032。
配布元SHA256は確認できず、固定checksumは設定していません。S3のmultipart ETagを
SHA256とは扱いません。ファイル本体の取得・HEF解析・実行は今回していません。

リポジトリ管理の[pose_model.yaml](pose_model.yaml)も`/models/pose_model.yaml`へ同梱します。
これは公開ソースで確認できた情報と未確定項目を記録した**未承認の設定**です。
テンプレートのコピーを検証済みと扱う変更ではありません。
`verified_for_hef: false`によりHailoBackendはデバイス設定前に起動を拒否します。
**手動配置は不要になりますが、標準設定での推論開始は未解決です。**

## 公開ソースと既存デコーダの照合

Model ZooのGit tag名は`v2.18`、解決したcommitは
`dc402c7ae70149c516324ef9ddff32d27609ae06`です。配布ディレクトリ名は`v2.18.0`。

| 項目 | 確認できた公開仕様 | 現在の対応・限界 |
|---|---|---|
| 入出力 | [モデル設定](https://github.com/hailo-ai/hailo_model_zoo/blob/dc402c7ae70149c516324ef9ddff32d27609ae06/hailo_model_zoo/cfg/networks/yolov8m_pose.yaml)はRGB 640×640×3、20/40/80ごとに64/1/51 channelの9出力 | rawデコーダの期待形状に対応するが、実HEFのVStream名・出力形式は未照合 |
| 正規化・量子化 | [ALLS](https://github.com/hailo-ai/hailo_model_zoo/blob/dc402c7ae70149c516324ef9ddff32d27609ae06/hailo_model_zoo/cfg/alls/generic/yolov8m_pose.alls)はネット内255除算、3つのconvへsigmoid、keypoint側出力にa16_w16指定 | host入力量子化、scale/zero point、dtypeは実HEF確認が必要。ネット内とhostで二重正規化しない |
| 前処理 | [base設定](https://github.com/hailo-ai/hailo_model_zoo/blob/dc402c7ae70149c516324ef9ddff32d27609ae06/hailo_model_zoo/cfg/base/yolo.yaml)のpadding_color=114 | 既存中央letterboxの丸め・補間・座標復元との厳密一致は未確認 |
| 姿勢復元 | [公式後処理](https://github.com/hailo-ai/hailo_model_zoo/blob/dc402c7ae70149c516324ef9ddff32d27609ae06/hailo_model_zoo/core/postprocessing/pose_estimation_postprocessing.py)はDFL 16 bins、stride 32/16/8、17組のxy/score、joint scoreへsigmoid | 既存rawのDFL/XY式は対応。ただし実出力のactivationと欠損意味は未照合 |
| 選択・NMS | [継承base](https://github.com/hailo-ai/hailo_model_zoo/blob/dc402c7ae70149c516324ef9ddff32d27609ae06/hailo_model_zoo/cfg/base/yolov8.yaml)はscore=0.001、IoU=0.7、max=300。公式後処理はstrict `>` とNMS後の件数制限 | 既存rawは`>=`、件数制限なし。公式と同一の後処理とは断定できない |
| HEF内後処理 | 公開ALLSにNMS追加指示はなく、Model Zoo評価ソースにCPU後処理がある | 配布HEF自体にNMS/後処理がない証拠にはしない。実HEF解析前にrawを選択しない |

## 残る作業（requires Pi verification）

1. ダウンロードされたHEFの出所・hash、HAILO8、実VStream名/形状/format/量子化、NMS有無を、後で許可されたPi段階で記録する。
2. 前処理・後処理の照合後、manifestの未確定値を埋める。`verified_for_hef`だけを変更しない。
3. 必要なら既存custom decoder APIを使う専用モジュールを実装する。
   `create_decoder(config, output_shapes, input_shape)`が返すオブジェクトの`decode(outputs)`は、
   モデル入力の画素座標でxyxy＋det_score＋COCO17(x,y,score)のfloat32 `(N,56)`を返す。
   正常0人は`(0,56)`、不整合は例外。明示的欠損だけNaN三値とし、低scoreの関節を捨てない。
4. 実HEFが組込NMSを持つ場合はその出力契約を使い、rawのDFL/NMSを重ねて適用しない。

Composeの標準起動から`./models:/models:ro`を削除したため、ホストのmodelsディレクトリは不要です。
`hef_path`・`model_manifest`は上記イメージ内パス、`PYTHONPATH=/models`は維持します。
将来の差替えは、明示的なCompose overrideでHEF/manifest/必要なcustom moduleを揃えて
マウントするか、イメージへ同梱してください。未検証の設定へ自動fallbackしません。

今回はビルド・テスト・lint・構文チェック・ROS実行・Pi操作・コミット・Pushは行っていません。
