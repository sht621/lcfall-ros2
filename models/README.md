# YOLOv8m-Poseの自動配置とmanifestの確認状況

両Dockerfileは共通の`prepare-pose-model()`から、Hailo-8用Model Zoo
v2.18.0の[公式HEF](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/yolov8m_pose.hef)を
ビルド中にHTTPSで取得し、`/models/yolov8m_pose.hef`へ配置します。
取得失敗・空ファイル・固定SHA256不一致ではビルドを停止し、別版やミラーへ切り替えません。
HailoRTは4.23.0固定。公式[対応表](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/config/config.yaml)は4.23.xとv2.18.0を対応付けています。

HEFのリンクは[公式モデル表](https://github.com/hailo-ai/hailo_model_zoo/blob/dc402c7ae70149c516324ef9ddff32d27609ae06/docs/public_models/HAILO8/HAILO8_pose_estimation.rst)で確認。
2026-11-10 07:52:00 UTCのHTTPS HEADは200、Content-Length=31,609,032。
今回ユーザーがPiのHailoRT 4.23.0で解析した実HEFのSHA256を固定しました。
`54f22e23ee5e161e3b5d8507add82035122f72fa6357b453eb8773c63c0eb4b7`
これは配布元公開checksumではなく、ユーザー提供の実測値です。Macでは再取得・解析・実行していません。

リポジトリ管理の[pose_model.yaml](pose_model.yaml)も`/models/pose_model.yaml`へ同梱します。
**ユーザー提供の実HEFメタデータと固定コミットの前後処理ソースへの静的照合を完了**し、
`verified_for_hef: true`としました。これはmanifestによる起動拒否を解除する設定であり、
推論の動作・精度・性能を検証済みとするものではありません。
`verification.inference_validated`と`performance_validated`はfalseです。

## 公開ソースと既存デコーダの照合

Model ZooのGit tag名は`v2.18`、解決したcommitは
`dc402c7ae70149c516324ef9ddff32d27609ae06`です。配布ディレクトリ名は`v2.18.0`。

| 項目 | 確認できた公開仕様 | 現在の対応・限界 |
|---|---|---|
| 入出力 | ユーザー実測はHAILO8、入力yolov8m_pose/input_layer1、UINT8 NHWC 640×640×3。9出力名と形状は[manifest](pose_model.yaml)に記載 | HailoBackendは出力名集合と各HWC形状をrawデコーダで照合。batch=1、出力は逆量子化FLOAT32を維持 |
| 正規化・量子化 | 実測入力scale=1/zero point=0、boxes/scores UINT8、keypoints UINT16。[ALLS](https://github.com/hailo-ai/hailo_model_zoo/blob/dc402c7ae70149c516324ef9ddff32d27609ae06/hailo_model_zoo/cfg/alls/generic/yolov8m_pose.alls)はネット内255除算、人物scoreへsigmoid | 入力はquantized UINT8の恒等スケール。出力quantized=false/FLOAT32でHailoRTが逆量子化。人物scoreへ二重sigmoidしない |
| 前処理 | [公式pose前処理](https://github.com/hailo-ai/hailo_model_zoo/blob/dc402c7ae70149c516324ef9ddff32d27609ae06/hailo_model_zoo/core/preprocessing/pose_preprocessing.py)の中央letterbox、padding=114、寸法切捨て、縮小INTER_AREA/拡大INTER_LINEAR | pose_processing.prepareを同じ処理へ修正。実リサイズ寸法とpaddingから元RGB座標へ復元。1280×720→640×360＋上下140画素 |
| 姿勢復元 | [公式後処理](https://github.com/hailo-ai/hailo_model_zoo/blob/dc402c7ae70149c516324ef9ddff32d27609ae06/hailo_model_zoo/core/postprocessing/pose_estimation_postprocessing.py)はDFL 16 bins、stride 32/16/8、17組のxy/score、joint scoreへsigmoid | max減算の安定softmaxでDFL期待値を復元。XY=(2×raw+grid)×stride、joint scoreへ安定sigmoid。低scoreを欠損化しない |
| 選択・NMS | [継承base](https://github.com/hailo-ai/hailo_model_zoo/blob/dc402c7ae70149c516324ef9ddff32d27609ae06/hailo_model_zoo/cfg/base/yolov8.yaml)と公式後処理はscore>0.001、IoU=0.7、NMS後max=300 | strict `>` とmax_detectionsを実装。[公式Cython NMS](https://github.com/hailo-ai/hailo_model_zoo/blob/dc402c7ae70149c516324ef9ddff32d27609ae06/hailo_model_zoo/core/postprocessing/cython_utils/cython_nms.pyx)の+1面積、IoU>=閾値で抑制、二段argsortと返却順へ合わせた |
| HEF内後処理 | ユーザーの実HEF解析で9 raw-head VStream、NMS出力なし | rawデコーダを選択。custom decoderはこのHEFでは不要 |

## 残る作業（requires Pi verification）

1. ビルド時のSHA256照合、実コンテナでのHailoRT 4.23.0/VStreamsとNumPy/OpenCVのABI、非rootデバイス権限を確認する。
2. 実画像で公式後処理とBBox/COCO17/score、正常0人・複数人、閾値境界、NMS同score時の結果を比較する。数式・ソースの照合は数値一致の実行検証ではない。
3. 元1280×720 RGB画素座標への復元と推論精度を確認する。`missing_joints: none`は明示的欠損sentinelなしのraw契約。低scoreや0座標を欠損にしない。別契約のHEFへ差替える場合は再照合し、明示的欠損のみNaN三値を返す。
4. Python/NumPy NMSは20²+40²+80²=最大8400候補、最悪O(K²)。NMS後の300件制限は前段の仕事量を制限しない。候補数別の時間・メモリ、Pose全体と全系のp95/p99遅延・持続10Hzを測定する。仕様外の事前top-kは追加していない。

出力はモデル入力の画素座標でxyxy＋det_score＋全COCO17(x,y,score)のfloat32 `(N,56)`。
正常0人は`(0,56)`、不整合は例外。NMS後の公式最大300件以外の人物削減・関節フィルタは行わない。

Composeの標準起動から`./models:/models:ro`を削除したため、ホストのmodelsディレクトリは不要です。
`hef_path`・`model_manifest`は上記イメージ内パス、`PYTHONPATH=/models`は維持します。
将来の差替えは、明示的なCompose overrideでHEF/manifest/必要なcustom moduleを揃えて
マウントするか、イメージへ同梱してください。未検証の設定へ自動fallbackしません。

今回はビルド・テスト・lint・構文チェック・ROS実行・Pi操作・コミット・Pushは行っていません。
