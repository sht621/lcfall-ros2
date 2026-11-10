# HailoRT配布物の事前配置は不要

両Dockerfileは共通の`hailo-runtime-tools.sh`の`install-hailort()`から、Hailo公式配布先の4.23.0
ARM64 runtime debとCPython 3.12 ARM64 wheelをHTTPSで自動取得します。
このディレクトリはビルド入力としてCOPYせず、旧HAILORT_DEB/HAILORT_WHEELも不要です。
debは一時展開し、ユーザー空間ライブラリ・CLI等を配置します。maintainer scriptと
systemd操作は実行せず、必要なリンク作成とldconfigを補います。
取得先・確認状況は[環境準備文書](../docs/pi_environment_preparation.md)を参照してください。

ホストのPCIeドライバ・FWは4.23.0を維持します。ビルド・ABI・実機通信は未検証です。
YOLOv8m-Pose HEFもビルド時に自動取得し、manifestを同梱します。
manifestはユーザー提供の実HEF解析結果と公式ソースに静的照合済みです。
実推論・精度・10Hz性能は未検証です。[モデルの確認状況](../models/README.md)を参照してください。
