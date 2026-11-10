# オフロード通信量の簡易比較（2026-11-10）

## 条件

- Pi TB-40 / Debian 13 ARM64、起動中の通常ROSコンテナ `lcfall-ros2-edge-1`。640×480 RGB8 / カメラ30fps、ATS後Pose約10Hz。既存のFast DDS SHM16MiB/参加者、コンテナshm256MiBを使用。
- 各CLIは22秒のtimeoutで実行し、おおむね20秒の受信を観測。最終のbw集計は生RGB570件、JPEG570件、Pose200件。hzとbwの開始時刻・Discovery到達時刻は完全一致しない。
- カメラ設定・撮影位置は変更せず順次測定。同一フレームの比較や実人物数の固定は行っていない。測定期間全体で人物候補数は2–5件、Poseのbw測定窓では2件177フレーム/3件23フレーム（平均2.115件）。これはモデルの検出候補数で、視覚確認した実人数ではない。
- 実トピック名は以下の表のとおり。圧縮トピックは当初存在せず、compressed_image_transportも未導入、ドライバのenable_pub_pluginsはimage_transport/rawのみだった。
- 既存OpenCV/CvBridgeで実画像からJPEGを作る一時Publisherを使用。品質は明示的に **90**（IMWRITE_JPEG_QUALITY）、出力型sensor_msgs/CompressedImage、format=`rgb8; jpeg compressed bgr8`、元画像headerを保持。プラグイン既定品質を測った結果ではない。
- 一時Publisherは150秒で正常終了し、測定後に残していない。実装・設計・本番Launch/Compose/モデルの変更、追加パッケージ導入、サーバへの生画像送信はしていない。

## 結果

1KB=1000 Byte、1MB=1000000 Byte。Byte/msgはCDRシリアライズされたROSメッセージ全体。DDS/UDP/IP等のオーバーヘッドは含まない。

| データ形式 | 平均Byte/msg | 最小～最大Byte/msg | 実測Hz | 実測KB/s | 10Hz換算KB/s | 削減率（同じ10Hzの生RGB比） |
|---|---:|---:|---:|---:|---:|---:|
| 生RGB | 921672.00 | 921672～921672 | 29.670 | 27500.46 | 9216.72 | 基準 |
| JPEG圧縮RGB（品質90） | 83982.68 | 83771～84215 | 29.749 | 2520.86 | 839.83 | 90.888% |
| 姿勢推定 | 521.76 | 496～720 | 10.002 | 5.24 | 5.22 | 99.943% |

実測帯域のMB/s表記: 生RGB27.50046、JPEG2.52086、Pose0.00523962。

| データ形式 | 実際に測定したトピック |
|---|---|
| 生RGB | `/camera/camera/color/image_raw` |
| JPEG | `/camera/camera/color/image_raw/compressed`（一時Publisher） |
| Pose | `/pose_frame` |

換算式:

- 10Hz換算KB/s = 平均Byte/msg × 10 ÷ 1000
- 共通レート削減率 = (1 − 対象の平均Byte/msg ÷ 生RGBの平均Byte/msg) × 100
- 実測KB/sはtopic bwの観測値。上の式で実測Hzを掛けて生成した値ではない。

実測帯域の単純比較ではJPEG90.833%、Pose99.981%の削減になるが、Poseは約10Hz、画像は約30Hzというレート差を含む。形式そのもののサイズ比較には表の共通10Hz換算を使う。

生画像のデータ本体は921600 bytes。ここでの921672 bytesはROSヘッダー・encoding等とCDR表現を含む測定値。Poseは実装の固定型どおり、この窓では2候補で496 bytes、3候補で720 bytesとなった。

## CLIと丸め精度

実機にインストールされた `ros2topic.verb.bw` と `hz` の実装を使った。標準bw表示は大きな帯域では全サイズをMB単位の小数2桁へ丸める（生画像はmean/min/maxとも0.92 MBとなる）。そのため一時ラッパーで、標準print_bwに加え、同じインスタンスのget_bw値をBW_EXACTというJSONで表示した。計測・サイズ取得・窓・QoS・帯域計算方法は変更していない。Poseのraw callbackには候補人数の集計も併記した。

一時ラッパーのCLI起動引数を調整する間に計測開始前のエラーが出たが、その試行は結果に含めていない。表は実際にサンプルを取得して正常集計した最後の各ログのみを使用。

## 再実行に使うコマンド

まず対象トピックを確認する。

```bash
docker compose exec -T edge /ros_entrypoint.sh ros2 topic list
```

比較用JPEGを必要とする場合は、今回の一時ファイルがPiの/tmpに残っている間、別ターミナルで次を実行できる（150秒で終了）。再起動でファイルが失われた場合はこのコマンドだけでは再実行できない。

```bash
docker compose exec -T edge /ros_entrypoint.sh python3 - \
  < /tmp/lcfall-bandwidth/jpeg_publisher.py
```

通常CLIでの測定例（timeout終了コード124は意図した時間制限）。同じ形式を3トピックに対して実行する。ウィンドウ1000は今回の全サンプル数より大きい。

```bash
docker compose exec -T edge /ros_entrypoint.sh \
  timeout -s INT 22s ros2 topic hz /camera/camera/color/image_raw --wall-time -w 1000

docker compose exec -T edge /ros_entrypoint.sh \
  timeout -s INT 22s ros2 topic bw /camera/camera/color/image_raw -w 1000
```

今回の丸め前の併記に使ったコマンド（同じCLIを呼ぶ一時ラッパー）:

```bash
docker compose exec -T edge /ros_entrypoint.sh timeout -s INT 22s \
  python3 - /camera/camera/color/image_raw \
  < /tmp/lcfall-bandwidth/bw_exact.py
```

JPEG測定ではトピック引数を `/camera/camera/color/image_raw/compressed`、Poseでは `/pose_frame` に置き換えた。JPEG/Poseではhzとbwを並行して実行した。生RGBは一時ラッパー修正に伴いhzとbwを別の短時間窓で測定している。

## 考察と範囲

- 今回のシーン・品質90ではJPEGは生RGBの約9.11%のメッセージサイズ。10Hz換算で約9.22MB/sから約0.84MB/sへ減った。
- Poseは2–3候補の測定窓で約522 bytes/件、10Hz換算約5.22KB/s。画像を含まずBBox/COCO17/score等だけを送る条件で大幅に小さい。
- 画像とPoseで送る情報の内容が異なる。JPEGサイズはシーン・品質、Poseサイズは候補人数に依存する。人物数・画質・精度の厳密な比較、JPEG圧縮処理の性能、回線上の実効帯域、オフロード往復時間は評価していない。
- 比較用Publisherの150秒中、本番最終出力1478件を受信し、初回～最終の受信レート9.993Hzを観測。今回の軽い比較負荷の下で継続出力を確認したもので、長時間性能保証ではない。
- 一時ログ・集計結果は `/tmp/lcfall-bandwidth/`（raw/jpeg/pose各hz/bwログ、summary.json、jpeg-publisher.txt）。画像ファイルは保存していない。
