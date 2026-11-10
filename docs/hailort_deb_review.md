# HailoRT 4.23.0 debの静的確認とコンテナ導入

2026-11-10、feature/edge-pipeline。Macで公式配布物を一時領域へダウンロードし、
ar/bsdtarでアーカイブとテキスト、objdumpでELFのdynamic sectionを読んだ。
maintainer script・ARM64バイナリ・Python拡張は実行していない。
Dockerビルド・テスト・lint・構文チェック・Pi操作も行っていない。

## 対象と確認済み内容

- [公式ARM64 deb](https://dev-public.hailo.ai/2025_10/Hailo8/hailort_4.23.0_arm64.deb):
  controlはPackage=hailort、Version=4.23.0、Architecture=arm64。Depends/Pre-Dependsの宣言はない。
- [公式cp312 ARM64 wheel](https://dev-public.hailo.ai/2025_10/Hailo8/hailort-4.23.0-cp312-cp312-linux_aarch64.whl):
  metadataはName=hailort、Version=4.23.0、Tag=cp312-cp312-linux_aarch64。
  Requires-Distはargcomplete/contextlib2/future/netaddr/netifaces/numpy<2。
  Requires-Pythonの記載はなかった。互換タグと実interpreterの照合は既存処理を維持する。
- runtime/wheelの配布元公開SHA256は未確認。HEFのユーザー提供SHA256とは別物で、捏造した値は追加しない。

## maintainer script省略の影響

| 処理 | 配布物で確認した内容 | コンテナでの扱い |
|---|---|---|
| preinst | /var/log/hailo、/run/hailoを作成 | サービス用のため今回作成しない。直接推論のログ書込先・権限は実機確認 |
| postinstのリンク | /usr/lib/libhailort.so → /usr/lib/libhailort.so.4.23.0を作成 | 相対リンクで同じ解決先を明示作成し、ldconfigも行う |
| postinstのサービス | /.dockerenvがなければsystemctl daemon-reload、対話選択後にrestart/enableまたはstop/disable | script自体を実行しない。BuildKit環境で/.dockerenvが存在することに依存しない |
| postrm | libhailortリンク等の削除、コンテナ外ならサービス停止 | dpkg登録しないため使用しない。版の変更・削除はイメージ再構築で管理 |

実エラーはpostinstのsystemd操作と整合する。ただし、失敗したPiビルド環境の
/.dockerenvやPID 1は今回照会しておらず、存在判定の実状態までは断定しない。
policy-rc.dや偽systemctlで失敗を無視する処理、dpkg状態の手修正は追加しない。

## 配置するユーザー空間ファイル

`dpkg-deb --extract`で一時展開し、次だけを配置する。

- /usr/lib/libhailort.so.4.23.0、明示作成するlibhailort.soリンク
- /usr/bin/hailortcli（実行権限を保持）
- /usr/include/hailo/、/usr/lib/cmake/HailoRT/、/usr/share/doc/hailort/copyright

deb内の/usr/local/bin/hailort_service、systemd unit、/etc/default/hailort_service、
GStreamer pluginとそのheaderは導入しない。Pythonから直接HailoRTを使う構成を維持する。
PCIeカーネルドライバは含めず、ホストdriver/FW 4.23.0は変更しない。

[公式既定値](https://github.com/hailo-ai/hailort/blob/v4.23.0/hailort/libhailort/src/hailort_defaults.cpp)の
get_vdevice_paramsはmulti_process_service=false。
現在のHailoBackendはhp.VDevice()を使用し、この値を有効化していない。
将来サービスによる複数プロセス共有を有効にする場合は、この導入方式だけでは足りない。

## ELF依存とローダー

libhailort.so.4.23.0はAArch64 ELFでSONAMEもlibhailort.so.4.23.0。
DT_NEEDEDはlibm.so.6、librt.so.1、libpthread.so.0、libstdc++.so.6、
libgcc_s.so.1、libc.so.6、ld-linux-aarch64.so.1。
aptでlibc6/libstdc++6/libgcc-s1を明示導入し、これらの基盤を満たす記述にした。
debにDependsがないため、extract後にaptが依存を解決するとは扱わない。

hailortcliとwheel内_pyhailort.cpython-312-aarch64-linux-gnu.soも
libhailort.so.4.23.0をDT_NEEDEDに持つ。CLIのRUNPATHは$ORIGINと../lib、
wheelのRUNPATHには配布元ビルド環境のパスが残っている。
そのパスへ配置したりバイナリをpatchしたりせず、/usr/libへの配置とldconfigで解決する。
pipは--no-deps --no-indexのままで、apt側NumPy/OpenCV/cv_bridgeを保持する。
ELF参照の静的確認は、実コンテナでのリンク成功・ABI適合を証明するものではない。

## キャッシュ境界と残る実機項目

HailoRT/HEF関数をbuild-tools.shからhailo-runtime-tools.shへ移し、
両DockerfileでSDK構築・driverソース取得の後にCOPYする（EdgeRMはdriver構築の後）。
この分離時はbuild-tools.sh自身も変わるため、前段キャッシュが一度無効になる。
以後、新ファイルだけの修正ではHailoより前の命令・入力が同一なら前段キャッシュを再利用できる。
通常ROS版のdriver/app colcon構築はHailoより後なので再実行対象。実際のキャッシュ利用は未確認。

Piで許可後に確認する事項:

1. 両イメージの修正後ビルド、maintainer scriptが実行されないこと、wheel導入とキャッシュ再利用。
2. hailortcliの起動とPython hailo_platformのimport、共有ライブラリの解決、非rootログ/デバイス権限。
3. サービスを起動せずにHailo-8へ直接アクセスできること、固定HEF推論、NumPy/OpenCV ABI。
4. 既存のモデル出力契約・通常ROS/EdgeRM通信・全系10Hz。今回、自動検証は開始しない。
