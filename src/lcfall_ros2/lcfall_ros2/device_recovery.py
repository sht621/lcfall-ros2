"""デバイス復旧ユーティリティ."""

from __future__ import annotations

from pathlib import Path
import os
import stat


def repair_realsense_video_nodes() -> list[str]:
    """欠損した RealSense 用 /dev/video* ノードを補修する.

    sysfs 上では認識されているのに /dev 配下だけ欠けているケースに対応する。
    root 権限がない場合は作成できないので、その場合は何もせず戻る。
    """
    repaired: list[str] = []
    video_root = Path("/sys/class/video4linux")
    if not video_root.exists():
        return repaired

    for video_dir in sorted(video_root.glob("video*")):
        name_file = video_dir / "name"
        dev_file = video_dir / "dev"
        if not name_file.exists() or not dev_file.exists():
            continue

        try:
            device_name = name_file.read_text().strip()
        except OSError:
            continue

        if "Intel(R) RealSense(TM)" not in device_name:
            continue

        dev_path = Path("/dev") / video_dir.name
        if dev_path.exists():
            continue

        try:
            major_str, minor_str = dev_file.read_text().strip().split(":")
            major = int(major_str)
            minor = int(minor_str)
        except (OSError, ValueError):
            continue

        try:
            mode = stat.S_IFCHR | 0o660
            os.mknod(dev_path, mode, os.makedev(major, minor))
            os.chown(dev_path, 0, 44)  # root:video
            repaired.append(str(dev_path))
        except FileExistsError:
            continue
        except PermissionError:
            # root でない環境では補修できない。
            continue
        except OSError:
            continue

    return repaired
