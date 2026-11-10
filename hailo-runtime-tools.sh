#!/bin/bash
# Sourced only after sensor SDK/driver stages in both Dockerfiles. Never run on Mac.
install-hailort() (
    # Image build only. The release/URLs match the user's host driver/FW lock.
    # No version override, alternate mirror or PCIe kernel driver installation.
    # Never let pip resolve dependencies over apt's NumPy/OpenCV/cv_bridge stack.
    set -euo pipefail
    local -r version=4.23.0
    local -r base_url=https://dev-public.hailo.ai/2025_10/Hailo8
    local download_directory
    download_directory="$(mktemp -d /tmp/edge-hailort.XXXXXXXX)"
    trap 'rm -rf -- "$download_directory"' EXIT
    chmod 755 "$download_directory"
    local deb="$download_directory/hailort_${version}_arm64.deb"
    local wheel="$download_directory/hailort-${version}-cp312-cp312-linux_aarch64.whl"
    test "$(dpkg --print-architecture)" = arm64
    # -q ignores curlrc; TLS certificate verification remains enabled. Redirects
    # must also use HTTPS. HTTP/network errors stop the build, without fallback.
    curl -q --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
        --connect-timeout 30 --max-time 300 --output "$deb" "$base_url/$(basename "$deb")"
    curl -q --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
        --connect-timeout 30 --max-time 300 --output "$wheel" "$base_url/$(basename "$wheel")"
    # No publisher SHA256 was found in the community announcement or HTTP
    # headers. Do not invent a lock or treat the multipart ETag as a checksum.
    test "$(dpkg-deb --field "$deb" Architecture)" = arm64
    test "$(dpkg-deb --field "$deb" Package)" = hailort
    apt-get update
    # The official deb declares no Depends. Its ELF runtime requirements are
    # libc/libm/libpthread/librt/loader, libstdc++ and libgcc_s; provide via apt.
    # Never configure hailort with dpkg/apt: postinst invokes systemctl.
    apt-get install -y --no-install-recommends libc6 libstdc++6 libgcc-s1
    /usr/bin/python3 - "$deb" "$wheel" "$version" <<'PYTHON'
import email
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import zipfile

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 12):
    raise SystemExit('Hailo artifacts require the Jazzy CPython 3.12 environment')
deb, wheel, expected_version = sys.argv[1:]
name, version, _, tags = parse_wheel_filename(Path(wheel).name)
if canonicalize_name(name) != 'hailort' or not tags.intersection(sys_tags()):
    raise SystemExit('Expected a hailort wheel compatible with this ARM64 CPython')
deb_version = subprocess.check_output(
    ['dpkg-deb', '--field', deb, 'Version'], text=True).strip()
upstream_version = deb_version.split(':', 1)[-1].split('-', 1)[0]
if Version(upstream_version) != version or version != Version(expected_version):
    raise SystemExit('HailoRT deb and wheel must both match the fixed 4.23.0 release')
with zipfile.ZipFile(wheel) as archive:
    paths = [p for p in archive.namelist() if p.endswith('.dist-info/METADATA')]
    if len(paths) != 1:
        raise SystemExit('Expected exactly one wheel METADATA file')
    metadata = email.message_from_bytes(archive.read(paths[0]))
if canonicalize_name(metadata['Name']) != 'hailort' or Version(metadata['Version']) != version:
    raise SystemExit('HailoRT wheel filename and METADATA differ')
if not SpecifierSet(metadata.get('Requires-Python', '')).contains(
        '.'.join(map(str, sys.version_info[:3]))):
    raise SystemExit('HailoRT Requires-Python excludes this interpreter')
for value in metadata.get_all('Requires-Dist', []):
    requirement = Requirement(value)
    if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
        continue
    if requirement.url or requirement.extras:
        raise SystemExit(f'Review unsupported wheel dependency manually: {value}')
    try:
        installed = importlib.metadata.version(requirement.name)
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit(f'Provide an apt dependency before installing HailoRT: {value}')
    if not requirement.specifier.contains(installed):
        raise SystemExit(f'Installed dependency {requirement.name}={installed} excludes {value}')
PYTHON
    # Extract into staging without executing preinst/postinst or registering a
    # dpkg package. Copy the direct Python/C++ runtime and CLI, not service units,
    # hailort_service, GStreamer plugins or kernel drivers.
    local runtime="$download_directory/runtime"
    dpkg-deb --extract "$deb" "$runtime"
    test -f "$runtime/usr/lib/libhailort.so.$version"
    test -x "$runtime/usr/bin/hailortcli"
    install -d /usr/lib /usr/bin /usr/include /usr/lib/cmake /usr/share/doc/hailort
    install -m 644 "$runtime/usr/lib/libhailort.so.$version" /usr/lib/
    install -m 755 "$runtime/usr/bin/hailortcli" /usr/bin/
    cp -a "$runtime/usr/include/hailo" /usr/include/
    cp -a "$runtime/usr/lib/cmake/HailoRT" /usr/lib/cmake/
    install -m 644 "$runtime/usr/share/doc/hailort/copyright" /usr/share/doc/hailort/
    # Reproduce postinst's required user-space symlink, then update the loader
    # cache for the exact SONAME used by hailortcli and the Python extension.
    ln -sfn "libhailort.so.$version" /usr/lib/libhailort.so
    ldconfig
    /usr/bin/python3 -m pip install --break-system-packages --no-cache-dir \
        --no-index --no-deps "$wheel"
    apt-get clean
)

prepare-pose-model() (
    # Image build only. Official Hailo-8 RGB HEF, Model Zoo 2.18.0 for RT 4.23.x.
    # Lock supplied by the user after HEF inspection on Pi with HailoRT 4.23.0.
    set -euo pipefail
    local -r url=https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/yolov8m_pose.hef
    local -r hef_sha256=54f22e23ee5e161e3b5d8507add82035122f72fa6357b453eb8773c63c0eb4b7
    local download_directory
    test -f /models/pose_model.yaml
    download_directory="$(mktemp -d /tmp/edge-pose-model.XXXXXXXX)"
    trap 'rm -rf -- "$download_directory"' EXIT
    curl -q --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
        --connect-timeout 30 --max-time 600 --output "$download_directory/model.hef" "$url"
    test -s "$download_directory/model.hef"
    printf '%s  %s\n' "$hef_sha256" "$download_directory/model.hef" | sha256sum --strict --check
    # Do not run HEF inspection/inference or alter manifest verification status.
    install -m 644 "$download_directory/model.hef" /models/yolov8m_pose.hef
)
