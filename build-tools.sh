#!/bin/bash
# Sourced only by Docker RUN instructions. Do not execute on the MacBook.
# Adapted from robot-agent-edgerm-main.zip (be22bb43865e3aea8185929a00ef8800f35bfd93).
if [[ ! "${BUILD_JOBS:-1}" =~ ^[1-9][0-9]*$ ]]; then
    echo 'BUILD_JOBS must be a positive integer' >&2
    return 1
fi

git-clone() (
    # Authentication must never be traced, including if the caller enabled xtrace.
    set +x
    set -euo pipefail
    local repository="$1" revision="$2" expected_commit="$3"
    shift 3
    case "$(pwd -P)" in
        /var/cache/edge-build/*) ;;
        *) echo 'git-clone requires the dedicated /var/cache/edge-build cache' >&2; exit 1 ;;
    esac
    if [[ ! "$repository" =~ ^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] ||
       [[ ! "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || [[ -z "$revision" ]]; then
        echo 'git-clone requires a credential-free GitHub URL, revision and full lowercase commit SHA' >&2
        exit 1
    fi
    unset GIT_TRACE GIT_TRACE_CURL GIT_CURL_VERBOSE GIT_TRACE_PACKET GIT_TRACE2 GIT_TRACE2_EVENT GIT_TRACE2_PERF
    export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0
    # An empty credential helper also applies to git processes started for submodules.
    export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=credential.helper GIT_CONFIG_VALUE_0=''
    export GIT_ASKPASS=/bin/false
    if [[ -s /run/secrets/github-access-token ]]; then
        # Dockerfile.edgerm mounts this directory as tmpfs. The script contains
        # only the secret path, never its contents; it is not written to a layer.
        [[ -d /run/edge-auth ]] || { echo 'Missing authentication tmpfs' >&2; exit 1; }
        cat > /run/edge-auth/askpass <<'ASKPASS'
#!/bin/sh
set +x
case "$1" in
  *Username*) printf '%s\n' x-access-token ;;
  *Password*) cat /run/secrets/github-access-token ;;
  *) exit 1 ;;
esac
ASKPASS
        chmod 700 /run/edge-auth/askpass
        export GIT_ASKPASS=/run/edge-auth/askpass
    fi
    if [[ ! -d .git ]]; then
        git init --quiet .
        git remote add origin "$repository"
    else
        git remote set-url origin "$repository"
        git config --unset-all remote.origin.pushurl || true
    fi
    git fetch --quiet --no-tags --depth 1 origin "$revision"
    if [[ "$(git rev-parse FETCH_HEAD)" != "$expected_commit" ]]; then
        echo 'Fetched revision differs from the locked commit; update the lock deliberately' >&2
        exit 1
    fi
    # These destructive operations are limited to the dedicated build cache by
    # the real-path guard above. Never point the helper at an application checkout.
    if git rev-parse --verify HEAD > /dev/null 2>&1; then
        git reset --hard --quiet
    fi
    git clean -ffdqx
    git checkout --quiet --detach "$expected_commit"
    git reset --hard --quiet "$expected_commit"
    git submodule sync --recursive
    if (( $# )); then
        git submodule update --init --recursive --checkout -- "$@"
    else
        git submodule update --init --recursive --checkout
    fi
    git submodule foreach --quiet --recursive 'git reset --hard --quiet; git clean -ffdqx'
    mkdir -p /opt/edge-source-locks
    {
        printf '%s\n' "$repository" "$expected_commit"
        git submodule status --recursive
    } > "/opt/edge-source-locks/$(basename "$PWD").txt"
)

colcon-build() (
    set -eo pipefail
    local base_path="$1" install_base="$2"
    shift 2
    if [[ ! "${BUILD_JOBS:-1}" =~ ^[1-9][0-9]*$ ]] || (( $# == 0 )); then
        echo 'colcon-build requires BUILD_JOBS >= 1 and explicit target packages' >&2
        exit 1
    fi
    export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS:-1}"
    export MAKEFLAGS="-j${BUILD_JOBS:-1}"
    # packages-up-to includes source dependencies; packages-select would omit
    # them when building the patched ROS underlay or our message package.
    PYTHONWARNINGS=ignore:pkg_resources:DeprecationWarning \
    colcon build --base-paths "$base_path" --merge-install \
        --install-base "$install_base" --executor sequential \
        --packages-up-to "$@" --catkin-skip-building-tests \
        --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
        "${EDGE_CMAKE_ARGS[@]}"
)
install-hailort() (
    # Image build only. Inspect supplied artifacts; never choose a runtime version
    # or let pip resolve dependencies over apt's NumPy/OpenCV/cv_bridge stack.
    set -euo pipefail
    : "${HAILORT_DEB:?Provide the real ARM64 HailoRT deb filename}"
    : "${HAILORT_WHEEL:?Provide the matching CPython 3.12 ARM64 wheel filename}"
    local deb="/tmp/hailo-artifacts/$HAILORT_DEB"
    local wheel="/tmp/hailo-artifacts/$HAILORT_WHEEL"
    test "$(dpkg --print-architecture)" = arm64
    test "$(dpkg-deb --field "$deb" Architecture)" = arm64
    test "$(dpkg-deb --field "$deb" Package)" = hailort
    apt-get update
    apt-get install -y --no-install-recommends "$deb"
    # Inspect Python dependencies after apt has resolved the deb dependencies.
    /usr/bin/python3 - "$deb" "$wheel" <<'PYTHON'
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
deb, wheel = sys.argv[1:]
name, version, _, tags = parse_wheel_filename(Path(wheel).name)
if canonicalize_name(name) != 'hailort' or not tags.intersection(sys_tags()):
    raise SystemExit('Expected a hailort wheel compatible with this ARM64 CPython')
deb_version = subprocess.check_output(
    ['dpkg-deb', '--field', deb, 'Version'], text=True).strip()
upstream_version = deb_version.split(':', 1)[-1].split('-', 1)[0]
if Version(upstream_version) != version:
    raise SystemExit('HailoRT deb and wheel release versions differ')
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
    /usr/bin/python3 -m pip install --break-system-packages --no-cache-dir \
        --no-index --no-deps "$wheel"
    apt-get clean
)

# Callers may append driver-specific CMake definitions before colcon-build.
EDGE_CMAKE_ARGS=()
