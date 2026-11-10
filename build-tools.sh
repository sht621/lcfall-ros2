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
# Callers may append driver-specific CMake definitions before colcon-build.
EDGE_CMAKE_ARGS=()
