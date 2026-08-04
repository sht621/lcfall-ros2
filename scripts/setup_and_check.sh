#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SELF_PATH="${SCRIPT_DIR}/setup_and_check.sh"
RUN_PATH="${SCRIPT_DIR}/run.sh"
COMPOSE_FILE="${REPO_ROOT}/compose.yaml"
LOG_FILE="/tmp/lcfall-setup-and-check.log"
APP_LOG_FILE=""
CHECK_CONTAINER_STARTED=0

WIFI_INTERFACE="wlo1"
LIVOX_INTERFACE="enxf8e43b827753"
LIVOX_HOST_CIDR="192.168.1.50/24"
LIVOX_HOST_IP="192.168.1.50"
LIVOX_SENSOR_IP="192.168.1.198"
ROS_DOMAIN="43"
IMAGE_NAME="lcfall-ros2-rtx3050-image"
FAST_DDS_TEMP_PATH=""

cd "${REPO_ROOT}" || exit 1

cleanup_internal_temp() {
  if [[ -n "${FAST_DDS_TEMP_PATH}" && -f "${FAST_DDS_TEMP_PATH}" ]]; then
    rm -f "${FAST_DDS_TEMP_PATH}"
  fi
}

configure_fastdds() {
  local template_path="${REPO_ROOT}/config/fastdds/fastdds_wlo1.xml.template"
  local output_path="${REPO_ROOT}/config/fastdds/fastdds_wlo1.xml"

  if [[ ! -d "/sys/class/net/${WIFI_INTERFACE}" ]]; then
    echo "[fastdds] Interface not found: ${WIFI_INTERFACE}" >&2
    return 1
  fi

  local -a wifi_addresses=()
  mapfile -t wifi_addresses < <(
    ip -4 -o address show dev "${WIFI_INTERFACE}" scope global \
      | awk '{split($4, address, "/"); print address[1]}'
  )
  if [[ "${#wifi_addresses[@]}" -ne 1 || -z "${wifi_addresses[0]:-}" ]]; then
    echo "[fastdds] Expected exactly one IPv4 on ${WIFI_INTERFACE}; found ${#wifi_addresses[@]}." >&2
    return 1
  fi

  local wifi_ipv4="${wifi_addresses[0]}"
  if [[ "${wifi_ipv4}" == 127.* ]]; then
    echo "[fastdds] Refusing loopback address: ${wifi_ipv4}" >&2
    return 1
  fi

  if [[ -d "/sys/class/net/${LIVOX_INTERFACE}" ]]; then
    local -a livox_addresses=()
    mapfile -t livox_addresses < <(
      ip -4 -o address show dev "${LIVOX_INTERFACE}" scope global \
        | awk '{split($4, address, "/"); print address[1]}'
    )
    local livox_ipv4
    for livox_ipv4 in "${livox_addresses[@]}"; do
      if [[ "${wifi_ipv4}" == "${livox_ipv4}" ]]; then
        echo "[fastdds] Wi-Fi IPv4 matches the Livox NIC: ${wifi_ipv4}" >&2
        return 1
      fi
    done
  fi

  if [[ ! -r "${template_path}" ]]; then
    echo "[fastdds] Template not readable: ${template_path}" >&2
    return 1
  fi

  FAST_DDS_TEMP_PATH="$(mktemp "$(dirname "${output_path}")/.fastdds_wlo1.xml.XXXXXX")"
  trap cleanup_internal_temp EXIT INT TERM
  sed "s/__WLO1_IPV4__/${wifi_ipv4}/g" \
    "${template_path}" > "${FAST_DDS_TEMP_PATH}"
  chmod 0644 "${FAST_DDS_TEMP_PATH}"

  if grep -q '__WLO1_IPV4__' "${FAST_DDS_TEMP_PATH}" || \
     ! grep -q "<address>${wifi_ipv4}</address>" "${FAST_DDS_TEMP_PATH}"; then
    echo "[fastdds] Generated XML validation failed." >&2
    return 1
  fi

  if [[ -f "${output_path}" ]] && cmp -s "${FAST_DDS_TEMP_PATH}" "${output_path}"; then
    rm -f "${FAST_DDS_TEMP_PATH}"
  else
    mv -f "${FAST_DDS_TEMP_PATH}" "${output_path}"
  fi
  FAST_DDS_TEMP_PATH=""
  trap - EXIT INT TERM

  printf '[fastdds] interface=%s\n' "${WIFI_INTERFACE}"
  printf '[fastdds] ipv4=%s\n' "${wifi_ipv4}"
  printf '[fastdds] ros_domain_id=%s\n' "${ROS_DOMAIN}"
  printf '[fastdds] xml=%s\n' "${output_path}"
}

case "${1:-}" in
  __configure_fastdds)
    configure_fastdds
    exit $?
    ;;
esac

: > "${LOG_FILE}"

log() {
  printf '%s\n' "$*" >> "${LOG_FILE}"
}

# shellcheck disable=SC2317
cleanup() {
  if [[ "${CHECK_CONTAINER_STARTED}" == "1" ]]; then
    docker compose -f "${COMPOSE_FILE}" down --remove-orphans \
      >> "${LOG_FILE}" 2>&1 || true
    CHECK_CONTAINER_STARTED=0
  fi
  if [[ -n "${APP_LOG_FILE}" && -f "${APP_LOG_FILE}" ]]; then
    rm -f "${APP_LOG_FILE}"
  fi
}
trap cleanup EXIT INT TERM

not_ready() {
  local reason="$1"
  local guidance="$2"
  printf '[NG] %s\n\n' "${reason}"
  printf 'NOT READY\n'
  printf '原因: %s\n' "${reason}"
  printf '確認: %s\n' "${guidance}"
  printf '詳細: %s\n' "${LOG_FILE}"
  exit 1
}

metric_value() {
  local key="$1"
  local line="$2"
  awk -v wanted="${key}" '
    {
      for (field_number = 1; field_number <= NF; field_number++) {
        split($field_number, pair, "=")
        if (pair[1] == wanted) {
          print pair[2]
          exit
        }
      }
    }
  ' <<< "${line}"
}

livox_profile_uuid() {
  local uuid=""
  uuid="$(
    nmcli -t -f UUID,DEVICE connection show --active 2>> "${LOG_FILE}" \
      | awk -F: -v device="${LIVOX_INTERFACE}" \
          '$2 == device {print $1; exit}'
  )"
  if [[ -n "${uuid}" ]]; then
    printf '%s\n' "${uuid}"
    return 0
  fi

  local interface_mac
  interface_mac="$(cat "/sys/class/net/${LIVOX_INTERFACE}/address" 2>> "${LOG_FILE}")"
  while IFS= read -r uuid; do
    [[ -n "${uuid}" ]] || continue
    local profile_interface profile_mac
    profile_interface="$(
      nmcli -g connection.interface-name connection show uuid "${uuid}" \
        2>> "${LOG_FILE}" || true
    )"
    profile_mac="$(
      nmcli -g 802-3-ethernet.mac-address connection show uuid "${uuid}" \
        2>> "${LOG_FILE}" || true
    )"
    if [[ "${profile_interface}" == "${LIVOX_INTERFACE}" ]] || \
       [[ -n "${profile_mac}" && "${profile_mac,,}" == "${interface_mac,,}" ]]; then
      printf '%s\n' "${uuid}"
      return 0
    fi
  done < <(
    nmcli -t -f UUID,TYPE connection show 2>> "${LOG_FILE}" \
      | awk -F: '$2 == "802-3-ethernet" || $2 == "ethernet" {print $1}'
  )
  return 1
}

livox_nic_is_ready() {
  [[ -d "/sys/class/net/${LIVOX_INTERFACE}" ]] || return 1
  ip link show dev "${LIVOX_INTERFACE}" 2>> "${LOG_FILE}" \
    | head -n 1 | grep -q '<[^>]*UP[^>]*>' || return 1

  mapfile -t addresses < <(
    ip -4 -o address show dev "${LIVOX_INTERFACE}" scope global \
      | awk '{print $4}'
  )
  [[ "${#addresses[@]}" -eq 1 ]] || return 1
  [[ "${addresses[0]}" == "${LIVOX_HOST_CIDR}" ]] || return 1
  [[ -z "$(ip -4 route show default dev "${LIVOX_INTERFACE}")" ]] || return 1

  local effective_gateway effective_dns uuid never_default
  effective_gateway="$(
    nmcli -g IP4.GATEWAY device show "${LIVOX_INTERFACE}" 2>> "${LOG_FILE}" \
      | sed '/^$/d'
  )"
  effective_dns="$(
    nmcli -g IP4.DNS device show "${LIVOX_INTERFACE}" 2>> "${LOG_FILE}" \
      | sed '/^$/d'
  )"
  [[ -z "${effective_gateway}" && -z "${effective_dns}" ]] || return 1

  uuid="$(livox_profile_uuid || true)"
  [[ -n "${uuid}" ]] || return 1
  never_default="$(
    nmcli -g ipv4.never-default connection show uuid "${uuid}" \
      2>> "${LOG_FILE}" || true
  )"
  [[ "${never_default}" == "yes" ]] || return 1
  return 0
}

configure_livox_nic() {
  if ! command -v nmcli >/dev/null 2>&1; then
    return 1
  fi

  printf '[INFO] Livox専用NICの設定にsudo認証が必要です。\n'
  if ! sudo -v; then
    log "sudo authentication failed"
    return 1
  fi

  local uuid
  uuid="$(livox_profile_uuid || true)"
  if [[ -z "${uuid}" ]]; then
    # The log belongs to the invoking user; only nmcli needs sudo.
    # shellcheck disable=SC2024
    if ! sudo nmcli connection add \
      type ethernet \
      ifname "${LIVOX_INTERFACE}" \
      con-name "LCFall-Livox" \
      ipv4.method manual \
      ipv4.addresses "${LIVOX_HOST_CIDR}" \
      ipv4.gateway "" \
      ipv4.dns "" \
      ipv4.never-default yes \
      ipv6.method disabled >> "${LOG_FILE}" 2>&1; then
      return 1
    fi
    uuid="$(livox_profile_uuid || true)"
    if [[ -z "${uuid}" ]]; then
      uuid="$(
        nmcli -t -f UUID,NAME connection show 2>> "${LOG_FILE}" \
          | awk -F: '$2 == "LCFall-Livox" {print $1; exit}'
      )"
    fi
  fi
  [[ -n "${uuid}" ]] || return 1

  # shellcheck disable=SC2024
  if ! sudo nmcli connection modify uuid "${uuid}" \
    connection.interface-name "${LIVOX_INTERFACE}" \
    ipv4.method manual \
    ipv4.addresses "${LIVOX_HOST_CIDR}" \
    ipv4.gateway "" \
    ipv4.dns "" \
    ipv4.routes "" \
    ipv4.never-default yes \
    ipv6.never-default yes >> "${LOG_FILE}" 2>&1; then
    return 1
  fi
  # shellcheck disable=SC2024
  if ! sudo nmcli connection up uuid "${uuid}" ifname "${LIVOX_INTERFACE}" \
    >> "${LOG_FILE}" 2>&1; then
    return 1
  fi
  sleep 2
  livox_nic_is_ready
}

printf 'LCFall setup and check\n\n'
log "LCFall setup and check started: $(date --iso-8601=seconds)"

# Host, Docker, and GPU prerequisites.
if [[ ! -r /etc/os-release ]]; then
  not_ready "Ubuntu情報を取得できません" "/etc/os-release"
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != 24.04* ]]; then
  log "Unsupported host OS: ID=${ID:-unknown} VERSION_ID=${VERSION_ID:-unknown}"
  not_ready "Ubuntu 24.04系ではありません" "ホストOS"
fi

for command_name in docker nvidia-smi ip ping lsusb python3 nmcli; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    log "Missing command: ${command_name}"
    not_ready "必要なコマンドがありません: ${command_name}" "ホスト環境"
  fi
done
if ! docker info >> "${LOG_FILE}" 2>&1; then
  not_ready "Docker daemonを利用できません" "Dockerサービスとユーザー権限"
fi
if ! docker compose version >> "${LOG_FILE}" 2>&1; then
  not_ready "Docker Compose pluginを利用できません" "docker compose"
fi
if ! nvidia-smi -L >> "${LOG_FILE}" 2>&1; then
  not_ready "NVIDIA GPUを認識できません" "NVIDIAドライバ"
fi
if ! docker info --format '{{json .Runtimes}}' 2>> "${LOG_FILE}" \
  | grep -q 'nvidia'; then
  not_ready "NVIDIA Container Toolkitを利用できません" "NVIDIAコンテナランタイム"
fi

running_containers="$(
  docker compose -f "${COMPOSE_FILE}" ps --status running -q \
    2>> "${LOG_FILE}" || true
)"
if [[ -n "${running_containers}" ]]; then
  log "Running project containers: ${running_containers}"
  not_ready "LCFallコンテナがすでに起動しています" "既存デモを停止してから再実行"
fi
docker compose -f "${COMPOSE_FILE}" rm -f >> "${LOG_FILE}" 2>&1 || true

image_was_built=0
if ! docker image inspect "${IMAGE_NAME}" >> "${LOG_FILE}" 2>&1; then
  log "Application image is missing; starting build."
  if ! docker compose -f "${COMPOSE_FILE}" build app >> "${LOG_FILE}" 2>&1; then
    not_ready "Dockerイメージのbuildに失敗しました" "詳細ログのbuildエラー"
  fi
  image_was_built=1
fi

# Wi-Fi and Fast DDS generation.
if [[ ! -d "/sys/class/net/${WIFI_INTERFACE}" ]]; then
  not_ready "Wi-Fi NICがありません: ${WIFI_INTERFACE}" "Wi-Fiデバイス"
fi
if ! ip link show dev "${WIFI_INTERFACE}" 2>> "${LOG_FILE}" \
  | head -n 1 | grep -q '<[^>]*UP[^>]*>'; then
  not_ready "Wi-FiリンクがUPではありません" "${WIFI_INTERFACE}の接続"
fi
mapfile -t wifi_addresses < <(
  ip -4 -o address show dev "${WIFI_INTERFACE}" scope global \
    | awk '{split($4, address, "/"); print address[1]}'
)
if [[ "${#wifi_addresses[@]}" -ne 1 || -z "${wifi_addresses[0]:-}" ]]; then
  not_ready "Wi-Fi IPv4を一意に取得できません" "${WIFI_INTERFACE}のIPv4設定"
fi
wifi_ipv4="${wifi_addresses[0]}"
wifi_default_route="$(
  ip -4 route show default | grep "dev ${WIFI_INTERFACE}" | head -n 1
)"
if [[ -z "${wifi_default_route}" ]]; then
  not_ready "Wi-Fi側のdefault routeがありません" "${WIFI_INTERFACE}の接続"
fi

fastdds_xml="${REPO_ROOT}/config/fastdds/fastdds_wlo1.xml"
fastdds_before="missing"
if [[ -f "${fastdds_xml}" ]]; then
  fastdds_before="$(sha256sum "${fastdds_xml}" | awk '{print $1}')"
fi
if ! "${SELF_PATH}" __configure_fastdds >> "${LOG_FILE}" 2>&1; then
  not_ready "Fast DDS XMLを生成できません" "Wi-Fi IPv4とXML template"
fi
fastdds_after="$(sha256sum "${fastdds_xml}" | awk '{print $1}')"
fastdds_was_set=0
if [[ "${fastdds_before}" != "${fastdds_after}" ]]; then
  fastdds_was_set=1
fi
if [[ "$(grep -Fc "<address>${wifi_ipv4}</address>" "${fastdds_xml}")" -ne 1 ]] || \
   grep -Fq "<address>${LIVOX_HOST_IP}</address>" "${fastdds_xml}" || \
   ! grep -Fq '<useBuiltinTransports>false</useBuiltinTransports>' "${fastdds_xml}" || \
   ! grep -Fq '<segment_size>16777216</segment_size>' "${fastdds_xml}"; then
  not_ready "Fast DDS XMLの内容が正式構成と一致しません" "interface allowlist"
fi

compose_config="$(docker compose -f "${COMPOSE_FILE}" config 2>> "${LOG_FILE}")"
printf '%s\n' "${compose_config}" >> "${LOG_FILE}"
for expected in \
  'RMW_IMPLEMENTATION: rmw_fastrtps_cpp' \
  'ROS_DOMAIN_ID: "43"' \
  'ROS_LOCALHOST_ONLY: "0"' \
  'FASTRTPS_DEFAULT_PROFILES_FILE: /etc/fastdds/fastdds_wlo1.xml' \
  'target: /etc/fastdds/fastdds_wlo1.xml'
do
  if ! grep -Fq "${expected}" <<< "${compose_config}"; then
    log "Missing Compose setting: ${expected}"
    not_ready "ComposeのROS 2／Fast DDS設定が不正です" "compose.yaml"
  fi
done

# Files and directories required by the production launch.
directories_were_set=0
for directory in \
  data/model-cache \
  data/checkpoints/camera \
  data/checkpoints/lidar \
  data/checkpoints/fusion
do
  if [[ ! -d "${directory}" ]]; then
    if ! mkdir -p "${directory}" >> "${LOG_FILE}" 2>&1; then
      not_ready "必要ディレクトリを作成できません" "${directory}の権限"
    fi
    directories_were_set=1
  fi
done

required_files=(
  Dockerfile
  compose.yaml
  config/fastdds/fastdds_wlo1.xml.template
  config/fastdds/fastdds_wlo1.xml
  config/livox/MID360_config.json
  src/lcfall_ros2/config/params.yaml
  src/lcfall_ros2/config/slowonly_r50_inference.py
  src/lcfall_ros2/launch/lcfall.launch.py
  data/checkpoints/camera/best_model.pth
  data/checkpoints/lidar/best_model.pth
  data/checkpoints/fusion/best_model.pth
  data/model-cache/torch/hub/checkpoints/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth
  data/model-cache/torch/hub/checkpoints/td-hm_ViTPose-small-simple_8xb64-210e_coco-256x192-4c101a76_20230314.pth
)
missing_files=()
for path in "${required_files[@]}"; do
  if [[ ! -s "${path}" ]]; then
    missing_files+=("${path}")
  fi
done
if (( ${#missing_files[@]} > 0 )); then
  printf 'Missing required files:\n' >> "${LOG_FILE}"
  printf '  %s\n' "${missing_files[@]}" >> "${LOG_FILE}"
  not_ready "必要ファイルが不足しています: ${missing_files[0]}" "data/checkpointsとmodel-cache"
fi
if [[ ! -d src/lcfall_msgs || ! -d src/lcfall_ros2 ]]; then
  not_ready "ROS 2 workspaceが不足しています" "src/lcfall_msgsとsrc/lcfall_ros2"
fi
if ! grep -Fq 'rgb_camera.profile": "1280x720x30"' \
     src/lcfall_ros2/launch/lcfall.launch.py || \
   ! grep -Fq '"enable_color": True' src/lcfall_ros2/launch/lcfall.launch.py; then
  not_ready "RealSense設定が正式構成と一致しません" "lcfall.launch.py"
fi
if ! grep -Fq '"ip": "192.168.1.198"' config/livox/MID360_config.json || \
   ! grep -Fq '"point_data_ip": "192.168.1.50"' config/livox/MID360_config.json; then
  not_ready "Livox設定が正式構成と一致しません" "MID360_config.json"
fi
if ! grep -Fq '/root/ros2_ws/src/LCFall/src/lcfall_ros2/config/params.yaml' \
     docker/run_app.sh || \
   ! grep -Fq '/root/ros2_ws/src/LCFall/src/lcfall_ros2/config/params.yaml' \
     compose.yaml; then
  not_ready "本番とROI GUIの設定参照先が一致しません" "params.yamlのマウント設定"
fi
if ! PYTHONDONTWRITEBYTECODE=1 python3 - \
  src/lcfall_ros2/config/params.yaml >> "${LOG_FILE}" 2>&1 <<'PY'
import math
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
parameters = yaml.safe_load(path.read_text(encoding="utf-8"))["/**"]["ros__parameters"]
for axis in ("x", "y", "z"):
    minimum = parameters[f"roi_{axis}_min"]
    maximum = parameters[f"roi_{axis}_max"]
    if isinstance(minimum, bool) or isinstance(maximum, bool):
        raise ValueError(f"ROI {axis} is not numeric")
    minimum = float(minimum)
    maximum = float(maximum)
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
        raise ValueError(f"ROI {axis} min/max is invalid")
PY
then
  not_ready "ROI設定が不正です" "src/lcfall_ros2/config/params.yaml"
fi

# Livox dedicated NIC; modify only this connection when required.
if [[ ! -d "/sys/class/net/${LIVOX_INTERFACE}" ]]; then
  not_ready "Livox専用NICがありません" "Ethernetアダプタ"
fi
livox_nic_was_set=0
if ! livox_nic_is_ready; then
  log "Livox NIC requires configuration."
  if ! configure_livox_nic; then
    not_ready "Livox専用NICを自動設定できません" "sudo権限とNetworkManager"
  fi
  livox_nic_was_set=1
fi
if ! livox_nic_is_ready; then
  not_ready "Livox専用NIC設定が不正です" "192.168.1.50/24、gatewayなし、DNSなし"
fi
if [[ "$(ip -4 route show default | grep "dev ${WIFI_INTERFACE}" | head -n 1)" \
      != "${wifi_default_route}" ]]; then
  not_ready "Livox設定後にWi-Fi default routeが失われました" "NetworkManager設定"
fi

if ! ping -I "${LIVOX_INTERFACE}" -c 2 -W 1 "${LIVOX_SENSOR_IP}" \
  >> "${LOG_FILE}" 2>&1; then
  not_ready "MID-360へ接続できません" "電源、Ethernetケーブル、192.168.1.198、Livox NIC IP"
fi

# Container-level GPU and ROS checks.
if ! docker compose -f "${COMPOSE_FILE}" run --rm --no-deps \
  --entrypoint /bin/bash app -lc '
    set -e
    nvidia-smi -L
    test "${ROS_DISTRO}" = "humble"
    test "${RMW_IMPLEMENTATION}" = "rmw_fastrtps_cpp"
    test "${ROS_DOMAIN_ID}" = "43"
    test "${ROS_LOCALHOST_ONLY}" = "0"
    source /opt/ros/humble/setup.bash
    ros2 --help >/dev/null
  ' >> "${LOG_FILE}" 2>&1; then
  not_ready "アプリコンテナでGPU／ROS 2を利用できません" "Container ToolkitとDockerイメージ"
fi

# RealSense host and container detection.
if ! lsusb | grep -Eiq '8086:0b5c|RealSense.*455'; then
  not_ready "RealSense D455をUSBで認識できません" "USBケーブルとD455電源"
fi
mapfile -t video_devices < <(find /dev -maxdepth 1 -type c -name 'video*' -print | sort -V)
if [[ "${#video_devices[@]}" -eq 0 ]]; then
  not_ready "RealSense用videoデバイスがありません" "/dev/video*"
fi
realsense_output="$(
  "${RUN_PATH}" __compose run --rm --no-deps \
    --entrypoint /bin/bash app -lc '
      source /opt/ros/humble/setup.bash
      rs-enumerate-devices -s
    ' 2>&1
)"
realsense_rc=$?
printf '%s\n' "${realsense_output}" >> "${LOG_FILE}"
if [[ "${realsense_rc}" -ne 0 ]] || \
   ! grep -Eiq 'D455|0B5C' <<< "${realsense_output}"; then
  not_ready "コンテナからRealSense D455を認識できません" "USB権限とvideoデバイス"
fi

# Start the exact production stack without visualization for a short probe.
rx_bytes_before="$(cat "/sys/class/net/${LIVOX_INTERFACE}/statistics/rx_bytes")"
CHECK_CONTAINER_STARTED=1
if ! ENABLE_VISUALIZATION=false "${RUN_PATH}" __compose \
  up -d --no-build app >> "${LOG_FILE}" 2>&1; then
  not_ready "確認用コンテナを起動できません" "Compose起動ログ"
fi

app_running=0
for _ in $(seq 1 45); do
  container_id="$(
    docker compose -f "${COMPOSE_FILE}" ps -q app 2>> "${LOG_FILE}" || true
  )"
  if [[ -n "${container_id}" ]] && \
     [[ "$(docker inspect -f '{{.State.Running}}' "${container_id}" 2>> "${LOG_FILE}")" == "true" ]]; then
    app_running=1
    break
  fi
  sleep 1
done
if [[ "${app_running}" != "1" ]]; then
  docker compose -f "${COMPOSE_FILE}" logs --no-color app >> "${LOG_FILE}" 2>&1 || true
  not_ready "確認用コンテナが起動状態になりません" "コンテナログ"
fi

workspace_ready=0
for _ in $(seq 1 90); do
  if docker compose -f "${COMPOSE_FILE}" exec -T app \
    test -f /root/ros2_ws/install/lcfall_msgs/share/lcfall_msgs/package.xml \
    >> "${LOG_FILE}" 2>&1; then
    workspace_ready=1
    break
  fi
  sleep 1
done
if [[ "${workspace_ready}" != "1" ]]; then
  not_ready "ROS 2 workspaceの準備が完了しません" "colcon buildログ"
fi

probe_output="$(
  timeout 170 docker compose -f "${COMPOSE_FILE}" exec -T app \
    /bin/bash -c '
      source /opt/ros/humble/setup.bash
      source /root/ros2_ws/install/setup.bash
      export PYTHONDONTWRITEBYTECODE=1
      python3 -
    ' 2>> "${LOG_FILE}" <<'PY'
import bisect
from collections import deque
import statistics
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from lcfall_msgs.msg import FallDetectionResult


def stamp_seconds(message):
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


class Probe(Node):
    def __init__(self):
        super().__init__("lcfall_setup_probe")
        sensor_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.camera_arrivals = deque(maxlen=600)
        self.lidar_arrivals = deque(maxlen=300)
        self.camera_stamps = deque(maxlen=600)
        self.lidar_stamps = deque(maxlen=300)
        self.width = 0
        self.height = 0
        self.result_count = 0
        self.create_subscription(Image, "/camera/image_raw", self.camera, sensor_qos)
        self.create_subscription(PointCloud2, "/livox/lidar", self.lidar, sensor_qos)
        self.create_subscription(
            FallDetectionResult, "/fall_detection/result", self.result, 10
        )

    def camera(self, message):
        self.camera_arrivals.append(time.monotonic())
        self.camera_stamps.append(stamp_seconds(message))
        self.width = int(message.width)
        self.height = int(message.height)

    def lidar(self, message):
        self.lidar_arrivals.append(time.monotonic())
        self.lidar_stamps.append(stamp_seconds(message))

    def result(self, message):
        del message
        self.result_count += 1


def stable_rate(stamps):
    if len(stamps) < 2:
        return 0.0
    ordered_stamps = list(stamps)
    intervals = [
        current - previous
        for previous, current in zip(ordered_stamps, ordered_stamps[1:])
        if current > previous
    ]
    if not intervals:
        return 0.0
    median_interval = statistics.median(intervals)
    return 1.0 / median_interval if median_interval > 0.0 else 0.0


def sync_metrics(camera_stamps, lidar_stamps):
    cameras = sorted(camera_stamps)
    if not cameras or not lidar_stamps:
        return 999999.0, 0
    differences = []
    for lidar_stamp in lidar_stamps:
        if lidar_stamp < cameras[0] or lidar_stamp > cameras[-1]:
            continue
        index = bisect.bisect_left(cameras, lidar_stamp)
        candidates = []
        if index < len(cameras):
            candidates.append(abs(cameras[index] - lidar_stamp))
        if index > 0:
            candidates.append(abs(cameras[index - 1] - lidar_stamp))
        if candidates:
            differences.append(min(candidates) * 1000.0)
    if not differences:
        return 999999.0, 0
    return statistics.median(differences), len(differences)


rclpy.init()
node = Probe()
deadline = time.monotonic() + 150.0
while time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.2)
    if (
        len(node.camera_arrivals) >= 90
        and len(node.lidar_arrivals) >= 30
        and node.result_count >= 1
    ):
        break

sync_ms, sync_count = sync_metrics(node.camera_stamps, node.lidar_stamps)
print(
    "METRICS"
    f" camera_count={len(node.camera_arrivals)}"
    f" camera_hz={stable_rate(node.camera_stamps):.3f}"
    f" width={node.width}"
    f" height={node.height}"
    f" lidar_count={len(node.lidar_arrivals)}"
    f" lidar_hz={stable_rate(node.lidar_stamps):.3f}"
    f" sync_ms={sync_ms:.3f}"
    f" sync_count={sync_count}"
    f" result_count={node.result_count}",
    flush=True,
)
node.destroy_node()
rclpy.shutdown()
PY
)"
probe_rc=$?
printf '%s\n' "${probe_output}" >> "${LOG_FILE}"
if [[ "${probe_rc}" -ne 0 ]]; then
  not_ready "ROSトピックの短時間確認に失敗しました" "カメラ、LiDAR、ROS 2通信"
fi
metrics_line="$(grep '^METRICS ' <<< "${probe_output}" | tail -n 1)"
if [[ -z "${metrics_line}" ]]; then
  not_ready "ROSトピックの測定結果を取得できません" "詳細ログ"
fi

camera_count="$(metric_value camera_count "${metrics_line}")"
camera_hz="$(metric_value camera_hz "${metrics_line}")"
camera_width="$(metric_value width "${metrics_line}")"
camera_height="$(metric_value height "${metrics_line}")"
lidar_count="$(metric_value lidar_count "${metrics_line}")"
lidar_hz="$(metric_value lidar_hz "${metrics_line}")"
sync_ms="$(metric_value sync_ms "${metrics_line}")"
sync_count="$(metric_value sync_count "${metrics_line}")"
result_count="$(metric_value result_count "${metrics_line}")"

APP_LOG_FILE="$(mktemp /tmp/lcfall-setup-app.XXXXXX.log)"
docker compose -f "${COMPOSE_FILE}" logs --no-color app \
  > "${APP_LOG_FILE}" 2>&1 || true
cat "${APP_LOG_FILE}" >> "${LOG_FILE}"

if ! grep -Fq 'Fusion model loaded successfully' "${APP_LOG_FILE}"; then
  not_ready "実モデルのロードを確認できません" "checkpointとモデルキャッシュ"
fi
if ! grep -Fq 'Inference result:' "${APP_LOG_FILE}" || \
   [[ "${result_count:-0}" -lt 1 ]]; then
  not_ready "実推論結果を確認できません" "48フレーム入力と推論ログ"
fi
if grep -Eiq \
  'sequence size exceeds remaining buffer|deserialize error|traceback|cuda out of memory|falling back to dummy inference|using dummy inference|skeletonextractor initialization failed|datareader.*(stop|fail|error)' \
  "${APP_LOG_FILE}"; then
  not_ready "確認中にDDS／推論エラーを検出しました" "詳細ログの該当エラー"
fi

if [[ "${camera_width}" != "1280" || "${camera_height}" != "720" ]] || \
   [[ "${camera_count:-0}" -lt 90 ]]; then
  not_ready "カメラ映像が1280×720で継続しません" "RealSense設定とUSB接続"
fi
if ! awk -v value="${camera_hz}" 'BEGIN {exit !(value >= 27.0 && value <= 33.0)}'; then
  not_ready "カメラレートが不足しています: ${camera_hz} Hz" "USB 3.x接続とカメラ占有"
fi
if [[ "${lidar_count:-0}" -lt 30 ]]; then
  not_ready "LiDAR点群が継続しません" "MID-360接続とLivoxドライバ"
fi
if ! awk -v value="${lidar_hz}" 'BEGIN {exit !(value >= 8.5 && value <= 11.5)}'; then
  not_ready "LiDARレートが不正です: ${lidar_hz} Hz" "MID-360通信"
fi
if [[ "${sync_count:-0}" -lt 20 ]] || \
   ! awk -v value="${sync_ms}" 'BEGIN {exit !(value <= 50.0)}'; then
  not_ready "Sync: ${sync_ms} ms（上限50 ms）" "カメラとLiDARの時刻"
fi

rx_bytes_after="$(cat "/sys/class/net/${LIVOX_INTERFACE}/statistics/rx_bytes")"
if [[ "${rx_bytes_after}" -le "${rx_bytes_before}" ]]; then
  not_ready "Livox専用NICで受信を確認できません" "MID-360のUDP通信"
fi

if ! docker compose -f "${COMPOSE_FILE}" exec -T app /bin/bash -c '
  source /opt/ros/humble/setup.bash
  source /root/ros2_ws/install/setup.bash
  test "$(ros2 topic type /camera/image_raw)" = "sensor_msgs/msg/Image"
  test "$(ros2 topic type /livox/lidar)" = "sensor_msgs/msg/PointCloud2"
  test "$(ros2 topic type /fall_detection/result)" = "lcfall_msgs/msg/FallDetectionResult"
' >> "${LOG_FILE}" 2>&1; then
  not_ready "ROSトピック型が正式構成と一致しません" "ROSインターフェース"
fi

if [[ "${image_was_built}" == "1" ]]; then
  printf '[SET] Docker image\n'
fi
printf '[OK] GPU / Docker\n'
printf '[OK] ROS 2 Humble\n'
printf '[OK] ROS Domain: %s / Fast DDS\n' "${ROS_DOMAIN}"
printf '[OK] Wi-Fi: %s %s\n' "${WIFI_INTERFACE}" "${wifi_ipv4}"
if [[ "${fastdds_was_set}" == "1" ]]; then
  printf '[SET] Fast DDS XML: %s\n' "${wifi_ipv4}"
fi
if [[ "${livox_nic_was_set}" == "1" ]]; then
  printf '[SET] Livox NIC: %s\n' "${LIVOX_HOST_CIDR}"
else
  printf '[OK] Livox NIC: %s\n' "${LIVOX_HOST_IP}"
fi
if [[ "${directories_were_set}" == "1" ]]; then
  printf '[SET] 必要ディレクトリ\n'
fi
printf '[OK] 必要ファイル\n'
printf '[OK] Camera: 1280x720, %.1f Hz\n' "${camera_hz}"
printf '[OK] LiDAR: %.1f Hz\n' "${lidar_hz}"
printf '[OK] Sync: %.0f ms\n' "${sync_ms}"
printf '[OK] 実推論\n'

if ! docker compose -f "${COMPOSE_FILE}" down --remove-orphans \
  >> "${LOG_FILE}" 2>&1; then
  not_ready "確認用コンテナを停止できません" "docker compose down"
fi
CHECK_CONTAINER_STARTED=0

printf '\nREADY\n'
printf 'Run: ./scripts/run.sh\n'
log "LCFall setup and check completed successfully: $(date --iso-8601=seconds)"
exit 0
