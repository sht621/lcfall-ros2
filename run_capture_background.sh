#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || echo "${SCRIPT_DIR}")"

print_config=0
for arg in "$@"; do
  case "${arg}" in
    --config)
      print_config=1
      ;;
    *)
      echo "[run_capture_background] Unknown argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

cd "${REPO_ROOT}"

if [[ "${print_config}" == "1" ]]; then
  exec docker compose -f "${REPO_ROOT}/compose.yaml" config
fi

exec docker compose run --rm background-capture
