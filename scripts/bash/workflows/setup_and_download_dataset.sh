#!/usr/bin/env bash
set -euo pipefail

# One-stop workflow:
# 1) setup environments (sam2d + sam3d-objects)
# 2) download and extract dataset

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

DATASET_URL="https://drive.google.com/file/d/1s682jlio6Gx1LvcRNlWs9MQEPnAiueek/view?usp=drive_link"
OUT_DIR="${ROOT_DIR}/data/datasets"
DATASET_NAME="generative_haptic_dataset_v2"

SETUP_ARGS=()

to_abs_under_root() {
  local p="$1"
  if [[ "${p}" == "~" ]]; then
    p="${HOME}"
  elif [[ "${p}" == ~/* ]]; then
    p="${HOME}/${p#~/}"
  fi
  if [[ "${p}" = /* ]]; then
    printf '%s' "${p}"
  else
    printf '%s' "${ROOT_DIR}/${p}"
  fi
}

usage() {
  cat <<USAGE
Usage:
  bash scripts/bash/workflows/setup_and_download_dataset.sh [options]

Dataset options:
  --dataset-url <url>             Google Drive file URL
  --out-dir <path>                Output base directory (default: data/datasets)
  --dataset-name <name>           Extract folder name (default: generative_haptic_dataset_v2)

Setup passthrough options:
  --sam2-model <size>
  --install-depthanything
  --download-sam3d-model
  --hf-token <token>
  --sam3d-repo-id <repo>
  --sam3d-local-dir <path>
  --da3-dir <path>

Other:
  -h, --help                      Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-url)
      DATASET_URL="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --dataset-name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --install-depthanything|--download-sam3d-model)
      SETUP_ARGS+=("$1")
      shift
      ;;
    --sam2-model|--hf-token|--sam3d-repo-id|--sam3d-local-dir|--da3-dir)
      SETUP_ARGS+=("$1" "$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

OUT_DIR="$(to_abs_under_root "${OUT_DIR}")"

echo "[workflow] Step 1/2: setup environments"
bash "${ROOT_DIR}/scripts/bash/env/setup.bash" "${SETUP_ARGS[@]}"

echo "[workflow] Step 2/2: download + extract dataset"
bash "${ROOT_DIR}/scripts/bash/data/download_and_extract_dataset.sh" "${DATASET_URL}" "${OUT_DIR}" "${DATASET_NAME}"

echo "[workflow] completed"
