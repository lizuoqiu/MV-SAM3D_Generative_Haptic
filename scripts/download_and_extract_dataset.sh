#!/usr/bin/env bash
set -euo pipefail

DEFAULT_URL="https://drive.google.com/file/d/1s682jlio6Gx1LvcRNlWs9MQEPnAiueek/view?usp=drive_link"
DATASET_URL="${1:-$DEFAULT_URL}"
OUT_DIR="${2:-data/datasets}"
DATASET_NAME="${3:-generative_haptic_dataset_v2}"

mkdir -p "${OUT_DIR}"
zip_path="${OUT_DIR}/${DATASET_NAME}.zip"
extract_dir="${OUT_DIR}/${DATASET_NAME}"

extract_file_id() {
  local url="$1"
  if [[ "${url}" =~ /file/d/([^/]+) ]]; then
    echo "${BASH_REMATCH[1]}"
    return
  fi
  if [[ "${url}" =~ id=([^&]+) ]]; then
    echo "${BASH_REMATCH[1]}"
    return
  fi
  echo ""
}

file_id="$(extract_file_id "${DATASET_URL}")"
if [[ -z "${file_id}" ]]; then
  echo "[dataset] Could not parse Google Drive file id from URL: ${DATASET_URL}" >&2
  exit 2
fi

if ! command -v gdown >/dev/null 2>&1; then
  echo "[dataset] gdown not found; installing with pip"
  python -m pip install --upgrade gdown
fi

if [[ -s "${zip_path}" ]]; then
  echo "[dataset] Reusing existing zip: ${zip_path}"
else
  echo "[dataset] Downloading Google Drive file id=${file_id}"
  gdown --fuzzy "${DATASET_URL}" -O "${zip_path}"
fi

if [[ ! -s "${zip_path}" ]]; then
  echo "[dataset] Download failed or empty zip: ${zip_path}" >&2
  exit 1
fi

mkdir -p "${extract_dir}"
echo "[dataset] Extracting ${zip_path} -> ${extract_dir}"
if command -v unzip >/dev/null 2>&1; then
  unzip -o "${zip_path}" -d "${extract_dir}" >/dev/null
else
  python - "${zip_path}" "${extract_dir}" <<'PY'
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(out_dir)
PY
fi

echo "[dataset] Download and extraction completed"
echo "[dataset] zip=${zip_path}"
echo "[dataset] extracted=${extract_dir}"
