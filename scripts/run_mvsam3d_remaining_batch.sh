#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'USAGE' >&2
Usage:
  scripts/run_mvsam3d_remaining_batch.sh <dataset_root> [mask_dir_name] [stage2_weight_source] [exclude_object_path]

Example:
  scripts/run_mvsam3d_remaining_batch.sh \
    data/datasets/generative_haptic_dataset_v2/Data_Mar6 \
    sam2_masks \
    entropy \
    data/datasets/generative_haptic_dataset_v2/Data_Mar6/bottleddrink_Mar7/bottleddrink1

Notes:
  - This script runs object-by-object and writes a done marker file after each successful run.
  - Marker file name defaults to '.mvsam3d_done' and can be overridden via MARKER_NAME.
USAGE
  exit 2
fi

DATASET_ROOT="$(realpath "$1")"
MASK_DIR_NAME="${2:-sam2_masks}"
STAGE2_WEIGHT_SOURCE="${3:-entropy}"  # entropy|visibility|mixed
EXCLUDE_OBJECT_PATH="${4:-}"
EXCLUDE_OBJECT_REALPATH=""

MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
SAM3D_ENV="${SAM3D_ENV:-sam3d-objects}"
MODEL_TAG="${MODEL_TAG:-hf}"
STAGE1_STEPS="${STAGE1_STEPS:-50}"
STAGE2_STEPS="${STAGE2_STEPS:-25}"
MARKER_NAME="${MARKER_NAME:-.mvsam3d_done}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 1
fi

if [[ -n "${EXCLUDE_OBJECT_PATH}" ]]; then
  EXCLUDE_OBJECT_REALPATH="$(realpath "${EXCLUDE_OBJECT_PATH}")"
fi

# shellcheck disable=SC1090
set +u
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${SAM3D_ENV}" ]]; then
  conda activate "${SAM3D_ENV}"
fi
set -u

processed=0
skipped=0
failed=0

while IFS= read -r -d '' image_dir; do
  obj_dir="$(dirname "${image_dir}")"
  obj_dir_real="$(realpath "${obj_dir}")"
  marker_path="${obj_dir}/${MARKER_NAME}"

  [[ -d "${obj_dir}/${MASK_DIR_NAME}" ]] || continue

  if [[ -n "${EXCLUDE_OBJECT_REALPATH}" && "${obj_dir_real}" == "${EXCLUDE_OBJECT_REALPATH}" ]]; then
    echo "[mvsam3d-remaining] skip (excluded) object=${obj_dir}"
    skipped=$((skipped + 1))
    continue
  fi

  if [[ -f "${marker_path}" ]]; then
    echo "[mvsam3d-remaining] skip (already done marker) object=${obj_dir}"
    skipped=$((skipped + 1))
    continue
  fi

  echo "[mvsam3d-remaining] run object=${obj_dir}"
  if python run_inference_weighted.py \
      --input_path "${obj_dir}" \
      --mask_prompt "${MASK_DIR_NAME}" \
      --model_tag "${MODEL_TAG}" \
      --stage1_steps "${STAGE1_STEPS}" \
      --stage2_steps "${STAGE2_STEPS}" \
      --stage2_weight_source "${STAGE2_WEIGHT_SOURCE}"; then
    {
      echo "completed_at=$(date -Iseconds)"
      echo "mask_dir_name=${MASK_DIR_NAME}"
      echo "stage2_weight_source=${STAGE2_WEIGHT_SOURCE}"
      echo "model_tag=${MODEL_TAG}"
      echo "stage1_steps=${STAGE1_STEPS}"
      echo "stage2_steps=${STAGE2_STEPS}"
    } > "${marker_path}"
    processed=$((processed + 1))
  else
    echo "[mvsam3d-remaining] fail object=${obj_dir}" >&2
    failed=$((failed + 1))
  fi
done < <(find "${DATASET_ROOT}" \( -type d -o -type l \) -name "images" -print0 | sort -z)

echo "[mvsam3d-remaining] processed=${processed} skipped=${skipped} failed=${failed}"
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi
