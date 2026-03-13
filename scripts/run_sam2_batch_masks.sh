#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'USAGE' >&2
Usage:
  scripts/run_sam2_batch_masks.sh <dataset_root> [images_subdir] [mask_dir_name] [cutout_dir_name]

Example:
  scripts/run_sam2_batch_masks.sh data/datasets/generative_haptic_dataset_v2 images sam2_masks rgb_cutout
USAGE
  exit 2
fi

DATASET_ROOT="$(realpath "$1")"
IMAGES_SUBDIR="${2:-images}"
MASK_DIR_NAME="${3:-sam2_masks}"
CUTOUT_DIR_NAME="${4:-rgb_cutout}"

MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
SAM2_ENV="${SAM2_ENV:-sam2d}"
SAM2_MODEL="${SAM2_MODEL:-small}"  # tiny|small|base_plus|large
SAM2_REPO_DIR="${SAM2_REPO_DIR:-$(pwd)/sam2d}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 1
fi

# shellcheck disable=SC1090
set +u
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${SAM2_ENV}"
set -u

case "${SAM2_MODEL}" in
  tiny)
    SAM2_CFG="configs/sam2.1/sam2.1_hiera_t.yaml"
    SAM2_CKPT="${SAM2_REPO_DIR}/checkpoints/sam2.1_hiera_tiny.pt"
    ;;
  small)
    SAM2_CFG="configs/sam2.1/sam2.1_hiera_s.yaml"
    SAM2_CKPT="${SAM2_REPO_DIR}/checkpoints/sam2.1_hiera_small.pt"
    ;;
  base_plus)
    SAM2_CFG="configs/sam2.1/sam2.1_hiera_b+.yaml"
    SAM2_CKPT="${SAM2_REPO_DIR}/checkpoints/sam2.1_hiera_base_plus.pt"
    ;;
  large)
    SAM2_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"
    SAM2_CKPT="${SAM2_REPO_DIR}/checkpoints/sam2.1_hiera_large.pt"
    ;;
  *)
    echo "Unsupported SAM2_MODEL=${SAM2_MODEL}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${SAM2_CKPT}" ]]; then
  echo "Missing checkpoint: ${SAM2_CKPT}" >&2
  exit 1
fi

processed=0
while IFS= read -r -d '' image_dir; do
  obj_dir="$(dirname "${image_dir}")"
  mask_dir="${obj_dir}/${MASK_DIR_NAME}"
  cutout_dir="${obj_dir}/${CUTOUT_DIR_NAME}"
  meta_json="${obj_dir}/${MASK_DIR_NAME}_metadata.json"

  echo "[sam2-batch] object=${obj_dir}"
  python scripts/sam2_segment_images.py \
    --image-dir "${image_dir}" \
    --output-mask-dir "${mask_dir}" \
    --output-cutout-dir "${cutout_dir}" \
    --sam2-checkpoint "${SAM2_CKPT}" \
    --sam2-config "${SAM2_CFG}" \
    --metadata-json "${meta_json}"

  processed=$((processed + 1))
done < <(find "${DATASET_ROOT}" \( -type d -o -type l \) -name "${IMAGES_SUBDIR}" -print0 | sort -z)

echo "[sam2-batch] processed_objects=${processed}"
