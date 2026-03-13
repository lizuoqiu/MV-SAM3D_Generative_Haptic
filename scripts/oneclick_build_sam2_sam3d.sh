#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# One-click model construction pipeline (SAM2 + MV-SAM3D):
# 1) Prepare dataset layout (images link + view_mapping.json)
# 2) Generate SAM2 masks for all objects
# 3) Run MV-SAM3D reconstruction batch (resume-capable)
#
# Usage:
#   scripts/oneclick_build_sam2_sam3d.sh <dataset_root> [options]

if [[ $# -lt 1 ]]; then
  cat <<'USAGE' >&2
Usage:
  scripts/oneclick_build_sam2_sam3d.sh <dataset_root> [options]

Options:
  --images-subdir <name>        images dir name (default: images)
  --mask-dir <name>             mask dir name (default: sam2_masks)
  --cutout-dir <name>           cutout dir name (default: rgb_cutout)
  --stage2-weight <name>        stage2 weight source: entropy|visibility|mixed (default: entropy)
  --exclude-object <path>       object path to skip in reconstruction
  --skip-prepare                skip prepare_dataset_for_mvsam3d.py
  -h, --help                    show help
USAGE
  exit 2
fi

DATASET_ROOT="$(realpath "$1")"
shift

IMAGES_SUBDIR="images"
MASK_DIR="sam2_masks"
CUTOUT_DIR="rgb_cutout"
STAGE2_WEIGHT="entropy"
EXCLUDE_OBJECT=""
SKIP_PREPARE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --images-subdir)
      IMAGES_SUBDIR="$2"; shift 2 ;;
    --mask-dir)
      MASK_DIR="$2"; shift 2 ;;
    --cutout-dir)
      CUTOUT_DIR="$2"; shift 2 ;;
    --stage2-weight)
      STAGE2_WEIGHT="$2"; shift 2 ;;
    --exclude-object)
      EXCLUDE_OBJECT="$2"; shift 2 ;;
    --skip-prepare)
      SKIP_PREPARE=1; shift ;;
    -h|--help)
      sed -n '1,32p' "$0"
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2 ;;
  esac
done

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 1
fi

log() {
  echo "[oneclick-build] $*"
}

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  log "Step 1/3: Preparing dataset layout"
  python "${ROOT_DIR}/scripts/prepare_dataset_for_mvsam3d.py" --dataset-root "${DATASET_ROOT}"
else
  log "Step 1/3: Skipped dataset prepare"
fi

log "Step 2/3: Running SAM2 mask generation"
SAM2_REPO_DIR="${SAM2_REPO_DIR:-${ROOT_DIR}/sam2d}" \
  "${ROOT_DIR}/scripts/run_sam2_batch_masks.sh" "${DATASET_ROOT}" "${IMAGES_SUBDIR}" "${MASK_DIR}" "${CUTOUT_DIR}"

log "Step 3/3: Running MV-SAM3D reconstruction (resume mode)"
if [[ -n "${EXCLUDE_OBJECT}" ]]; then
  "${ROOT_DIR}/scripts/run_mvsam3d_remaining_batch.sh" "${DATASET_ROOT}" "${MASK_DIR}" "${STAGE2_WEIGHT}" "${EXCLUDE_OBJECT}"
else
  "${ROOT_DIR}/scripts/run_mvsam3d_remaining_batch.sh" "${DATASET_ROOT}" "${MASK_DIR}" "${STAGE2_WEIGHT}"
fi

log "Completed"
