#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  cat <<'USAGE' >&2
Usage:
  scripts/run_task2_for_object.sh <object_dir> <reconstruction_dir> <thermal_intrinsics> <processed_root> [rgb_to_thermal_transform_json]

Example:
  scripts/run_task2_for_object.sh \
    data/datasets/generative_haptic_dataset_v2/Data_Mar6/bottleddrink_Mar7/bottleddrink1 \
    visualization/bottleddrink1/sam2_masks/bottleddrink1_sam2_masks_mv_s1a30_s2e30_20260313_033446 \
    thermal_intrinsics.yaml \
    processed_dataset
USAGE
  exit 2
fi

OBJECT_DIR="$(realpath "$1")"
RECON_DIR="$(realpath "$2")"
THERMAL_INTRINSICS="$(realpath "$3")"
PROCESSED_ROOT="$(realpath -m "$4")"
RGB_TO_THERMAL_JSON="${5:-}"

MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
SAM3D_ENV="${SAM3D_ENV:-sam3d-objects}"

if [[ ! -d "${OBJECT_DIR}" ]]; then
  echo "Missing object dir: ${OBJECT_DIR}" >&2
  exit 1
fi
if [[ ! -d "${RECON_DIR}" ]]; then
  echo "Missing reconstruction dir: ${RECON_DIR}" >&2
  exit 1
fi
if [[ ! -f "${RECON_DIR}/result.ply" ]]; then
  echo "Missing reconstruction mesh: ${RECON_DIR}/result.ply" >&2
  exit 1
fi
if [[ ! -f "${THERMAL_INTRINSICS}" ]]; then
  echo "Missing thermal intrinsics: ${THERMAL_INTRINSICS}" >&2
  exit 1
fi
if [[ ! -f "${OBJECT_DIR}/view_mapping.json" ]]; then
  echo "Missing view_mapping.json: ${OBJECT_DIR}/view_mapping.json" >&2
  exit 1
fi

# shellcheck disable=SC1090
set +u
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${SAM3D_ENV}"
set -u

DA3_DIR="${RECON_DIR}/da3"
DA3_OUT="${DA3_DIR}/da3_output.npz"
THERMAL_POSES="${RECON_DIR}/thermal_poses.json"
THERMAL_PREFIX="${RECON_DIR}/thermal_map"
SUMMARY_IMG="${RECON_DIR}/task2_summary.png"

mkdir -p "${DA3_DIR}"

if [[ ! -f "${DA3_OUT}" ]]; then
  echo "[task2] running DA3: ${OBJECT_DIR}/images"
  python scripts/run_da3.py \
    --image_dir "${OBJECT_DIR}/images" \
    --output_dir "${DA3_DIR}" \
    --no_vis
else
  echo "[task2] reuse DA3 output: ${DA3_OUT}"
fi

echo "[task2] composing thermal poses"
if [[ -n "${RGB_TO_THERMAL_JSON}" ]]; then
  python scripts/compose_thermal_poses_from_da3.py \
    --da3-output "${DA3_OUT}" \
    --view-mapping "${OBJECT_DIR}/view_mapping.json" \
    --output-json "${THERMAL_POSES}" \
    --rgb-to-thermal-transform "$(realpath "${RGB_TO_THERMAL_JSON}")"
else
  python scripts/compose_thermal_poses_from_da3.py \
    --da3-output "${DA3_OUT}" \
    --view-mapping "${OBJECT_DIR}/view_mapping.json" \
    --output-json "${THERMAL_POSES}"
fi

echo "[task2] thermal mapping (avg/max)"
scripts/run_thermal_mapping.sh \
  "${RECON_DIR}/result.ply" \
  "${OBJECT_DIR}/thermal" \
  "${THERMAL_INTRINSICS}" \
  "${THERMAL_POSES}" \
  "${THERMAL_PREFIX}"

echo "[task2] summary visualization"
python - <<'PY' "${OBJECT_DIR}" "${THERMAL_PREFIX}" "${SUMMARY_IMG}"
import json
from pathlib import Path
import subprocess
import sys

obj = Path(sys.argv[1]).resolve()
thermal_prefix = Path(sys.argv[2]).resolve()
summary_img = Path(sys.argv[3]).resolve()

rgb_files = sorted([p for p in (obj / "rgb").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
th_files = sorted([p for p in (obj / "thermal").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
if not rgb_files or not th_files:
    raise RuntimeError("Could not find sample RGB/Thermal image for summary.")

cmd = [
    "python",
    "scripts/visualize_task2_summary.py",
    "--rgb-image", str(rgb_files[0]),
    "--thermal-image", str(th_files[0]),
    "--verify-avg", str(thermal_prefix.with_name(thermal_prefix.name + "_verify_avg.png")),
    "--verify-max", str(thermal_prefix.with_name(thermal_prefix.name + "_verify_max.png")),
    "--output", str(summary_img),
]
subprocess.run(cmd, check=True)
PY

echo "[task2] packaging to processed_dataset"
python scripts/package_processed_object.py \
  --object-dir "${OBJECT_DIR}" \
  --reconstruction-dir "${RECON_DIR}" \
  --processed-root "${PROCESSED_ROOT}" \
  --thermal-prefix "${THERMAL_PREFIX}" \
  --thermal-poses "${THERMAL_POSES}" \
  --da3-output "${DA3_OUT}" \
  --summary-image "${SUMMARY_IMG}"

echo "[task2] done object=${OBJECT_DIR}"
