#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [[ $# -lt 5 ]]; then
  cat <<'USAGE' >&2
Usage:
  bash scripts/bash/thermal/run_thermal_mapping.sh <mesh> <thermal_dir> <thermal_intrinsics> <thermal_poses_json> <output_prefix>

Example:
  bash scripts/bash/thermal/run_thermal_mapping.sh \
    outputs/cup/mesh.ply \
    data/cup/thermal \
    thermal_intrinsics.yaml \
    data/cup/thermal_poses.json \
    outputs/cup/thermal_map
USAGE
  exit 2
fi

MESH="$(realpath "$1")"
THERMAL_DIR="$(realpath "$2")"
THERMAL_INTRINSICS="$(realpath "$3")"
THERMAL_POSES="$(realpath "$4")"
OUT_PREFIX="$(realpath "$5")"

MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
SAM3D_ENV="${SAM3D_ENV:-sam3d-objects}"

# shellcheck disable=SC1090
set +u
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${SAM3D_ENV}"
set -u

for mode in avg max; do
  python "${ROOT_DIR}/scripts/python/thermal/map_thermal_to_mesh.py" \
    --mesh "${MESH}" \
    --thermal-dir "${THERMAL_DIR}" \
    --thermal-intrinsics "${THERMAL_INTRINSICS}" \
    --thermal-poses "${THERMAL_POSES}" \
    --thermal-source csv \
    --aggregation "${mode}" \
    --output-prefix "${OUT_PREFIX}"

  python "${ROOT_DIR}/scripts/python/thermal/visualize_temperature_mapping.py" \
    --mesh "${OUT_PREFIX}_thermal_${mode}.ply" \
    --temperature "${OUT_PREFIX}_temperature_${mode}.npy" \
    --output "${OUT_PREFIX}_verify_${mode}.png"
done

echo "[thermal-runner] done"
