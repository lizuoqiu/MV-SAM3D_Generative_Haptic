#!/usr/bin/env bash
set -euo pipefail

# Stable one-click setup:
# 1) Run dual-environment setup (sam2d + sam3d-objects)
# 2) Optionally install Depth Anything 3 into sam3d env
# 3) Optionally download SAM3D model files from Hugging Face
#
# Usage examples:
#   scripts/oneclick_setup_dual_env.sh
#   scripts/oneclick_setup_dual_env.sh --install-depthanything --download-sam3d-model --hf-token "$HF_TOKEN"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
SAM3D_ENV="${SAM3D_ENV:-sam3d-objects}"

INSTALL_DEPTHANYTHING=0
DOWNLOAD_SAM3D_MODEL=0
HF_TOKEN="${HF_TOKEN:-}"
SAM3D_REPO_ID="${SAM3D_REPO_ID:-facebook/sam-3d-objects}"
SAM3D_LOCAL_DIR="${SAM3D_LOCAL_DIR:-$ROOT_DIR/checkpoints/hf}"
DA3_DIR="${DA3_DIR:-/workspace/Depth-Anything-3}"

usage() {
  cat <<USAGE
Usage:
  scripts/oneclick_setup_dual_env.sh [options]

Options:
  --install-depthanything         Install Depth-Anything-3 into sam3d env
  --download-sam3d-model          Download Hugging Face model files for SAM3D
  --hf-token <token>              Hugging Face token (or set HF_TOKEN env)
  --sam3d-repo-id <repo>          Model repo id (default: facebook/sam-3d-objects)
  --sam3d-local-dir <path>        Local destination (default: checkpoints/hf)
  -h, --help                      Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-depthanything)
      INSTALL_DEPTHANYTHING=1
      shift
      ;;
    --download-sam3d-model)
      DOWNLOAD_SAM3D_MODEL=1
      shift
      ;;
    --hf-token)
      HF_TOKEN="$2"
      shift 2
      ;;
    --sam3d-repo-id)
      SAM3D_REPO_ID="$2"
      shift 2
      ;;
    --sam3d-local-dir)
      SAM3D_LOCAL_DIR="$2"
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

log() {
  echo "[oneclick-setup] $*"
}

safe_conda_activate() {
  # shellcheck disable=SC1090
  source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
  set +u
  conda activate "$1"
  set -u
}

log "Step 1/3: Running base dual-environment setup"
bash "${ROOT_DIR}/setup.bash"

if [[ "${INSTALL_DEPTHANYTHING}" == "1" ]]; then
  log "Step 2/3: Installing Depth-Anything-3"
  if [[ ! -d "${DA3_DIR}" ]]; then
    log "Depth-Anything-3 not found at ${DA3_DIR}; cloning"
    git clone https://github.com/DepthAnything/Depth-Anything-3.git "${DA3_DIR}"
  fi

  safe_conda_activate "${SAM3D_ENV}"
  pip install -e "${DA3_DIR}"
  python - <<'PY'
import importlib
m = importlib.import_module("depth_anything_3.api")
print("depth_anything_3_import_ok", m.__name__)
PY
fi

if [[ "${DOWNLOAD_SAM3D_MODEL}" == "1" ]]; then
  log "Step 3/3: Downloading SAM3D model from Hugging Face"
  if [[ -z "${HF_TOKEN}" ]]; then
    echo "--download-sam3d-model requires --hf-token or HF_TOKEN env" >&2
    exit 2
  fi

  mkdir -p "${SAM3D_LOCAL_DIR}"
  safe_conda_activate "${SAM3D_ENV}"

  python - <<PY
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = "${SAM3D_REPO_ID}"
local_dir = Path("${SAM3D_LOCAL_DIR}").resolve()
token = "${HF_TOKEN}"

snapshot_download(
    repo_id=repo_id,
    local_dir=str(local_dir),
    local_dir_use_symlinks=False,
    token=token,
)
print("download_complete", repo_id, local_dir)
PY
fi

log "Completed"
