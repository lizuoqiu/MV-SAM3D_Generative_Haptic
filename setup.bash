#!/usr/bin/env bash
set -euo pipefail

# Unified setup for this workspace:
# - Install Miniconda if needed
# - Setup MV-SAM3D env in this repo (do not reclone)
# - Setup SAM2 env and checkpoints
#
# Usage:
#   ./setup.bash
#   INSTALL_P3D=0 ./setup.bash
#   SAM2_MODEL=large ./setup.bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"

SAM3D_ENV="${SAM3D_ENV:-sam3d-objects}"
SAM2_ENV="${SAM2_ENV:-sam2d}"
SAM2_REPO_DIR="${SAM2_REPO_DIR:-$ROOT_DIR/sam2d}"

INSTALL_P3D="${INSTALL_P3D:-1}"
INSTALL_INFERENCE="${INSTALL_INFERENCE:-1}"

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
KAOLIN_WHL_INDEX="${KAOLIN_WHL_INDEX:-https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

SAM2_MODEL="${SAM2_MODEL:-small}"  # tiny|small|base_plus|large

log() {
  echo "[setup] $*"
}

install_miniconda_if_needed() {
  if [[ -x "${MINICONDA_DIR}/bin/conda" ]]; then
    log "Miniconda found at ${MINICONDA_DIR}"
    return
  fi

  if [[ -d "${MINICONDA_DIR}" && ! -x "${MINICONDA_DIR}/bin/conda" ]]; then
    log "Found partial Miniconda directory at ${MINICONDA_DIR}; removing and reinstalling"
    rm -rf "${MINICONDA_DIR}"
  fi

  log "Installing Miniconda at ${MINICONDA_DIR}"
  tmp_installer="/tmp/miniconda_installer.sh"
  curl -fsSL -o "${tmp_installer}" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "${tmp_installer}" -b -p "${MINICONDA_DIR}"
  rm -f "${tmp_installer}"
}

init_conda() {
  # shellcheck disable=SC1090
  source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
  conda config --set auto_activate_base false || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
}

safe_conda_activate() {
  set +u
  conda activate "$1"
  set -u
}

safe_conda_deactivate() {
  set +u
  conda deactivate
  set -u
}

ensure_sam2_repo() {
  if [[ -d "${SAM2_REPO_DIR}/.git" ]]; then
    log "SAM2 repo found: ${SAM2_REPO_DIR}"
    return
  fi

  log "Cloning SAM2 into ${SAM2_REPO_DIR}"
  git clone https://github.com/facebookresearch/sam2.git "${SAM2_REPO_DIR}"
}

ensure_sam3d_env() {
  if conda env list | awk '{print $1}' | grep -qx "${SAM3D_ENV}"; then
    log "Conda env ${SAM3D_ENV} already exists"
  else
    log "Creating ${SAM3D_ENV} from environments/default.yml"
    conda env create -f "${ROOT_DIR}/environments/default.yml"
  fi

  safe_conda_activate "${SAM3D_ENV}"

  log "Installing PyTorch 2.5.1+cu121 in ${SAM3D_ENV}"
  pip install --index-url "${TORCH_INDEX_URL}" \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121

  log "Installing MV-SAM3D requirements"
  pip install -r "${ROOT_DIR}/requirements.txt"

  if [[ "${INSTALL_P3D}" == "1" ]]; then
    log "Installing PyTorch3D + flash-attn requirements (INSTALL_P3D=1)"
    export TORCH_CUDA_ARCH_LIST
    export CUDA_HOME="${CONDA_PREFIX}"
    pip install -r "${ROOT_DIR}/requirements.p3d.txt"
  else
    log "Skipping requirements.p3d.txt (INSTALL_P3D=${INSTALL_P3D})"
  fi

  if [[ "${INSTALL_INFERENCE}" == "1" ]]; then
    log "Installing kaolin + inference requirements (INSTALL_INFERENCE=1)"
    pip install kaolin==0.17.0 -f "${KAOLIN_WHL_INDEX}"
    pip install -r "${ROOT_DIR}/requirements.inference.txt"
  else
    log "Skipping requirements.inference.txt (INSTALL_INFERENCE=${INSTALL_INFERENCE})"
  fi

  if [[ -x "${ROOT_DIR}/patching/hydra" ]]; then
    log "Applying local hydra patch script"
    "${ROOT_DIR}/patching/hydra" || true
  fi

  python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

  safe_conda_deactivate
}

sam2_model_cfg_and_ckpt() {
  case "${SAM2_MODEL}" in
    tiny)
      echo "configs/sam2.1/sam2.1_hiera_t.yaml sam2.1_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
      ;;
    small)
      echo "configs/sam2.1/sam2.1_hiera_s.yaml sam2.1_hiera_small.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
      ;;
    base_plus)
      echo "configs/sam2.1/sam2.1_hiera_b+.yaml sam2.1_hiera_base_plus.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
      ;;
    large)
      echo "configs/sam2.1/sam2.1_hiera_l.yaml sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
      ;;
    *)
      echo "Unsupported SAM2_MODEL=${SAM2_MODEL}. Use tiny|small|base_plus|large" >&2
      exit 2
      ;;
  esac
}

ensure_sam2_env() {
  if conda env list | awk '{print $1}' | grep -qx "${SAM2_ENV}"; then
    log "Conda env ${SAM2_ENV} already exists"
  else
    log "Creating ${SAM2_ENV} with Python 3.11"
    conda create -y -n "${SAM2_ENV}" python=3.11 pip
  fi

  safe_conda_activate "${SAM2_ENV}"

  log "Installing PyTorch 2.5.1+cu121 in ${SAM2_ENV}"
  pip install --index-url "${TORCH_INDEX_URL}" \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121

  log "Installing SAM2 package from ${SAM2_REPO_DIR}"
  SAM2_BUILD_ALLOW_ERRORS=1 pip install -e "${SAM2_REPO_DIR}"

  read -r cfg ckpt ckpt_url <<<"$(sam2_model_cfg_and_ckpt)"
  mkdir -p "${SAM2_REPO_DIR}/checkpoints"
  if [[ -f "${SAM2_REPO_DIR}/checkpoints/${ckpt}" ]]; then
    log "SAM2 checkpoint already exists: ${ckpt}"
  else
    log "Downloading SAM2 ${SAM2_MODEL} checkpoint: ${ckpt}"
    curl -L --retry 5 --retry-delay 5 -o "${SAM2_REPO_DIR}/checkpoints/${ckpt}" "${ckpt_url}"
  fi

  python - <<PY
import torch
from sam2.build_sam import build_sam2

cfg = "${cfg}"
ckpt = "${SAM2_REPO_DIR}/checkpoints/${ckpt}"
model = build_sam2(cfg, ckpt, device="cuda" if torch.cuda.is_available() else "cpu", apply_postprocessing=False)
print("sam2_model_loaded:", type(model).__name__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY

  safe_conda_deactivate
}

main() {
  log "Root directory: ${ROOT_DIR}"
  install_miniconda_if_needed
  init_conda
  ensure_sam2_repo
  ensure_sam3d_env
  ensure_sam2_env
  log "Setup complete"
}

main "$@"
