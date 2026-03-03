#!/usr/bin/env bash
set -euo pipefail

# MV-SAM3D full environment setup (CUDA 12.1 / PyTorch 2.5.1+cu121)
# Tested on Linux with NVIDIA driver >= 12.1 support.

ROOT_DIR="$(pwd)"
REPO_DIR="${ROOT_DIR}/MV-SAM3D"
MINICONDA_DIR="${HOME}/miniconda3"
ENV_NAME="sam3d-objects"
CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-9.0}"  # H100; override via env, e.g. CUDA_ARCH_LIST=8.6
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
KAOLIN_WHL_INDEX="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
GDOWN_FOLDER_URL="https://drive.google.com/drive/folders/1PXaSvmke0_vZncvR5dFJKTDJvJzRuBSY?usp=drive_link"
GDOWN_OUT_DIR="${ROOT_DIR}/downloads/generative_haptic_dataset"

log() { echo "[setup] $*"; }

# 1) System prereqs
if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  log "Installing system prerequisites (git, curl)"
  sudo apt-get update
  sudo apt-get install -y git curl
fi

# 2) Miniconda install
if [ ! -x "${MINICONDA_DIR}/bin/conda" ]; then
  log "Installing Miniconda to ${MINICONDA_DIR}"
  cd "${HOME}"
  curl -fsSLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p "${MINICONDA_DIR}"
  rm -f Miniconda3-latest-Linux-x86_64.sh
fi

# 3) Init conda for this shell
# shellcheck disable=SC1090
source "${MINICONDA_DIR}/bin/activate"

# 4) Accept Anaconda TOS (avoids non-interactive errors)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# 5) Clone repo
if [ ! -d "${REPO_DIR}/.git" ]; then
  log "Cloning MV-SAM3D"
  git clone https://github.com/devinli123/MV-SAM3D.git
else
  log "MV-SAM3D already exists at ${REPO_DIR}"
fi

# 6) Create conda env (skip if exists)
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  log "Conda env ${ENV_NAME} already exists; skipping create"
else
  log "Creating conda env ${ENV_NAME}"
  conda env create -f "${REPO_DIR}/environments/default.yml"
fi

# 7) Activate env
conda activate "${ENV_NAME}"

# 8) Install PyTorch (CUDA 12.1)
log "Installing PyTorch 2.5.1+cu121"
pip install --index-url "${TORCH_INDEX_URL}" \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121

# 9) Install core requirements
log "Installing base requirements"
pip install -r "${REPO_DIR}/requirements.txt"

# 9.1) Download Google Drive cookies (assumes file is shared)
log "Downloading Google Drive cookies for gdown"
mkdir -p "${HOME}/.cache/gdown"
gdown --fuzzy -O /tmp/gdown_cookies.txt "https://drive.google.com/file/d/1NZJgxXzCZdmZlKV5F2cBb5kd3eQNh6YN/view?usp=sharing"
mv /tmp/gdown_cookies.txt "${HOME}/.cache/gdown/cookies.txt"

# 9.2) Download Google Drive folder (resume-friendly)
log "Downloading Google Drive folder with gdown"
mkdir -p "${GDOWN_OUT_DIR}"
gdown --folder --remaining-ok --continue -O "${GDOWN_OUT_DIR}/" "${GDOWN_FOLDER_URL}"

# 10) Build PyTorch3D + FlashAttention
log "Building pytorch3d + flash_attn"
export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}"
export CUDA_HOME="${CONDA_PREFIX}"
pip install -r "${REPO_DIR}/requirements.p3d.txt"

# 11) Kaolin + inference requirements
log "Installing kaolin from NVIDIA wheel index"
pip install kaolin==0.17.0 -f "${KAOLIN_WHL_INDEX}"

log "Installing inference requirements"
pip install -r "${REPO_DIR}/requirements.inference.txt"

# 12) Quick sanity check
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('is_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
PY

log "Setup complete"
