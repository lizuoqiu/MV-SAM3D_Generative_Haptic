# MV-SAM3D Environment Setup (Conda + CUDA 12.1)

This README explains how to use `setup_mvsam3d.sh` to reproduce the full MV-SAM3D environment setup that we prepared in this workspace.

## What the script does

- Installs Miniconda to `~/miniconda3` (if missing)
- Clones `MV-SAM3D` from GitHub (if missing)
- Creates the conda env `sam3d-objects` from `MV-SAM3D/environments/default.yml`
- Installs PyTorch `2.5.1+cu121` (CUDA 12.1)
- Installs all pip requirements in:
  - `MV-SAM3D/requirements.txt`
  - `MV-SAM3D/requirements.p3d.txt` (builds `pytorch3d` + `flash_attn`)
  - `MV-SAM3D/requirements.inference.txt` (installs `kaolin`, `gsplat`, `gradio`, etc.)
- Downloads a Google Drive cookies file (via `gdown`) and saves it to `~/.cache/gdown/cookies.txt`
- Downloads the Google Drive dataset folder via `gdown` into `downloads/generative_haptic_dataset/`
- Runs a quick `torch.cuda.is_available()` check

## Prerequisites

- Ubuntu Linux
- NVIDIA driver that supports CUDA 12.1 or newer
- Sufficient disk space (multiple GB; this setup downloads large CUDA packages)

## Step-by-step usage

1. Open a shell in this workspace (the folder containing `setup_mvsam3d.sh`).
2. Run the setup script:

```bash
./setup_mvsam3d.sh
```

3. Activate the environment afterwards:

```bash
source ~/miniconda3/bin/activate
conda activate sam3d-objects
```

4. Verify CUDA works:

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## CUDA architecture override (important for non-H100 GPUs)

The script defaults to `CUDA_ARCH_LIST=9.0` (H100). For a different GPU, override it when running the script:

```bash
CUDA_ARCH_LIST=8.6 ./setup_mvsam3d.sh
```

Common values:
- A100: `8.0`
- A6000 / 3090: `8.6`
- H100: `9.0`

## Known notes / conflicts

- `kaolin==0.17.0` depends on `jupyter-client<8`, which conflicts with `ipykernel>=8.8`. If you need Jupyter, tell me and I’ll align versions.
- The conda environment installs CUDA 12.1 toolchain even if your driver reports CUDA 12.8. This is OK; drivers are backward compatible.

## Google Drive cookies (gdown)

The setup script assumes you shared a cookies file on Google Drive and will download it automatically to:

```
~/.cache/gdown/cookies.txt
```

This enables `gdown` to access your Drive session and helps avoid quota errors.

Security note: this cookie file grants access to your Drive session. Keep it private.

## Google Drive folder downloads (gdown)

`gdown` is installed via `requirements.txt`, so it is available in the conda env. The setup script will also download the full folder automatically after the cookie file is in place.

Example folder download (resume-friendly):

```bash
gdown --folder --remaining-ok --continue -q -O downloads/drive_folder/ "https://drive.google.com/drive/folders/<FOLDER_ID>"
```

### If Google Drive quota errors happen

Make sure the cookie file above exists. If it expires, re-upload a fresh one and re-run the script or replace the file.

## Files created

- `setup_mvsam3d.sh`
- `MV-SAM3D/` (repo clone)
- Conda env: `sam3d-objects`

## Next steps

- Follow the MV-SAM3D README for dataset preparation and model usage.
- If you need Depth Anything 3, follow the upstream install in their repo and ask me to integrate it here.
