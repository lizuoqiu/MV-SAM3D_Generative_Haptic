# MV-SAM3D + Thermal Mapping Progress Handoff (2026-03-13)

## Scope
This document summarizes:
- completed work and validated outputs,
- open issues not yet solved,
- script inventory and current readiness,
- one-click entry points for setup and reconstruction.

## Completed Work

### 1) Environment and dependencies
- Miniconda-based environments were configured by `setup.bash`:
  - `sam3d-objects` for MV-SAM3D and thermal mapping scripts
  - `sam2d` for SAM2 mask generation
- Depth Anything 3 integration is available via `scripts/run_da3.py` and verified in `sam3d-objects`.

### 2) Model/checkpoint setup
- Hugging Face model files for `facebook/sam-3d-objects` were downloaded under:
  - `checkpoints/hf/`
- Missing/corrupted checkpoint artifacts were fixed (including `ss_generator.yaml` and checkpoint integrity).

### 3) Dataset preparation
- Dataset structure preparation script implemented and used:
  - `scripts/prepare_dataset_for_mvsam3d.py`
- Per-object `images -> rgb` linkage and `view_mapping.json` generation are supported.

### 4) SAM2 foreground masking
- Batch SAM2 masking pipeline implemented:
  - `scripts/run_sam2_batch_masks.sh`
  - `scripts/sam2_segment_images.py`
- Important format note:
  - SAM2 masks are saved as **RGBA** images.
  - Foreground mask is stored in the **alpha** channel.

### 5) MV-SAM3D reconstruction
- Batch and resume-capable reconstruction scripts implemented:
  - `scripts/run_mvsam3d_remaining_batch.sh`
- One-object reconstruction test completed successfully for:
  - `data/datasets/generative_haptic_dataset_v2/Data_Mar6/bottleddrink_Mar7/bottleddrink1`

### 6) Thermal mapping (Task 2)
- Core scripts implemented:
  - `scripts/compose_thermal_poses_from_da3.py`
  - `scripts/map_thermal_to_mesh.py`
  - `scripts/run_thermal_mapping.sh`
  - `scripts/run_task2_for_object.sh`
  - `scripts/run_task2_batch.py`
  - `scripts/package_processed_object.py`
  - `scripts/visualize_temperature_mapping.py`
  - `scripts/visualize_task2_summary.py`
- Processed output packaging implemented under:
  - `processed_dataset/<category>/<object_name>/...`

### 7) Calibration tooling
- Standard OpenCV RGB-thermal stereo calibration script implemented:
  - `scripts/calibrate_rgb_thermal_stereo.py`

## Open Issues (Identified, Not Fully Solved Yet)

### A) Thermal mapping geometric misalignment
- Thermal fit quality is currently limited by geometric alignment.
- For the tested object, thermal silhouette overlap metrics are low in many views.
- Main likely causes:
  - missing or inaccurate `T_thermal_from_rgb`,
  - potential pose convention/scale mismatch between reconstruction mesh frame and DA3 pose frame,
  - thermal distortion currently not fully integrated in mapping projection path.

### B) Calibration still needed for production-quality mapping
- Current Task-2 pipeline can run with identity transform fallback.
- Identity fallback is only for bring-up/testing and is not expected to be physically correct.
- A calibrated `T_thermal_from_rgb` should be generated and passed into Task 2 runs.

## Clarified Non-Issue
- SAM2 masks were previously suspected as all-white.
- This was a read-mode mistake in diagnostics.
- Actual masks are RGBA with valid alpha-mask foreground.

## Key Output Locations
- Reconstruction output example:
  - `visualization/bottleddrink1/sam2_masks/bottleddrink1_sam2_masks_mv_s1a30_s2e30_20260313_033446/`
- Processed dataset root:
  - `processed_dataset/`

## Script Readiness

### Stable / ready for routine usage
- `setup.bash`
- `scripts/prepare_dataset_for_mvsam3d.py`
- `scripts/run_sam2_batch_masks.sh`
- `scripts/run_mvsam3d_remaining_batch.sh`
- `scripts/run_task2_for_object.sh`
- `scripts/run_task2_batch.py`
- `scripts/package_processed_object.py`

### Verification scripts
- `scripts/visualize_temperature_mapping.py`
- `scripts/visualize_task2_summary.py`

### Calibration scripts
- `scripts/estimate_rgb_to_thermal_extrinsics.py`
  - PnP from manual correspondences.
- `scripts/calibrate_rgb_thermal_stereo.py`
  - Standard OpenCV checkerboard stereo calibration (`stereoCalibrate`).

## New One-Click Entry Scripts

### 1) Environment setup (stable dual-env)
- `scripts/oneclick_setup_dual_env.sh`
- Wraps full setup and optional DA3/model download.

### 2) Model building (SAM2 + MV-SAM3D)
- `scripts/oneclick_build_sam2_sam3d.sh`
- Performs dataset preparation, SAM2 batch mask generation, and MV-SAM3D reconstruction batch.

## Recommended Operational Path
1. Use dual-env setup for reliability (`sam2d` + `sam3d-objects`).
2. Run one-click SAM2+SAM3D reconstruction.
3. Calibrate RGB-thermal transform with checkerboard script.
4. Run Task 2 mapping with calibrated transform JSON.
5. Use verification outputs before full batch packaging.
