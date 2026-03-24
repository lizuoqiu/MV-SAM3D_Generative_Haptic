# MV-SAM3D + Thermal Mapping Handoff (Updated: 2026-03-24)

## Purpose

This document captures the current operational state of the repository after the script reorganization and pipeline cleanup.

## What Changed in This Iteration

1. Script layout was reorganized into:
- `scripts/bash/*` for orchestration
- `scripts/python/*` for Python entrypoints

2. Legacy top-level wrappers were removed, including:
- `setup.bash`
- `setup_mvsam3d.sh`
- flat `scripts/*.py` and `scripts/*.sh` wrappers that now live under `scripts/bash` or `scripts/python`

3. New/active utility coverage includes:
- DA3 runner under `scripts/python/da3/`
- thermal pipeline tooling under `scripts/python/thermal/`
- workflow wrappers under `scripts/bash/workflows/`

## Current Stable Workflow

### A) Environment setup

```bash
bash scripts/bash/env/setup.bash
```

### B) Optional setup + dataset download

```bash
bash scripts/bash/workflows/setup_and_download_dataset.sh
```

### C) Full reconstruction flow (prepare + SAM2 + MV-SAM3D)

```bash
bash scripts/bash/pipeline/oneclick_build_sam2_sam3d.sh <dataset_root>
```

### D) Task-2 thermal batch flow

```bash
python scripts/python/thermal/run_task2_batch.py \
  --dataset-root <dataset_root> \
  --visualization-root <visualization_root> \
  --thermal-intrinsics thermal_intrinsics.yaml
```

## Current Script Inventory

### Setup and data
- `scripts/bash/env/setup.bash`
- `scripts/bash/data/download_and_extract_dataset.sh`
- `scripts/bash/workflows/setup_and_download_dataset.sh`

### Reconstruction
- `scripts/bash/pipeline/oneclick_build_sam2_sam3d.sh`
- `scripts/bash/pipeline/run_sam2_batch_masks.sh`
- `scripts/bash/pipeline/run_mvsam3d_remaining_batch.sh`
- `scripts/python/inference/run_inference.py`
- `scripts/python/inference/run_inference_weighted.py`

### Thermal / Task-2
- `scripts/bash/thermal/run_task2_for_object.sh`
- `scripts/bash/thermal/run_thermal_mapping.sh`
- `scripts/python/thermal/run_task2_batch.py`
- `scripts/python/thermal/compose_thermal_poses_from_da3.py`
- `scripts/python/thermal/map_thermal_to_mesh.py`
- `scripts/python/thermal/package_processed_object.py`
- `scripts/python/thermal/visualize_temperature_mapping.py`
- `scripts/python/thermal/visualize_task2_summary.py`
- `scripts/python/thermal/convert_colored_ply_to_glb.py`

### Data and DA3 helpers
- `scripts/python/data/prepare_dataset_for_mvsam3d.py`
- `scripts/python/data/inspect_dataset_structure.py`
- `scripts/python/da3/run_da3.py`

## Known Operational Risks

1. Thermal mapping quality still depends heavily on accurate RGB-to-thermal extrinsics.
2. Identity fallback transforms are useful for debugging only; they are not physically accurate.
3. Before large batch runs, verify one object end-to-end (reconstruction + thermal projection + packaging).

## Recommended Validation Checklist

1. Confirm environments: `sam3d-objects`, `sam2d`.
2. Verify SAM2 mask format (RGBA alpha mask) on one object.
3. Verify reconstruction artifacts (`result.ply`, `result.glb`, logs).
4. Verify thermal projection outputs (`*_thermal_avg.ply`, verification PNGs).
5. Check packaged output under `processed_dataset/<category>/<object>/`.

## Related Documentation

- [README.md](README.md)
- [README_PARAMETERS.md](README_PARAMETERS.md)
- [scripts/README.md](scripts/README.md)
- [doc/setup.md](doc/setup.md)
