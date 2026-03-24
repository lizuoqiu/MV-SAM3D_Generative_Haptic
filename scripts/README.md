# Scripts Directory Guide

This folder is organized by runtime and responsibility:
- `scripts/bash/` for orchestration and automation.
- `scripts/python/` for task-focused Python entrypoints.

The previous flat `scripts/*.py` and top-level helper scripts were migrated into this structure.

## Directory Layout

### Bash workflows
- `scripts/bash/env/`: environment setup
- `scripts/bash/data/`: dataset download and extraction
- `scripts/bash/pipeline/`: SAM2 + MV-SAM3D reconstruction pipeline
- `scripts/bash/thermal/`: Task-2 thermal mapping pipeline
- `scripts/bash/workflows/`: cross-stage one-click workflows

### Python entrypoints
- `scripts/python/inference/`: reconstruction inference entrypoints
- `scripts/python/sam2/`: SAM2 segmentation helpers
- `scripts/python/data/`: dataset preparation and validation
- `scripts/python/da3/`: Depth Anything 3 integration
- `scripts/python/thermal/`: thermal pose composition, mapping, visualization, packaging

## Recommended Entry Commands

### Environment setup

```bash
bash scripts/bash/env/setup.bash
```

### Environment setup + dataset download

```bash
bash scripts/bash/workflows/setup_and_download_dataset.sh
```

### One-click reconstruction pipeline

```bash
bash scripts/bash/pipeline/oneclick_build_sam2_sam3d.sh <dataset_root>
```

### Thermal Task-2 batch

```bash
python scripts/python/thermal/run_task2_batch.py \
  --dataset-root <dataset_root> \
  --visualization-root <visualization_root> \
  --thermal-intrinsics thermal_intrinsics.yaml
```

## Important Script Notes

### `scripts/bash/env/setup.bash`
- This is the only maintained setup entrypoint.
- Creates/updates both conda environments: `sam3d-objects` and `sam2d`.
- Supports optional DA3 installation and SAM3D model download.
- Resolves relative paths from project root (not current shell directory).

### `scripts/bash/workflows/setup_and_download_dataset.sh`
- Runs setup first, then dataset download/extraction.
- Forwards setup options such as `--sam2-model` and DA3/model flags.

### `scripts/python/thermal/convert_colored_ply_to_glb.py`
- Converts an ordered colored point cloud plus temperature array into GLB output with texture-ready attributes.

## Migration Summary (Old -> New)

- `setup.bash` -> `scripts/bash/env/setup.bash`
- `scripts/download_and_extract_dataset.sh` -> `scripts/bash/data/download_and_extract_dataset.sh`
- `scripts/oneclick_build_sam2_sam3d.sh` -> `scripts/bash/pipeline/oneclick_build_sam2_sam3d.sh`
- `scripts/run_sam2_batch_masks.sh` -> `scripts/bash/pipeline/run_sam2_batch_masks.sh`
- `scripts/run_mvsam3d_remaining_batch.sh` -> `scripts/bash/pipeline/run_mvsam3d_remaining_batch.sh`
- `scripts/run_thermal_mapping.sh` -> `scripts/bash/thermal/run_thermal_mapping.sh`
- `scripts/run_task2_for_object.sh` -> `scripts/bash/thermal/run_task2_for_object.sh`
- `scripts/run_task2_batch.py` -> `scripts/python/thermal/run_task2_batch.py`
- `run_inference.py` -> `scripts/python/inference/run_inference.py`
- `run_inference_weighted.py` -> `scripts/python/inference/run_inference_weighted.py`
- `demo.py` -> `scripts/python/inference/demo.py`

## Related Documentation

- [../README.md](../README.md)
- [../README_PARAMETERS.md](../README_PARAMETERS.md)
- [../WORK_PROGRESS_AND_HANDOFF.md](../WORK_PROGRESS_AND_HANDOFF.md)
