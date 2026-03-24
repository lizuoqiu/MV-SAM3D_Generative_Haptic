# MV-SAM3D Generative Haptic Pipeline

This repository extends **SAM 3D Objects** to support a practical multi-stage workflow:
- multi-view 3D reconstruction with weighted fusion,
- SAM2 mask generation,
- optional Depth Anything 3 (DA3) geometry integration,
- thermal-to-mesh mapping and export for the Generative Haptic dataset.

The focus is not only model inference, but also end-to-end dataset processing.

## What This Project Does

Given an object folder with multi-view RGB images (and optionally thermal frames), this project can:
1. generate object masks with SAM2,
2. reconstruct the 3D object with MV-SAM3D,
3. align thermal frames to the reconstructed mesh,
4. package outputs into a deployable dataset layout.

## Pipeline Overview

1. **Environment setup**
- Build/update `sam3d-objects` and `sam2d` conda environments.

2. **Dataset preparation**
- Normalize dataset structure and generate `view_mapping.json` when needed.

3. **SAM2 masking**
- Batch-generate RGBA masks (`alpha` channel is foreground).

4. **MV-SAM3D reconstruction**
- Run weighted or average multi-view reconstruction.

5. **Task-2 thermal mapping (optional)**
- Compose thermal camera poses, project temperatures to mesh, render verification outputs, and package processed data.

## Quick Start

### 1) Setup environments

```bash
bash scripts/bash/env/setup.bash
```

Optional setup features:

```bash
bash scripts/bash/env/setup.bash --sam2-model large
bash scripts/bash/env/setup.bash --install-depthanything
bash scripts/bash/env/setup.bash --download-sam3d-model --hf-token "$HF_TOKEN"
```

### 2) Setup + dataset download in one command

```bash
bash scripts/bash/workflows/setup_and_download_dataset.sh
```

### 3) One-click reconstruction pipeline (prepare + SAM2 + MV-SAM3D)

```bash
bash scripts/bash/pipeline/oneclick_build_sam2_sam3d.sh <dataset_root>
```

### 4) Direct weighted inference for one object/folder

```bash
python scripts/python/inference/run_inference_weighted.py \
  --input_path ./data/example \
  --mask_prompt stuffed_toy \
  --image_names 0,1,2,3,4,5,6,7
```

### 5) Task-2 thermal batch processing

```bash
python scripts/python/thermal/run_task2_batch.py \
  --dataset-root <dataset_root> \
  --visualization-root <visualization_root> \
  --thermal-intrinsics thermal_intrinsics.yaml
```

## Input Data Layout

Typical object directory (for reconstruction + thermal mapping):

```text
<object_dir>/
├── images/                 # RGB images for reconstruction
├── thermal/                # Thermal images
├── rgb/                    # RGB frames aligned with thermal sequence
└── view_mapping.json       # RGB-to-thermal frame mapping
```

For weighted inference with masks, the input can also be:

```text
<input_path>/
├── images/
└── <mask_prompt>/          # RGBA masks, alpha channel is foreground
```

## Key Output Locations

- Reconstruction results: `visualization/<object_name>/<mask_name>/<run_id>/`
- Processed thermal dataset: `processed_dataset/<category>/<object_name>/`
- Task-2 batch report: `processed_dataset/task2_batch_report.json`

## Repository Structure

- `scripts/bash/`: orchestration scripts (setup, data, pipeline, thermal, workflows)
- `scripts/python/`: Python entry scripts grouped by domain (`inference`, `sam2`, `data`, `da3`, `thermal`)
- `sam3d_objects/`: core MV-SAM3D implementation
- `sam2d/`: third-party dependency used by this project
- `third_party/Depth-Anything-3/`: third-party dependency used by this project

## Documentation Map

### Core project docs
- [README.md](README.md)
- [README_PARAMETERS.md](README_PARAMETERS.md)
- [WORK_PROGRESS_AND_HANDOFF.md](WORK_PROGRESS_AND_HANDOFF.md)
- [doc/setup.md](doc/setup.md)

### Script index
- [scripts/README.md](scripts/README.md)

### Third-party components used
- [SAM2](https://github.com/facebookresearch/sam2)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)

## Acknowledgments

This project builds on:
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)
- [SAM2](https://github.com/facebookresearch/sam2)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)

## License

This project follows the [LICENSE](LICENSE) in this repository.
