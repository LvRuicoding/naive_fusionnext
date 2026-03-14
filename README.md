# FusionNeXt

FusionNeXt is an independent OpenMMLab-style repository for camera-LiDAR
fusion experiments. The current codebase focuses on:

- unified 1D token serialization for camera and LiDAR features
- cross-modal attention over the unified token sequence
- a minimal 3D detection head for nuScenes-mini

The repo no longer depends on the external `BEVDet` codebase at runtime.

## Repository Structure

Top-level layout:

```text
configs/
  fusionnext_nuscenes_mini_3d.py
fusionnext/
  datasets/
    fusionnext_nuscenes_dataset.py
    pipelines/
      fusionnext_nuscenes.py
  models/
    backbones/
      fusionnext_backbone.py
    data_preprocessors/
      fusion_det3d_data_preprocessor.py
    dense_heads/
      fusionnext_simple_head.py
    detectors/
      fusionnext.py
    fusion_models/
      fusionnext_core.py
    layers/
      flash_window_block.py
    serialization/
      geometry_serializer.py
    tokenizers/
      image_tokenizer.py
      lidar_tokenizer.py
    utils/
      fusion_inputs.py
      geometry.py
tools/
  train.py
pyproject.toml
README.md
```

What each part is for:

- `configs/`: training config files
- `fusionnext/datasets/`: dataset definition and data pipeline
- `fusionnext/models/`: model, tokenizers, serializer, backbone, head
- `tools/train.py`: smoke test and training entrypoint
- `pyproject.toml`: editable install metadata

Important files:

- detector wrapper: `fusionnext/models/detectors/fusionnext.py`
- unified fusion core: `fusionnext/models/fusion_models/fusionnext_core.py`
- 1D token sorting: `fusionnext/models/serialization/geometry_serializer.py`
- fusion backbone: `fusionnext/models/backbones/fusionnext_backbone.py`
- image/LiDAR input preparation: `fusionnext/datasets/pipelines/fusionnext_nuscenes.py`
- dataset for local BEVDet-style nuScenes infos: `fusionnext/datasets/fusionnext_nuscenes_dataset.py`
- training entrypoint: `tools/train.py`

## Environment Setup

Recommended environment:

```bash
conda activate fusion
pip install -e .
```

The code assumes your `fusion` environment already contains the required
OpenMMLab stack, including:

- `torch`
- `torchvision`
- `mmengine`
- `mmcv`
- `mmdet`
- `mmdet3d`
- `pyquaternion`

## Data Layout

The default config reads nuScenes-mini data from:

```text
/home/dataset-local/lr/data/nuscenes
```

At runtime, the repository uses the environment variable
`FUSIONNEXT_DATA_ROOT`:

```bash
export FUSIONNEXT_DATA_ROOT=/home/dataset-local/lr/data/nuscenes
```

Expected files for the current config:

- `nuscenes_infos_train_mini_sweep.pkl`
- `nuscenes_infos_val_mini_sweep.pkl`
- image files under `samples/CAM_*`
- lidar files under `samples/LIDAR_TOP`

The current dataset loader is designed for the local BEVDet-style info format
already present in this workspace.

## How To Use

Smoke test one sample:

```bash
conda run -n fusion env FUSIONNEXT_DATA_ROOT=/home/dataset-local/lr/data/nuscenes \
python tools/train.py \
  --mode smoke \
  --config configs/fusionnext_nuscenes_mini_3d.py \
  --gpu-id 0
```

Smoke test with backward:

```bash
conda run -n fusion env FUSIONNEXT_DATA_ROOT=/home/dataset-local/lr/data/nuscenes \
python tools/train.py \
  --mode smoke \
  --config configs/fusionnext_nuscenes_mini_3d.py \
  --gpu-id 0 \
  --backward
```

Start training:

```bash
conda run -n fusion env FUSIONNEXT_DATA_ROOT=/home/dataset-local/lr/data/nuscenes \
python tools/train.py \
  --mode train \
  --config configs/fusionnext_nuscenes_mini_3d.py \
  --work-dir work_dirs/fusionnext_nuscenes_mini_3d
```

Run a short training check:

```bash
conda run -n fusion env FUSIONNEXT_DATA_ROOT=/home/dataset-local/lr/data/nuscenes \
python tools/train.py \
  --mode train \
  --config configs/fusionnext_nuscenes_mini_3d.py \
  --work-dir work_dirs/short_train_check \
  --cfg-options \
    train_cfg.max_epochs=1 \
    train_dataloader.dataset.indices=2 \
    train_dataloader.num_workers=0 \
    train_dataloader.persistent_workers=False \
    default_hooks.logger.interval=1
```

Override config fields from CLI:

```bash
conda run -n fusion env FUSIONNEXT_DATA_ROOT=/home/dataset-local/lr/data/nuscenes \
python tools/train.py \
  --mode train \
  --config configs/fusionnext_nuscenes_mini_3d.py \
  --cfg-options train_dataloader.num_workers=0 optim_wrapper.optimizer.lr=1e-4
```

## Visualization

The repo provides two visualization scripts:

- `tools/visualize_sorted_tokens.py`: single-view serialization and projection plots
- `tools/visualize_serialized_frame.py`: whole-frame overview with BEV and all 6 cameras

Visualize 2D serialization order on the active main view:

```bash
conda run -n fusion env FUSIONNEXT_DATA_ROOT=/home/dataset-local/lr/data/nuscenes \
python tools/visualize_sorted_tokens.py \
  --sample-index 0 \
  --vis-mode 2d_serialization
```

Visualize 3D serialization order in BEV / voxel space:

```bash
conda run -n fusion env FUSIONNEXT_DATA_ROOT=/home/dataset-local/lr/data/nuscenes \
python tools/visualize_sorted_tokens.py \
  --sample-index 0 \
  --vis-mode 3d_serialization
```

Visualize how LiDAR tokens project onto the selected main view:

```bash
conda run -n fusion env FUSIONNEXT_DATA_ROOT=/home/dataset-local/lr/data/nuscenes \
python tools/visualize_sorted_tokens.py \
  --sample-index 0 \
  --vis-mode lidar_projection_2d
```

Enhanced joint-order visualization on the main view:

```bash
conda run -n fusion env FUSIONNEXT_DATA_ROOT=/home/dataset-local/lr/data/nuscenes \
python tools/visualize_sorted_tokens.py \
  --sample-index 0 \
  --vis-mode lidar_projection_2d_enhanced \
  --vis-patch-size 12 \
  --vis-voxel-size 0.8 0.8 1.6 \
  --label-count 35
```

Whole-frame serialization overview:

```bash
conda run -n fusion env FUSIONNEXT_DATA_ROOT=/home/dataset-local/lr/data/nuscenes \
python tools/visualize_serialized_frame.py \
  --sample-index 0 \
  --vis-mode 2d_serialization
```

Notes:

- `2d_serialization`: image-space serialization order on one view
- `3d_serialization`: BEV / voxel-space serialization order
- `lidar_projection_2d`: projected LiDAR tokens on the main view
- `lidar_projection_2d_enhanced`: sparse joint-order plot for projected LiDAR tokens and main-view image tokens
- `--vis-patch-size` and `--vis-voxel-size` are visualization-only overrides and do not change the training config
- outputs are written to `work_dirs/visualizations/` unless `--output` is specified

## Training Entry Script

`tools/train.py` supports two modes:

- `--mode smoke`: build dataset/model and run one training forward
- `--mode train`: build `mmengine.Runner` and start training

Useful arguments:

- `--config`: config path, defaults to `configs/fusionnext_nuscenes_mini_3d.py`
- `--work-dir`: override output directory
- `--gpu-id`: CUDA device for smoke mode
- `--cfg-options`: override config keys
- `--num-samples`: number of samples used by smoke mode
- `--sample-index`: starting dataset index for smoke mode
- `--backward`: run backward in smoke mode

## Current Config

The default config:

- uses `FusionNuScenesDataset`
- uses `FusionDet3DDataPreprocessor`
- trains `FusionNeXt` with `FusionNeXtSimple3DHead`
- is configured for nuScenes-mini
- disables validation and evaluator by default

Training outputs are written to `work_dirs/`.

## Debugging

For step-by-step debugging, use `--mode smoke --num-samples 1`.

The most useful places to set breakpoints are:

- `tools/train.py`: sample loading and model call
- `fusionnext/datasets/pipelines/fusionnext_nuscenes.py`: image/annotation pipeline
- `fusionnext/models/fusion_models/fusionnext_core.py`: token generation
- `fusionnext/models/serialization/geometry_serializer.py`: 1D token sorting
- `fusionnext/models/backbones/fusionnext_backbone.py`: sort/recover flow

## Notes

- `FusionNeXtSimple3DHead` is intended for integration and debugging, not final detection performance.
- The current pipeline is tuned for the local nuScenes-mini info files in this workspace.
- Validation/evaluation is not wired up in the default config yet.
