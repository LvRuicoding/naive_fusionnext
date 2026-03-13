# FusionNeXt

FusionNeXt is organized as an OpenMMLab-style project plugin for the `bevdet`
conda environment. The current implementation focuses on:

- unified 1D token serialization for camera and LiDAR features
- cross-modal attention over the unified token sequence
- a minimal 3D detection head for nuScenes-mini

## Repository Layout

The active code lives under `projects/FusionNeXt`:

```text
projects/FusionNeXt/
  configs/
    fusionnext_nuscenes_mini_3d.py
  fusionnext/
    datasets/pipelines/
      fusionnext_nuscenes.py
    models/
      detectors/fusionnext.py
      dense_heads/fusionnext_simple_head.py
tools/
  train.py
```

Key files:

- detector: `projects/FusionNeXt/fusionnext/models/detectors/fusionnext.py`
- detection head: `projects/FusionNeXt/fusionnext/models/dense_heads/fusionnext_simple_head.py`
- nuScenes/BEVDet pipeline adapters: `projects/FusionNeXt/fusionnext/datasets/pipelines/fusionnext_nuscenes.py`
- training entrypoint: `tools/train.py`

## Environment

Use the `bevdet` conda environment described in `AGENTS.md`.

Example:

```bash
conda activate bevdet
```

The code expects the BEVDet codebase at:

```text
/home/dataset-local/lr/code/BEVDet
```

and nuScenes mini info files/data under:

```text
/home/dataset-local/lr/data/nuscenes
```

## Training

Smoke test one batch:

```bash
conda run -n bevdet python /home/dataset-local/lr/code/fusionnext/tools/train.py \
  --mode smoke \
  --config /home/dataset-local/lr/code/fusionnext/projects/FusionNeXt/configs/fusionnext_nuscenes_mini_3d.py \
  --gpu-id 0
```

Start formal training:

```bash
conda run -n bevdet python /home/dataset-local/lr/code/fusionnext/tools/train.py \
  --mode train \
  --config /home/dataset-local/lr/code/fusionnext/projects/FusionNeXt/configs/fusionnext_nuscenes_mini_3d.py \
  --gpu-id 0
```

Run with fewer dataloader workers:

```bash
conda run -n bevdet python /home/dataset-local/lr/code/fusionnext/tools/train.py \
  --mode train \
  --config /home/dataset-local/lr/code/fusionnext/projects/FusionNeXt/configs/fusionnext_nuscenes_mini_3d.py \
  --gpu-id 0 \
  --cfg-options data.workers_per_gpu=0
```

## Notes

- The config uses a minimal `FusionNeXtSimple3DHead` intended for framework
  integration and sequence-model experimentation, not for final detection
  performance.
- The dataset pipeline contains local compatibility adapters for the available
  nuScenes mini info format used in this workspace.
- Training outputs are written to `work_dirs/fusionnext_nuscenes_mini_3d/`.
