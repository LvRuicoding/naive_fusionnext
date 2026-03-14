import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "fusionnext_nuscenes_mini_3d.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train import ensure_pythonpath, load_cfg

IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize geometry-aware token serialization.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Config file path")
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index")
    parser.add_argument(
        "--vis-mode",
        choices=["2d_serialization", "3d_serialization", "lidar_projection_2d", "lidar_projection_2d_enhanced"],
        required=True,
        help="Visualization mode for 2D/3D serialization, or lidar projection onto the main view.",
    )
    parser.add_argument(
        "--dataset-split",
        choices=["train"],
        default="train",
        help="Dataset split to visualize",
    )
    parser.add_argument(
        "--view-index",
        type=int,
        default=None,
        help="Image view index to draw for 2d serialization. Defaults to the selected main view.",
    )
    parser.add_argument(
        "--main-view",
        type=int,
        default=1,
        help="Main view index used by the 2d serializer in eval mode.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint to load before visualization",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output png path",
    )
    parser.add_argument(
        "--vis-patch-size",
        type=int,
        default=None,
        help="Visualization-only override for image patch size",
    )
    parser.add_argument(
        "--vis-voxel-size",
        nargs=3,
        type=float,
        default=None,
        metavar=("VX", "VY", "VZ"),
        help="Visualization-only override for lidar voxel size",
    )
    parser.add_argument(
        "--label-count",
        type=int,
        default=40,
        help="Approximate number of order labels to draw in enhanced numbering mode",
    )
    return parser.parse_args()


def build_dataset(cfg, split):
    from mmengine.registry import init_default_scope
    from mmdet3d.registry import DATASETS

    if cfg.get("default_scope"):
        init_default_scope(cfg.default_scope)
    if split != "train":
        raise ValueError(f"Unsupported split: {split}")
    return DATASETS.build(cfg.train_dataloader.dataset)


def build_model(cfg, checkpoint, device):
    from mmengine.registry import init_default_scope
    from mmengine.runner.checkpoint import load_checkpoint
    from mmdet3d.registry import MODELS

    if cfg.get("default_scope"):
        init_default_scope(cfg.default_scope)

    model = MODELS.build(cfg.model)
    model.init_weights()
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location="cpu")
    model = model.to(device)
    model.eval()
    return model


def prepare_batch(dataset, model, sample_index):
    from mmengine.dataset import pseudo_collate

    batch = pseudo_collate([dataset[sample_index]])
    with torch.no_grad():
        return model.data_preprocessor(batch, training=False)


def denormalize_image(img_tensor):
    image = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * IMG_STD + IMG_MEAN
    image = np.clip(image / 255.0, 0.0, 1.0)
    return image


def get_rank_map(sorted_indices, sorted_padding_mask):
    inverse_indices = torch.argsort(sorted_indices, dim=1)
    sorted_valid_mask = ~sorted_padding_mask
    original_valid_mask = torch.gather(sorted_valid_mask, 1, inverse_indices)
    valid_count = int(sorted_valid_mask[0].sum().item())
    if valid_count <= 1:
        rank_map = torch.zeros_like(inverse_indices, dtype=torch.float32)
    else:
        rank_map = inverse_indices.to(dtype=torch.float32) / float(valid_count - 1)
    rank_map = rank_map.masked_fill(~original_valid_mask, float("nan"))
    return rank_map


def collect_serialization(model, processed, vis_mode, main_view, vis_patch_size=None, vis_voxel_size=None):
    from fusionnext.models.utils import prepare_fusion_inputs

    inputs = processed["inputs"]
    data_samples = processed["data_samples"]
    img_metas = [sample.metainfo for sample in data_samples]

    with torch.no_grad():
        if vis_patch_size is not None:
            model.core.image_branch.patch_size = vis_patch_size
        if vis_voxel_size is not None:
            model.core.lidar_branch.voxel_size.data = torch.tensor(
                vis_voxel_size,
                dtype=model.core.lidar_branch.voxel_size.dtype,
                device=model.core.lidar_branch.voxel_size.device,
            )
        img, points, intrins, cam_to_lidar, post_rots, post_trans = prepare_fusion_inputs(
            inputs["points"],
            inputs["img_inputs"],
            img_metas,
        )
        core = model.core
        img_tokens, img_coords, img_depths = core.image_branch(img)
        lidar_tokens, lidar_coords, lidar_padding_mask = core.lidar_branch(points)
        img_kuvd = torch.cat([img_coords, img_depths], dim=-1)
        img_padding_mask = torch.zeros((img_tokens.shape[0], img_tokens.shape[1]), dtype=torch.bool, device=img_tokens.device)

        serializer = core.fusion_backbone.serializer
        serializer.default_main_view = main_view
        mode = "2d" if vis_mode in ("2d_serialization", "lidar_projection_2d", "lidar_projection_2d_enhanced") else "3d"
        serializer_out = serializer(
            lidar_tokens,
            lidar_coords,
            img_tokens,
            img_kuvd,
            intrins,
            cam_to_lidar,
            post_rots=post_rots,
            post_trans=post_trans,
            mode=mode,
            lidar_padding_mask=lidar_padding_mask,
            img_padding_mask=img_padding_mask,
            return_debug=True,
        )
        sorted_tokens, sorted_indices, sorted_padding_mask, _, _, debug = serializer_out
        del sorted_tokens

    return {
        "img": img,
        "img_coords": img_coords,
        "lidar_coords": lidar_coords,
        "lidar_padding_mask": lidar_padding_mask,
        "img_padding_mask": img_padding_mask,
        "sorted_indices": sorted_indices,
        "sorted_padding_mask": sorted_padding_mask,
        "debug": debug,
    }


def render_3d_serialization(data, output_path, sample_index):
    batch_index = 0
    num_lidar = int(data["lidar_coords"].shape[1])
    rank_map = get_rank_map(data["sorted_indices"], data["sorted_padding_mask"])[batch_index].detach().cpu()
    lidar_valid_mask = (~data["lidar_padding_mask"][batch_index]).detach().cpu()
    lidar_rank = rank_map[:num_lidar][lidar_valid_mask].numpy()
    lidar_xy = data["lidar_coords"][batch_index, lidar_valid_mask, :2].detach().cpu().numpy()

    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    scatter = ax.scatter(
        lidar_xy[:, 0],
        lidar_xy[:, 1],
        c=lidar_rank,
        cmap="viridis",
        norm=norm,
        s=8,
        linewidths=0,
    )
    ax.set_title(f"3D Serialization Order, sample={sample_index}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.95)
    colorbar.set_label("Normalized Rank")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_2d_serialization(data, output_path, sample_index, view_index):
    batch_index = 0
    num_lidar = int(data["lidar_coords"].shape[1])
    rank_map = get_rank_map(data["sorted_indices"], data["sorted_padding_mask"])[batch_index].detach().cpu()
    img_coords_batch = data["img_coords"][batch_index].detach().cpu()
    img_valid_mask = (~data["img_padding_mask"][batch_index]).detach().cpu()
    view_mask = (img_coords_batch[:, 0].long() == view_index) & img_valid_mask
    view_img_indices = torch.nonzero(view_mask, as_tuple=False).squeeze(1)
    view_img_abs_indices = view_img_indices + num_lidar
    image_rank = rank_map[view_img_abs_indices].numpy()
    image_uv = img_coords_batch[view_img_indices, 1:3].numpy()
    image = denormalize_image(data["img"][batch_index, view_index])

    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.imshow(image)
    scatter = ax.scatter(
        image_uv[:, 0],
        image_uv[:, 1],
        c=image_rank,
        cmap="viridis",
        norm=norm,
        s=18,
        marker="s",
        linewidths=0,
        alpha=0.85,
    )
    ax.set_title(f"2D Serialization Order, sample={sample_index}, view={view_index}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.95)
    colorbar.set_label("Normalized Rank")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_lidar_projection_2d(data, output_path, sample_index, view_index):
    batch_index = 0
    image = denormalize_image(data["img"][batch_index, view_index])
    projected = data["debug"]["projected_lidar"][batch_index].detach().cpu()
    lidar_valid_mask = (~data["lidar_padding_mask"][batch_index]).detach().cpu()
    sorted_indices = data["sorted_indices"][batch_index].detach().cpu()
    sorted_padding_mask = data["sorted_padding_mask"][batch_index].detach().cpu()
    num_lidar = int(data["lidar_coords"].shape[1])

    width = image.shape[1]
    height = image.shape[0]
    uv = projected[:, 1:3]
    visible_mask = (
        lidar_valid_mask
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    visible_uv = uv[visible_mask].numpy()
    lidar_inverse_rank = torch.argsort(sorted_indices, dim=0)[:num_lidar].to(dtype=torch.float32)
    valid_lidar_count = int(lidar_valid_mask.sum().item())
    if valid_lidar_count <= 1:
        lidar_rank = torch.zeros((num_lidar,), dtype=torch.float32)
    else:
        lidar_rank = lidar_inverse_rank / float(valid_lidar_count - 1)
    visible_rank = lidar_rank[visible_mask].numpy()

    sorted_lidar_positions = []
    for sorted_pos, token_idx in enumerate(sorted_indices.tolist()):
        if sorted_padding_mask[sorted_pos]:
            continue
        if token_idx >= num_lidar:
            continue
        if not visible_mask[token_idx]:
            continue
        sorted_lidar_positions.append((sorted_pos, uv[token_idx].numpy()))
    sorted_lidar_positions.sort(key=lambda item: item[0])
    path_points = np.array([point for _, point in sorted_lidar_positions], dtype=np.float32)
    if path_points.shape[0] > 400:
        step = max(path_points.shape[0] // 400, 1)
        path_points = path_points[::step]

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.imshow(image)
    if path_points.shape[0] >= 2:
        ax.plot(path_points[:, 0], path_points[:, 1], color="white", linewidth=0.7, alpha=0.35)
    scatter = ax.scatter(
        visible_uv[:, 0],
        visible_uv[:, 1],
        c=visible_rank,
        cmap="viridis",
        norm=colors.Normalize(vmin=0.0, vmax=1.0),
        s=10,
        linewidths=0,
        alpha=0.9,
    )
    ax.set_title(f"Lidar Projection Order on Main View, sample={sample_index}, view={view_index}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.95)
    colorbar.set_label("Lidar Token Order")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compute_main_block_local_ranks(data, view_index):
    batch_index = 0
    num_lidar = int(data["lidar_coords"].shape[1])
    sorted_indices = data["sorted_indices"][batch_index].detach().cpu()
    sorted_padding_mask = data["sorted_padding_mask"][batch_index].detach().cpu()
    img_coords_batch = data["img_coords"][batch_index].detach().cpu()
    img_valid_mask = (~data["img_padding_mask"][batch_index]).detach().cpu()
    main_view_mask = (img_coords_batch[:, 0].long() == view_index) & img_valid_mask
    main_view_img_rel = torch.nonzero(main_view_mask, as_tuple=False).squeeze(1)
    main_view_img_abs = set((main_view_img_rel + num_lidar).tolist())

    main_block_order = []
    for token_idx, is_pad in zip(sorted_indices.tolist(), sorted_padding_mask.tolist()):
        if is_pad:
            continue
        if token_idx < num_lidar or token_idx in main_view_img_abs:
            main_block_order.append(token_idx)

    local_rank_map = {}
    denom = max(len(main_block_order) - 1, 1)
    for order_idx, token_idx in enumerate(main_block_order):
        local_rank_map[token_idx] = order_idx / denom
    return local_rank_map


def get_view_patch_grid(img_coords_batch, view_index):
    view_mask = img_coords_batch[:, 0].long() == view_index
    view_coords = img_coords_batch[view_mask, 1:3]
    unique_u = torch.unique(view_coords[:, 0]).sort()[0]
    unique_v = torch.unique(view_coords[:, 1]).sort()[0]
    if unique_u.numel() < 2 or unique_v.numel() < 2:
        raise ValueError("Need at least two patch centers per axis to infer visualization grid.")
    step_u = float((unique_u[1] - unique_u[0]).item())
    step_v = float((unique_v[1] - unique_v[0]).item())
    u0 = float(unique_u[0].item())
    v0 = float(unique_v[0].item())
    return u0, v0, step_u, step_v


def render_lidar_projection_2d_enhanced(data, output_path, sample_index, view_index, label_count=40):
    batch_index = 0
    image = denormalize_image(data["img"][batch_index, view_index])
    projected = data["debug"]["projected_lidar"][batch_index].detach().cpu()
    img_coords_batch = data["img_coords"][batch_index].detach().cpu()
    lidar_valid_mask = (~data["lidar_padding_mask"][batch_index]).detach().cpu()
    img_valid_mask = (~data["img_padding_mask"][batch_index]).detach().cpu()
    num_lidar = int(data["lidar_coords"].shape[1])
    width = image.shape[1]
    height = image.shape[0]
    u0, v0, step_u, step_v = get_view_patch_grid(img_coords_batch, view_index)

    view_img_mask = (img_coords_batch[:, 0].long() == view_index) & img_valid_mask
    view_img_rel = torch.nonzero(view_img_mask, as_tuple=False).squeeze(1)
    view_img_abs = view_img_rel + num_lidar
    view_img_abs_set = set(view_img_abs.tolist())

    uv = projected[:, 1:3]
    visible_lidar_mask = (
        lidar_valid_mask
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    sorted_indices = data["sorted_indices"][batch_index].detach().cpu()
    sorted_padding_mask = data["sorted_padding_mask"][batch_index].detach().cpu()
    main_block_entries = []
    for token_idx, is_pad in zip(sorted_indices.tolist(), sorted_padding_mask.tolist()):
        if is_pad:
            continue
        if token_idx < num_lidar:
            if not visible_lidar_mask[token_idx]:
                continue
            x = float(uv[token_idx, 0].item())
            y = float(uv[token_idx, 1].item())
            col = int(round((x - u0) / step_u))
            row = int(round((y - v0) / step_v))
            point = np.array([u0 + col * step_u, v0 + row * step_v], dtype=np.float32)
            main_block_entries.append(("lidar", token_idx, point))
        elif token_idx in view_img_abs_set:
            rel_idx = token_idx - num_lidar
            point = img_coords_batch[rel_idx, 1:3].numpy().astype(np.float32)
            main_block_entries.append(("image", token_idx, point))

    if main_block_entries:
        dedup_entries = [(0, main_block_entries[0][0], main_block_entries[0][1], main_block_entries[0][2])]
        for order_idx, (modality, token_idx, point) in enumerate(main_block_entries[1:], start=1):
            if not np.allclose(point, dedup_entries[-1][3]) or modality != dedup_entries[-1][1]:
                dedup_entries.append((order_idx, modality, token_idx, point))
    else:
        dedup_entries = []

    if len(dedup_entries) > 400:
        step = max(len(dedup_entries) // 400, 1)
        dedup_entries = dedup_entries[::step]

    path_points = np.array([point for _, _, _, point in dedup_entries], dtype=np.float32) if dedup_entries else np.zeros((0, 2), dtype=np.float32)
    if label_count <= 0:
        label_count = 1
    if dedup_entries:
        label_step = max(len(dedup_entries) // label_count, 1)
        label_entries = dedup_entries[::label_step]
        if label_entries[-1][0] != dedup_entries[-1][0]:
            label_entries.append(dedup_entries[-1])
    else:
        label_entries = []

    sampled_image_points = np.array(
        [point for _, modality, _, point in label_entries if modality == "image"],
        dtype=np.float32,
    ) if label_entries else np.zeros((0, 2), dtype=np.float32)
    sampled_lidar_points = np.array(
        [point for _, modality, _, point in label_entries if modality == "lidar"],
        dtype=np.float32,
    ) if label_entries else np.zeros((0, 2), dtype=np.float32)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.imshow(image)
    if path_points.shape[0] >= 2:
        ax.plot(path_points[:, 0], path_points[:, 1], color="white", linewidth=0.8, alpha=0.35)
    if sampled_image_points.shape[0] > 0:
        ax.scatter(
            sampled_image_points[:, 0],
            sampled_image_points[:, 1],
            c="#66c2a5",
            s=26,
            marker="s",
            edgecolors="black",
            linewidths=0.2,
            alpha=0.85,
            label="image token",
        )
    if sampled_lidar_points.shape[0] > 0:
        ax.scatter(
            sampled_lidar_points[:, 0],
            sampled_lidar_points[:, 1],
            c="#ff8c42",
            s=54,
            marker="s",
            edgecolors="white",
            linewidths=0.3,
            alpha=0.92,
            label="lidar token",
        )
    for order_idx, modality, _, point in label_entries:
        ax.text(
            float(point[0]),
            float(point[1]),
            f"{order_idx}",
            fontsize=6,
            color="yellow" if modality == "lidar" else "white",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="black", edgecolor="none", alpha=0.55),
        )

    ax.legend(loc="lower right", fontsize=8, framealpha=0.7)
    ax.set_title(f"Enhanced Joint Order Labels, sample={sample_index}, view={view_index}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def default_output_path(args):
    output_dir = PROJECT_ROOT / "work_dirs" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{args.vis_mode}_sample{args.sample_index}.png"


def run_visualization(args):
    ensure_pythonpath()
    cfg = load_cfg(args.config, None)
    dataset = build_dataset(cfg, args.dataset_split)
    device = torch.device(args.device)
    model = build_model(cfg, args.checkpoint, device)
    data = collect_serialization(
        model,
        prepare_batch(dataset, model, args.sample_index),
        args.vis_mode,
        args.main_view,
        vis_patch_size=args.vis_patch_size,
        vis_voxel_size=args.vis_voxel_size,
    )

    output_path = Path(args.output) if args.output is not None else default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    main_view = None
    if data["debug"]["main_view_indices"] is not None:
        main_view = int(data["debug"]["main_view_indices"][0].item())

    if args.vis_mode == "3d_serialization":
        render_3d_serialization(data, output_path, args.sample_index)
    elif args.vis_mode == "2d_serialization":
        render_view = main_view if args.view_index is None else args.view_index
        render_2d_serialization(data, output_path, args.sample_index, render_view)
    elif args.vis_mode == "lidar_projection_2d":
        render_view = main_view if args.view_index is None else args.view_index
        render_lidar_projection_2d(data, output_path, args.sample_index, render_view)
    else:
        render_view = main_view if args.view_index is None else args.view_index
        render_lidar_projection_2d_enhanced(
            data,
            output_path,
            args.sample_index,
            render_view,
            label_count=args.label_count,
        )

    print(f"Saved visualization to: {output_path}")
    print(f"Sample index: {args.sample_index}")
    if args.vis_mode in ("2d_serialization", "lidar_projection_2d", "lidar_projection_2d_enhanced"):
        print(f"Main view: {main_view}")
        print(f"Rendered view: {main_view if args.view_index is None else args.view_index}")
    print(f"Visualization mode: {args.vis_mode}")


def main():
    args = parse_args()
    run_visualization(args)


if __name__ == "__main__":
    main()
