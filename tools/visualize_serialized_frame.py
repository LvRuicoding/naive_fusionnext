import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train import ensure_pythonpath, load_cfg
from tools.visualize_sorted_tokens import (
    build_dataset,
    build_model,
    collect_serialization,
    denormalize_image,
    get_rank_map,
    prepare_batch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize serialization order for one whole frame.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "fusionnext_nuscenes_mini_3d.py"),
        help="Config file path",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index")
    parser.add_argument(
        "--vis-mode",
        choices=["2d_serialization", "3d_serialization"],
        required=True,
        help="Serialization mode to visualize across the whole frame",
    )
    parser.add_argument(
        "--dataset-split",
        choices=["train"],
        default="train",
        help="Dataset split to visualize",
    )
    parser.add_argument(
        "--main-view",
        type=int,
        default=1,
        help="Main view index used by the 2d serializer in eval mode",
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
    return parser.parse_args()


def default_output_path(args):
    output_dir = PROJECT_ROOT / "work_dirs" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{args.vis_mode}_frame_sample{args.sample_index}.png"


def render_frame(data, output_path, sample_index, camera_order, vis_mode):
    batch_index = 0
    num_lidar = int(data["lidar_coords"].shape[1])
    rank_map = get_rank_map(data["sorted_indices"], data["sorted_padding_mask"])[batch_index].detach().cpu()

    lidar_valid_mask = (~data["lidar_padding_mask"][batch_index]).detach().cpu()
    lidar_rank = rank_map[:num_lidar][lidar_valid_mask].numpy()
    lidar_xy = data["lidar_coords"][batch_index, lidar_valid_mask, :2].detach().cpu().numpy()

    img_coords_batch = data["img_coords"][batch_index].detach().cpu()
    img_valid_mask = (~data["img_padding_mask"][batch_index]).detach().cpu()

    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    grid = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1.1, 1.0, 1.0, 1.0])
    bev_ax = fig.add_subplot(grid[:, 0])
    image_axes = [
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[0, 3]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
        fig.add_subplot(grid[1, 3]),
    ]

    bev_scatter = bev_ax.scatter(
        lidar_xy[:, 0],
        lidar_xy[:, 1],
        c=lidar_rank,
        cmap="viridis",
        norm=norm,
        s=8,
        linewidths=0,
    )
    bev_ax.set_title("Lidar / BEV")
    bev_ax.set_xlabel("X")
    bev_ax.set_ylabel("Y")
    bev_ax.set_aspect("equal", adjustable="box")

    for view_index, ax in enumerate(image_axes):
        image = denormalize_image(data["img"][batch_index, view_index])
        view_mask = (img_coords_batch[:, 0].long() == view_index) & img_valid_mask
        view_img_indices = torch.nonzero(view_mask, as_tuple=False).squeeze(1)
        view_img_abs_indices = view_img_indices + num_lidar
        image_rank = rank_map[view_img_abs_indices].numpy()
        image_uv = img_coords_batch[view_img_indices, 1:3].numpy()

        ax.imshow(image)
        ax.scatter(
            image_uv[:, 0],
            image_uv[:, 1],
            c=image_rank,
            cmap="viridis",
            norm=norm,
            s=12,
            marker="s",
            linewidths=0,
            alpha=0.85,
        )
        title = f"view {view_index}: {camera_order[view_index]}"
        main_view_indices = data["debug"]["main_view_indices"]
        if main_view_indices is not None and int(main_view_indices[batch_index].item()) == view_index:
            title += " [main]"
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])

    title = f"{vis_mode}, sample={sample_index}"
    main_view_indices = data["debug"]["main_view_indices"]
    if main_view_indices is not None:
        title += f", main_view={int(main_view_indices[batch_index].item())}"
    fig.suptitle(title, fontsize=14)

    colorbar = fig.colorbar(bev_scatter, ax=[bev_ax, *image_axes], shrink=0.92)
    colorbar.set_label("Normalized Rank")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args):
    ensure_pythonpath()
    cfg = load_cfg(args.config, None)
    dataset = build_dataset(cfg, args.dataset_split)
    camera_order = list(cfg.data_config["cams"])
    device = torch.device(args.device)
    model = build_model(cfg, args.checkpoint, device)
    processed = prepare_batch(dataset, model, args.sample_index)
    data = collect_serialization(model, processed, args.vis_mode, args.main_view)

    output_path = Path(args.output) if args.output is not None else default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_frame(data, output_path, args.sample_index, camera_order, args.vis_mode)

    print(f"Saved frame visualization to: {output_path}")
    print(f"Sample index: {args.sample_index}")
    print(f"Visualization mode: {args.vis_mode}")
    if data["debug"]["main_view_indices"] is not None:
        print(f"Main view: {int(data['debug']['main_view_indices'][0].item())}")


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
