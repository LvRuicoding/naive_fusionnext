import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    parser = argparse.ArgumentParser(description="Visualize sorted FusionNeXt token ranks on image and lidar BEV.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Config file path")
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index")
    parser.add_argument("--layer-index", type=int, default=0, help="Backbone layer index to visualize")
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
        help="Image view index to draw. In 2d mode defaults to the selected main view.",
    )
    parser.add_argument(
        "--main-view",
        type=int,
        default=1,
        help="Default main view used by 2d serializer visualization in eval mode.",
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
    model.core.fusion_backbone.serializer.default_main_view = cfg.get("default_main_view", 1)
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


def save_rgb_image(image, path):
    plt.imsave(path, image)


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


def advance_one_layer(backbone, layer_index, lidar_tokens, lidar_coords, img_tokens, img_kuvd, K, T_c2w, post_rots, post_trans, lidar_padding_mask, img_padding_mask):
    mode = backbone.layer_modes[layer_index]
    layer_input_tokens = torch.cat([lidar_tokens, img_tokens], dim=1)
    layer_input_padding_mask = torch.cat([lidar_padding_mask, img_padding_mask], dim=1)
    serializer_out = backbone.serializer(
        lidar_tokens,
        lidar_coords,
        img_tokens,
        img_kuvd,
        K,
        T_c2w,
        post_rots=post_rots,
        post_trans=post_trans,
        mode=mode,
        lidar_padding_mask=lidar_padding_mask,
        img_padding_mask=img_padding_mask,
        return_debug=True,
    )
    sorted_tokens, sorted_indices, sorted_padding_mask, num_serialized_lidar, num_views, debug = serializer_out
    attended_tokens = backbone.blocks[layer_index](sorted_tokens, padding_mask=sorted_padding_mask)
    sorted_updates = attended_tokens - sorted_tokens
    sorted_updates = sorted_updates.masked_fill(sorted_padding_mask.unsqueeze(-1), 0)

    inverse_indices = torch.argsort(sorted_indices, dim=1)
    expanded_inv_indices = inverse_indices.unsqueeze(-1).expand_as(sorted_updates)
    recovered_updates = torch.gather(sorted_updates, 1, expanded_inv_indices)
    recovered_padding_mask = torch.gather(sorted_padding_mask, 1, inverse_indices)
    recovered_tokens = layer_input_tokens + recovered_updates
    recovered_tokens = recovered_tokens.masked_fill(layer_input_padding_mask.unsqueeze(-1), 0)

    recovered_lidar_tokens = recovered_tokens[:, :num_serialized_lidar, :]
    recovered_lidar_padding_mask = recovered_padding_mask[:, :num_serialized_lidar]
    current_img_tokens = recovered_tokens[:, num_serialized_lidar:, :]
    current_img_padding_mask = recovered_padding_mask[:, num_serialized_lidar:]

    if mode == "2d" and num_views > 1:
        batch_size, num_lidar_tokens, channels = lidar_tokens.shape
        recovered_lidar_tokens = recovered_lidar_tokens.reshape(batch_size, num_views, num_lidar_tokens, channels)
        recovered_lidar_padding_mask = recovered_lidar_padding_mask.reshape(batch_size, num_views, num_lidar_tokens)
        lidar_valid_mask = ~recovered_lidar_padding_mask
        valid_counts = lidar_valid_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        current_lidar_tokens = (recovered_lidar_tokens * lidar_valid_mask.unsqueeze(-1)).sum(dim=1) / valid_counts
        current_lidar_padding_mask = ~lidar_valid_mask.any(dim=1)
    else:
        current_lidar_tokens = recovered_lidar_tokens
        current_lidar_padding_mask = recovered_lidar_padding_mask

    current_lidar_tokens = current_lidar_tokens[:, : lidar_tokens.shape[1], :]
    current_lidar_padding_mask = current_lidar_padding_mask[:, : lidar_tokens.shape[1]]
    return (
        current_lidar_tokens,
        current_img_tokens,
        current_lidar_padding_mask,
        current_img_padding_mask,
        sorted_indices,
        sorted_padding_mask,
        debug,
        mode,
    )


def run_visualization(args):
    ensure_pythonpath()
    from fusionnext.models.utils import prepare_fusion_inputs

    cfg = load_cfg(args.config, None)
    dataset = build_dataset(cfg, args.dataset_split)
    raw_data_info = dataset.get_data_info(args.sample_index)
    camera_order = list(cfg.data_config["cams"])
    device = torch.device(args.device)
    model = build_model(cfg, args.checkpoint, device)
    model.core.fusion_backbone.serializer.default_main_view = args.main_view

    processed = prepare_batch(dataset, model, args.sample_index)
    inputs = processed["inputs"]
    data_samples = processed["data_samples"]
    img_metas = [sample.metainfo for sample in data_samples]

    with torch.no_grad():
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

        if args.layer_index < 0 or args.layer_index >= core.fusion_backbone.num_layers:
            raise ValueError(
                f"layer_index must be in [0, {core.fusion_backbone.num_layers - 1}], got {args.layer_index}"
            )

        debug = None
        mode = None
        for layer_index in range(args.layer_index + 1):
            (
                lidar_tokens,
                img_tokens,
                lidar_padding_mask,
                img_padding_mask,
                sorted_indices,
                sorted_padding_mask,
                debug,
                mode,
            ) = advance_one_layer(
                core.fusion_backbone,
                layer_index,
                lidar_tokens,
                lidar_coords,
                img_tokens,
                img_kuvd,
                intrins,
                cam_to_lidar,
                post_rots,
                post_trans,
                lidar_padding_mask,
                img_padding_mask,
            )

    if debug is None or mode is None:
        raise RuntimeError("Failed to collect serializer debug information.")

    batch_index = 0
    num_lidar = int(lidar_coords.shape[1])
    rank_map = get_rank_map(sorted_indices, sorted_padding_mask)[batch_index].detach().cpu()
    lidar_valid_mask = (~lidar_padding_mask[batch_index]).detach().cpu()
    lidar_rank = rank_map[:num_lidar][lidar_valid_mask]
    lidar_xy = lidar_coords[batch_index, lidar_valid_mask, :2].detach().cpu().numpy()

    if args.view_index is None:
        if mode == "2d" and debug["main_view_indices"] is not None:
            view_index = int(debug["main_view_indices"][batch_index].item())
        else:
            view_index = args.main_view
    else:
        view_index = args.view_index

    img_coords_batch = img_coords[batch_index].detach().cpu()
    img_valid_mask = (~img_padding_mask[batch_index]).detach().cpu()
    view_mask = (img_coords_batch[:, 0].long() == view_index) & img_valid_mask
    view_img_indices = torch.nonzero(view_mask, as_tuple=False).squeeze(1)
    view_img_abs_indices = view_img_indices + num_lidar
    image_rank = rank_map[view_img_abs_indices]
    image_uv = img_coords_batch[view_img_indices, 1:3].numpy()

    image = denormalize_image(img[batch_index, view_index])

    output_path = args.output
    if output_path is None:
        output_dir = PROJECT_ROOT / "work_dirs" / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"sorted_tokens_sample{args.sample_index}_layer{args.layer_index}_{mode}.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if view_index < 0 or view_index >= len(camera_order):
        raise ValueError(f"view_index must be in [0, {len(camera_order) - 1}], got {view_index}")
    missing_cams = [cam_name for cam_name in camera_order if cam_name not in raw_data_info["curr"]["cams"]]
    if missing_cams:
        raise KeyError(f"Sample is missing camera records for: {missing_cams}")
    copied_raw_image_paths = []
    for cam_idx, cam_name in enumerate(camera_order):
        raw_image_path = Path(raw_data_info["curr"]["cams"][cam_name]["data_path"])
        copied_raw_image_path = output_path.parent / (
            f"{output_path.stem}_raw_view{cam_idx}_{cam_name}{raw_image_path.suffix}"
        )
        shutil.copy2(raw_image_path, copied_raw_image_path)
        copied_raw_image_paths.append((cam_idx, cam_name, copied_raw_image_path))

    augmented_image_paths = []
    for cam_idx, cam_name in enumerate(camera_order):
        augmented_image = denormalize_image(img[batch_index, cam_idx])
        augmented_image_path = output_path.parent / f"{output_path.stem}_aug_view{cam_idx}_{cam_name}.png"
        save_rgb_image(augmented_image, augmented_image_path)
        augmented_image_paths.append((cam_idx, cam_name, augmented_image_path))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    bev_scatter = axes[0].scatter(
        lidar_xy[:, 0],
        lidar_xy[:, 1],
        c=lidar_rank.numpy(),
        cmap="viridis",
        s=8,
        linewidths=0,
    )
    axes[0].set_title(f"Lidar Token Rank, layer={args.layer_index}, mode={mode}")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].imshow(image)
    img_scatter = axes[1].scatter(
        image_uv[:, 0],
        image_uv[:, 1],
        c=image_rank.numpy(),
        cmap="viridis",
        s=18,
        marker="s",
        linewidths=0,
        alpha=0.85,
    )
    axes[1].set_title(f"Image Token Rank on view={view_index}")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    axes[1].set_xlim(0, image.shape[1])
    axes[1].set_ylim(image.shape[0], 0)

    colorbar = fig.colorbar(img_scatter, ax=axes, shrink=0.95)
    colorbar.set_label("Normalized Rank")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    main_view = None
    if debug["main_view_indices"] is not None:
        main_view = int(debug["main_view_indices"][batch_index].item())
    print(f"Saved visualization to: {output_path}")
    print("Copied raw images:")
    for cam_idx, cam_name, copied_path in copied_raw_image_paths:
        marker = " [rendered]" if cam_idx == view_index else ""
        print(f"  view {cam_idx} ({cam_name}): {copied_path}{marker}")
    print("Saved augmented images:")
    for cam_idx, cam_name, augmented_path in augmented_image_paths:
        marker = " [rendered]" if cam_idx == view_index else ""
        print(f"  view {cam_idx} ({cam_name}): {augmented_path}{marker}")
    print(f"Sample index: {args.sample_index}")
    print(f"Sample token: {raw_data_info.get('token')}")
    print(f"Layer index: {args.layer_index}")
    print(f"Layer mode: {mode}")
    print(f"Main view: {main_view}")
    print(f"Rendered view: {view_index}")


def main():
    args = parse_args()
    run_visualization(args)


if __name__ == "__main__":
    main()
