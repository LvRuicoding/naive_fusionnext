from typing import Sequence

import torch

from .geometry import build_homogeneous_transform_torch


def unpack_img_inputs(img_inputs):
    if not isinstance(img_inputs, (list, tuple)):
        raise TypeError(f"img_inputs must be a tuple/list, but got {type(img_inputs)}")
    if len(img_inputs) not in (6, 7):
        raise ValueError(f"Expected img_inputs with 6 or 7 items, but got {len(img_inputs)}")

    imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans = img_inputs[:6]
    bda = img_inputs[6] if len(img_inputs) == 7 else None
    return imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda


def get_lidar_to_global(img_metas: Sequence[dict], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    lidar_to_global = []
    for meta in img_metas:
        fusion_meta = meta.get("fusionnext_meta")
        if fusion_meta is None:
            raise KeyError(
                "img_metas must contain 'fusionnext_meta'. "
                "Use FusionNeXtPrepareMeta in the dataset pipeline."
            )
        lidar2ego = build_homogeneous_transform_torch(
            fusion_meta["lidar2ego_translation"],
            fusion_meta["lidar2ego_rotation"],
            device=device,
            dtype=dtype,
        )
        ego2global = build_homogeneous_transform_torch(
            fusion_meta["ego2global_translation"],
            fusion_meta["ego2global_rotation"],
            device=device,
            dtype=dtype,
        )
        lidar_to_global.append(ego2global @ lidar2ego)
    return torch.stack(lidar_to_global, dim=0)


def prepare_fusion_inputs(points, img_inputs, img_metas):
    imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = unpack_img_inputs(img_inputs)
    device = imgs.device
    dtype = imgs.dtype

    if not isinstance(points, list):
        raise TypeError(f"points must be a list[Tensor] in OpenMMLab mode, but got {type(points)}")

    sensor2egos = sensor2egos.to(device=device, dtype=dtype)
    ego2globals = ego2globals.to(device=device, dtype=dtype)
    intrins = intrins.to(device=device, dtype=dtype)
    post_rots = post_rots.to(device=device, dtype=dtype)
    post_trans = post_trans.to(device=device, dtype=dtype)

    lidar_to_global = get_lidar_to_global(img_metas, device=device, dtype=dtype)
    global_to_lidar = torch.inverse(lidar_to_global).unsqueeze(1)
    cam_to_global = ego2globals @ sensor2egos
    cam_to_lidar = global_to_lidar @ cam_to_global

    if bda is not None:
        bda = bda.to(device=device, dtype=dtype)
        cam_to_lidar = bda.unsqueeze(1) @ cam_to_lidar

    return imgs, points, intrins, cam_to_lidar, post_rots, post_trans
