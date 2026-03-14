import torch
import torch.nn as nn

from ..backbones import FusionNeXtBackbone
from ..tokenizers import ImageTokenizer, RealLidarTokenizer


class FusionNeXtMini(nn.Module):
    def __init__(
        self,
        point_cloud_range,
        voxel_size,
        embed_dim=256,
        image_weights=None,
        lidar_in_channels=5,
    ):
        super().__init__()
        self.image_branch = ImageTokenizer(embed_dim, image_weights=image_weights)
        self.lidar_branch = RealLidarTokenizer(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            in_channels=lidar_in_channels,
            embed_dim=embed_dim,
        )
        self.fusion_backbone = FusionNeXtBackbone(embed_dim, num_layers=4)

    def forward(self, img, voxels, K, T_c2w, post_rots=None, post_trans=None, return_dict=False):
        if K.dim() == 4 and K.shape[1] != img.shape[1]:
            raise ValueError("K must provide one intrinsic matrix per input camera view")
        if T_c2w.dim() == 4 and T_c2w.shape[1] != img.shape[1]:
            raise ValueError("T_c2w must provide one extrinsic matrix per input camera view")

        img_tokens, img_coords, img_depths = self.image_branch(img)
        voxel_tokens, voxel_coords, voxel_padding_mask = self.lidar_branch(voxels)
        img_kuvd = torch.cat([img_coords, img_depths], dim=-1)

        final_1d_sequence = self.fusion_backbone(
            voxel_tokens,
            voxel_coords,
            img_tokens,
            img_kuvd,
            K,
            T_c2w,
            post_rots=post_rots,
            post_trans=post_trans,
            lidar_padding_mask=voxel_padding_mask,
        )
        num_lidar_tokens = voxel_tokens.shape[1]
        fused_lidar_tokens = final_1d_sequence[:, :num_lidar_tokens, :]
        fused_img_tokens = final_1d_sequence[:, num_lidar_tokens:, :]

        if return_dict:
            img_padding_mask = torch.zeros(
                (img_tokens.shape[0], img_tokens.shape[1]),
                dtype=torch.bool,
                device=img_tokens.device,
            )
            return {
                "fusion_tokens": final_1d_sequence,
                "lidar_tokens": fused_lidar_tokens,
                "img_tokens": fused_img_tokens,
                "lidar_coords": voxel_coords,
                "lidar_padding_mask": voxel_padding_mask,
                "unified_padding_mask": torch.cat([voxel_padding_mask, img_padding_mask], dim=1),
                "num_lidar_tokens": num_lidar_tokens,
                "img_kuvd": img_kuvd,
            }
        return final_1d_sequence
