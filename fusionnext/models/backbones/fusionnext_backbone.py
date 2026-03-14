import torch
import torch.nn as nn

from ..layers import FlashWindowBlock
from ..serialization import GeometrySerializer


class FusionNeXtBackbone(nn.Module):
    def __init__(self, embed_dim=256, num_layers=4, window_size=80):
        super().__init__()
        self.num_layers = num_layers
        self.window_size = window_size
        self.serializer = GeometrySerializer()
        self.layer_modes = ["2d" if i % 2 == 0 else "3d" for i in range(num_layers)]
        self.blocks = nn.ModuleList(
            [
                FlashWindowBlock(
                    embed_dim=embed_dim,
                    num_heads=8,
                    mlp_ratio=4,
                    window_size=window_size,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        lidar_tokens,
        lidar_coords,
        img_tokens,
        img_kuvd,
        K,
        T_c2w,
        post_rots=None,
        post_trans=None,
        lidar_padding_mask=None,
    ):
        B, N_lidar, C = lidar_tokens.shape
        _, N_img, _ = img_tokens.shape
        current_lidar_tokens = lidar_tokens
        current_img_tokens = img_tokens
        if lidar_padding_mask is None:
            lidar_padding_mask = torch.zeros((B, N_lidar), dtype=torch.bool, device=lidar_tokens.device)
        current_lidar_padding_mask = lidar_padding_mask
        current_img_padding_mask = torch.zeros((B, N_img), dtype=torch.bool, device=img_tokens.device)

        for i in range(self.num_layers):
            mode = self.layer_modes[i]
            layer_input_tokens = torch.cat([current_lidar_tokens, current_img_tokens], dim=1)
            layer_input_padding_mask = torch.cat([current_lidar_padding_mask, current_img_padding_mask], dim=1)
            sorted_tokens, sorted_indices, sorted_padding_mask, num_serialized_lidar, num_views = self.serializer(
                current_lidar_tokens,
                lidar_coords,
                current_img_tokens,
                img_kuvd,
                K,
                T_c2w,
                post_rots=post_rots,
                post_trans=post_trans,
                mode=mode,
                lidar_padding_mask=current_lidar_padding_mask,
                img_padding_mask=current_img_padding_mask,
            )
            attended_tokens = self.blocks[i](sorted_tokens, padding_mask=sorted_padding_mask)
            sorted_updates = attended_tokens - sorted_tokens
            sorted_updates = sorted_updates.masked_fill(sorted_padding_mask.unsqueeze(-1), 0)

            inverse_indices = torch.argsort(sorted_indices, dim=1)
            expanded_inv_indices = inverse_indices.unsqueeze(-1).expand(-1, -1, C)
            recovered_updates = torch.gather(sorted_updates, 1, expanded_inv_indices)
            recovered_padding_mask = torch.gather(sorted_padding_mask, 1, inverse_indices)
            recovered_tokens = layer_input_tokens + recovered_updates
            recovered_tokens = recovered_tokens.masked_fill(layer_input_padding_mask.unsqueeze(-1), 0)

            recovered_lidar_tokens = recovered_tokens[:, :num_serialized_lidar, :]
            recovered_lidar_padding_mask = recovered_padding_mask[:, :num_serialized_lidar]
            current_img_tokens = recovered_tokens[:, num_serialized_lidar:, :]
            current_img_padding_mask = recovered_padding_mask[:, num_serialized_lidar:]

            if mode == "2d" and num_views > 1:
                recovered_lidar_tokens = recovered_lidar_tokens.reshape(B, num_views, N_lidar, C)
                recovered_lidar_padding_mask = recovered_lidar_padding_mask.reshape(B, num_views, N_lidar)
                lidar_valid_mask = ~recovered_lidar_padding_mask
                valid_counts = lidar_valid_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
                current_lidar_tokens = (
                    recovered_lidar_tokens * lidar_valid_mask.unsqueeze(-1)
                ).sum(dim=1) / valid_counts
                current_lidar_padding_mask = ~lidar_valid_mask.any(dim=1)
            else:
                current_lidar_tokens = recovered_lidar_tokens
                current_lidar_padding_mask = recovered_lidar_padding_mask

            current_lidar_tokens = current_lidar_tokens[:, :N_lidar, :]
            current_lidar_padding_mask = current_lidar_padding_mask[:, :N_lidar]

        final_unified_tokens = torch.cat([current_lidar_tokens, current_img_tokens], dim=1)
        return final_unified_tokens
