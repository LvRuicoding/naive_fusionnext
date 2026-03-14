import torch
import torch.nn as nn


class RealLidarTokenizer(nn.Module):
    def __init__(self, point_cloud_range, voxel_size, in_channels=5, embed_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.register_buffer("pc_range", torch.tensor(point_cloud_range, dtype=torch.float32))
        self.register_buffer("voxel_size", torch.tensor(voxel_size, dtype=torch.float32))
        self.vfe_mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, points_list):
        if isinstance(points_list, torch.Tensor):
            points_list = [points for points in points_list]

        device = points_list[0].device
        pc_range = self.pc_range.to(device)
        voxel_size = self.voxel_size.to(device)

        all_voxel_tokens = []
        all_voxel_coords = []

        for points in points_list:
            if points.dim() != 2:
                raise ValueError("Each point cloud sample must have shape (N_points, C)")
            if points.shape[-1] != self.in_channels:
                raise ValueError(
                    f"Expected point features with {self.in_channels} channels, "
                    f"but got {points.shape[-1]}"
                )
            grid_coords = torch.floor((points[:, :3] - pc_range[:3]) / voxel_size).long()

            pc_min = pc_range[:3]
            pc_max = pc_range[3:6]
            valid_mask = (
                (points[:, 0] >= pc_min[0]) & (points[:, 0] < pc_max[0]) &
                (points[:, 1] >= pc_min[1]) & (points[:, 1] < pc_max[1]) &
                (points[:, 2] >= pc_min[2]) & (points[:, 2] < pc_max[2])
            )
            points = points[valid_mask]
            grid_coords = grid_coords[valid_mask]
            if points.shape[0] == 0:
                embed_dim = self.vfe_mlp[3].out_features
                all_voxel_tokens.append(points.new_zeros((0, embed_dim)))
                all_voxel_coords.append(points.new_zeros((0, 3)))
                continue

            unq_coords, inverse_indices = torch.unique(grid_coords, dim=0, return_inverse=True)
            num_voxels = unq_coords.shape[0]

            point_feats = self.vfe_mlp(points)
            voxel_feats = point_feats.new_zeros((num_voxels, point_feats.shape[-1]))
            scatter_indices = inverse_indices.unsqueeze(1).expand(-1, point_feats.shape[-1])
            voxel_feats.scatter_reduce_(0, scatter_indices, point_feats, reduce="amax", include_self=False)

            real_physical_coords = unq_coords.float() * voxel_size + (voxel_size / 2.0) + pc_range[:3]
            all_voxel_tokens.append(voxel_feats)
            all_voxel_coords.append(real_physical_coords)

        max_voxels = max(tokens.shape[0] for tokens in all_voxel_tokens)
        embed_dim = self.vfe_mlp[3].out_features
        batch_size = len(all_voxel_tokens)

        padded_tokens = all_voxel_tokens[0].new_zeros((batch_size, max_voxels, embed_dim))
        padded_coords = all_voxel_coords[0].new_zeros((batch_size, max_voxels, 3))
        padding_mask = torch.ones((batch_size, max_voxels), dtype=torch.bool, device=device)

        for i, (tokens, coords) in enumerate(zip(all_voxel_tokens, all_voxel_coords)):
            num_voxels = tokens.shape[0]
            if num_voxels == 0:
                continue
            padded_tokens[i, :num_voxels] = tokens
            padded_coords[i, :num_voxels] = coords
            padding_mask[i, :num_voxels] = False

        return padded_tokens, padded_coords, padding_mask
