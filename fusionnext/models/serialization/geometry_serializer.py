import torch
import torch.nn as nn


class GeometrySerializer(nn.Module):
    def __init__(self, grid_size_3d=1.0, grid_size_2d=16.0):
        super().__init__()
        self.grid_size_3d = grid_size_3d
        self.grid_size_2d = grid_size_2d

    def normalize_camera_params(self, K, T_c2w):
        if K.dim() == 3:
            K = K.unsqueeze(1)
        if T_c2w.dim() == 3:
            T_c2w = T_c2w.unsqueeze(1)
        if K.shape[:2] != T_c2w.shape[:2]:
            raise ValueError("K and T_c2w must have matching batch size and number of camera views")
        return K, T_c2w

    def normalize_post_transforms(self, post_rots, post_trans, K):
        B, N, _, _ = K.shape
        if post_rots is None:
            post_rots = torch.eye(3, device=K.device, dtype=K.dtype).view(1, 1, 3, 3).expand(B, N, -1, -1)
        elif post_rots.dim() == 3:
            post_rots = post_rots.unsqueeze(1)

        if post_trans is None:
            post_trans = torch.zeros((B, N, 3), device=K.device, dtype=K.dtype)
        elif post_trans.dim() == 2:
            post_trans = post_trans.unsqueeze(1)

        if post_rots.shape[:2] != K.shape[:2] or post_trans.shape[:2] != K.shape[:2]:
            raise ValueError("post_rots and post_trans must match batch size and number of camera views")
        return post_rots, post_trans

    def apply_post_transform(self, uv, post_rot, post_tran):
        uv1 = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1)
        uv1 = torch.matmul(post_rot, uv1.unsqueeze(-1)).squeeze(-1) + post_tran
        return uv1[..., :2]

    def invert_post_transform(self, uv, post_rot, post_tran):
        uv1 = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1)
        uv1 = uv1 - post_tran
        uv1 = torch.matmul(torch.inverse(post_rot), uv1.unsqueeze(-1)).squeeze(-1)
        return uv1[..., :2]

    def project_lidar_to_all_views(self, lidar_coords, K, T_c2w, post_rots=None, post_trans=None):
        K, T_c2w = self.normalize_camera_params(K, T_c2w)
        post_rots, post_trans = self.normalize_post_transforms(post_rots, post_trans, K)
        B, N_lidar, _ = lidar_coords.shape
        num_views = K.shape[1]

        xyz1 = torch.cat([lidar_coords, torch.ones_like(lidar_coords[..., :1])], dim=-1)
        xyz1 = xyz1.unsqueeze(1).unsqueeze(-1).expand(-1, num_views, -1, -1, -1)

        T_w2c = torch.inverse(T_c2w).unsqueeze(2)
        cam_coords_homo = torch.matmul(T_w2c, xyz1).squeeze(-1)
        cam_coords = cam_coords_homo[..., :3]

        img_coords_homo = torch.matmul(K.unsqueeze(2), cam_coords.unsqueeze(-1)).squeeze(-1)
        depth = torch.clamp(img_coords_homo[..., 2:3], min=1e-5)
        uv = img_coords_homo[..., :2] / depth
        uv = self.apply_post_transform(uv, post_rots.unsqueeze(2), post_trans.unsqueeze(2))

        camera_ids = torch.arange(num_views, device=lidar_coords.device, dtype=uv.dtype)
        camera_ids = camera_ids.view(1, num_views, 1, 1).expand(B, -1, N_lidar, -1)
        return torch.cat([camera_ids, uv], dim=-1)

    def unproject_image_to_3d(self, img_kuvd, K, T_c2w, post_rots=None, post_trans=None):
        K, T_c2w = self.normalize_camera_params(K, T_c2w)
        post_rots, post_trans = self.normalize_post_transforms(post_rots, post_trans, K)

        camera_indices = img_kuvd[..., 0].long()
        if camera_indices.numel() > 0 and camera_indices.max() >= K.shape[1]:
            raise ValueError("Image token camera index exceeds the provided camera parameters")
        u, v, D = img_kuvd[..., 1], img_kuvd[..., 2], img_kuvd[..., 3]

        batch_size, num_tokens = camera_indices.shape
        batch_indices = torch.arange(batch_size, device=img_kuvd.device).unsqueeze(1).expand(-1, num_tokens)
        token_K = K[batch_indices, camera_indices]
        token_T_c2w = T_c2w[batch_indices, camera_indices]
        token_post_rots = post_rots[batch_indices, camera_indices]
        token_post_trans = post_trans[batch_indices, camera_indices]

        uv = torch.stack([u, v], dim=-1)
        uv = self.invert_post_transform(uv, token_post_rots, token_post_trans)
        uv1 = torch.cat([uv, torch.ones_like(u).unsqueeze(-1)], dim=-1)

        K_inv = torch.inverse(token_K)
        cam_coords = torch.matmul(K_inv, uv1.unsqueeze(-1)).squeeze(-1) * D.unsqueeze(-1)
        cam_coords_homo = torch.cat([cam_coords, torch.ones_like(D).unsqueeze(-1)], dim=-1)
        world_coords_homo = torch.matmul(token_T_c2w, cam_coords_homo.unsqueeze(-1)).squeeze(-1)
        return world_coords_homo[..., :3]

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
        mode="3d",
        lidar_padding_mask=None,
        img_padding_mask=None,
    ):
        B, N_lidar, C = lidar_tokens.shape
        _, N_img, _ = img_tokens.shape
        K, T_c2w = self.normalize_camera_params(K, T_c2w)
        if lidar_padding_mask is None:
            lidar_padding_mask = torch.zeros((B, N_lidar), dtype=torch.bool, device=lidar_tokens.device)
        if img_padding_mask is None:
            img_padding_mask = torch.zeros((B, N_img), dtype=torch.bool, device=img_tokens.device)

        if mode == "3d":
            img_coords_3d = self.unproject_image_to_3d(img_kuvd, K, T_c2w, post_rots, post_trans)
            serialized_lidar_tokens = lidar_tokens
            serialized_lidar_padding = lidar_padding_mask
            unified_coords = torch.cat([lidar_coords, img_coords_3d], dim=1)
            unified_tokens = torch.cat([serialized_lidar_tokens, img_tokens], dim=1)
            unified_padding_mask = torch.cat([serialized_lidar_padding, img_padding_mask], dim=1)
            grid_size = self.grid_size_3d
            num_serialized_lidar = N_lidar
            num_views = 1
        elif mode == "2d":
            num_views = K.shape[1]
            lidar_coords_2d = self.project_lidar_to_all_views(lidar_coords, K, T_c2w, post_rots, post_trans)
            serialized_lidar_tokens = (
                lidar_tokens.unsqueeze(1).expand(-1, num_views, -1, -1).reshape(B, num_views * N_lidar, C)
            )
            serialized_lidar_padding = (
                lidar_padding_mask.unsqueeze(1).expand(-1, num_views, -1).reshape(B, num_views * N_lidar)
            )
            img_kuv = img_kuvd[..., :3]
            unified_coords = torch.cat([lidar_coords_2d.reshape(B, num_views * N_lidar, 3), img_kuv], dim=1)
            unified_tokens = torch.cat([serialized_lidar_tokens, img_tokens], dim=1)
            unified_padding_mask = torch.cat([serialized_lidar_padding, img_padding_mask], dim=1)
            grid_size = self.grid_size_2d
            num_serialized_lidar = num_views * N_lidar
        else:
            raise ValueError("mode must be '2d' or '3d'")

        q_coords = torch.floor(unified_coords / grid_size).to(torch.int64)
        min_coords = q_coords.min(dim=1, keepdim=True)[0]
        q_coords = q_coords - min_coords

        d0, d1, d2 = q_coords[:, :, 0], q_coords[:, :, 1], q_coords[:, :, 2]
        max_d1 = d1.max(dim=1, keepdim=True)[0]
        max_d2 = d2.max(dim=1, keepdim=True)[0]

        snake_d1 = torch.where(d0 % 2 == 1, max_d1 - d1, d1)
        snake_d2 = torch.where((d0 + d1) % 2 == 1, max_d2 - d2, d2)

        multiplier = 100000
        sort_keys = d0 * (multiplier ** 2) + snake_d1 * multiplier + snake_d2
        pad_key = torch.full_like(sort_keys, torch.iinfo(sort_keys.dtype).max)
        sort_keys = torch.where(unified_padding_mask, pad_key, sort_keys)

        sorted_indices = torch.argsort(sort_keys, dim=1)
        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, C)
        sorted_tokens = torch.gather(unified_tokens, 1, expanded_indices)
        sorted_padding_mask = torch.gather(unified_padding_mask, 1, sorted_indices)
        return sorted_tokens, sorted_indices, sorted_padding_mask, num_serialized_lidar, num_views
