import torch
import torch.nn as nn


class GeometrySerializer(nn.Module):
    def __init__(self, grid_size_3d=1.0, grid_size_2d=16.0, default_main_view=1):
        super().__init__()
        self.grid_size_3d = grid_size_3d
        self.grid_size_2d = grid_size_2d
        self.default_main_view = default_main_view

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

    def choose_main_views(self, batch_size, num_views, device):
        if self.training:
            return torch.randint(0, num_views, (batch_size,), device=device)
        default_view = min(self.default_main_view, num_views - 1)
        return torch.full((batch_size,), default_view, device=device, dtype=torch.long)

    def project_lidar_to_selected_views(self, lidar_coords, K, T_c2w, view_indices, post_rots=None, post_trans=None):
        K, T_c2w = self.normalize_camera_params(K, T_c2w)
        post_rots, post_trans = self.normalize_post_transforms(post_rots, post_trans, K)
        B, N_lidar, _ = lidar_coords.shape

        batch_indices = torch.arange(B, device=lidar_coords.device)
        selected_K = K[batch_indices, view_indices]
        selected_T_c2w = T_c2w[batch_indices, view_indices]
        selected_post_rots = post_rots[batch_indices, view_indices]
        selected_post_trans = post_trans[batch_indices, view_indices]

        xyz1 = torch.cat([lidar_coords, torch.ones_like(lidar_coords[..., :1])], dim=-1)
        cam_coords_homo = torch.matmul(
            torch.inverse(selected_T_c2w).unsqueeze(1),
            xyz1.unsqueeze(-1),
        ).squeeze(-1)
        cam_coords = cam_coords_homo[..., :3]

        img_coords_homo = torch.matmul(selected_K.unsqueeze(1), cam_coords.unsqueeze(-1)).squeeze(-1)
        depth = torch.clamp(img_coords_homo[..., 2:3], min=1e-5)
        uv = img_coords_homo[..., :2] / depth
        uv = self.apply_post_transform(uv, selected_post_rots.unsqueeze(1), selected_post_trans.unsqueeze(1))

        camera_ids = view_indices.view(B, 1, 1).expand(-1, N_lidar, -1).to(dtype=uv.dtype)
        return torch.cat([camera_ids, uv], dim=-1)

    def compute_snake_sort_keys(self, coords, padding_mask, grid_size):
        q_coords = torch.floor(coords / grid_size).to(torch.int64)
        valid_mask = ~padding_mask
        if valid_mask.any():
            min_coords = q_coords[valid_mask].min(dim=0)[0]
            q_coords = q_coords - min_coords
            d0, d1, d2 = q_coords[:, 0], q_coords[:, 1], q_coords[:, 2]
            max_d1 = d1[valid_mask].max()
            max_d2 = d2[valid_mask].max()
        else:
            d0 = q_coords[:, 0]
            d1 = q_coords[:, 1]
            d2 = q_coords[:, 2]
            max_d1 = torch.zeros((), dtype=q_coords.dtype, device=q_coords.device)
            max_d2 = torch.zeros((), dtype=q_coords.dtype, device=q_coords.device)

        snake_d1 = torch.where(d0 % 2 == 1, max_d1 - d1, d1)
        snake_d2 = torch.where((d0 + d1) % 2 == 1, max_d2 - d2, d2)

        multiplier = 100000
        sort_keys = d0 * (multiplier ** 2) + snake_d1 * multiplier + snake_d2
        pad_key = torch.full_like(sort_keys, torch.iinfo(sort_keys.dtype).max)
        return torch.where(padding_mask, pad_key, sort_keys)

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
        return_debug=False,
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
            num_serialized_lidar = N_lidar
            num_views = 1
            sort_keys = torch.stack(
                [
                    self.compute_snake_sort_keys(
                        unified_coords[batch_idx],
                        unified_padding_mask[batch_idx],
                        self.grid_size_3d,
                    )
                    for batch_idx in range(B)
                ],
                dim=0,
            )
            sorted_indices = torch.argsort(sort_keys, dim=1)
            debug_info = {
                "mode": mode,
                "main_view_indices": None,
                "unified_coords": unified_coords,
                "unified_padding_mask": unified_padding_mask,
                "sort_keys": sort_keys,
            }
        elif mode == "2d":
            total_views = K.shape[1]
            main_view_indices = self.choose_main_views(B, total_views, lidar_tokens.device)
            lidar_coords_2d = self.project_lidar_to_selected_views(
                lidar_coords,
                K,
                T_c2w,
                main_view_indices,
                post_rots,
                post_trans,
            )
            serialized_lidar_tokens = lidar_tokens
            serialized_lidar_padding = lidar_padding_mask
            img_kuv = img_kuvd[..., :3]
            unified_coords = torch.cat([lidar_coords_2d, img_kuv], dim=1)
            unified_tokens = torch.cat([serialized_lidar_tokens, img_tokens], dim=1)
            unified_padding_mask = torch.cat([serialized_lidar_padding, img_padding_mask], dim=1)
            num_serialized_lidar = N_lidar
            num_views = 1
            lidar_indices = torch.arange(N_lidar, device=lidar_tokens.device, dtype=torch.long)
            sorted_indices_list = []
            for batch_idx in range(B):
                cam_ids = img_kuv[batch_idx, :, 0].long()
                main_view = int(main_view_indices[batch_idx].item())
                ordered_parts = []

                for view_idx in range(total_views):
                    view_img_rel = torch.nonzero(cam_ids == view_idx, as_tuple=False).squeeze(1)
                    view_img_abs = view_img_rel + N_lidar
                    if view_idx != main_view:
                        ordered_parts.append(view_img_abs)
                        continue

                    main_coords = torch.cat(
                        [lidar_coords_2d[batch_idx], img_kuv[batch_idx, view_img_rel]],
                        dim=0,
                    )
                    main_padding = torch.cat(
                        [serialized_lidar_padding[batch_idx], img_padding_mask[batch_idx, view_img_rel]],
                        dim=0,
                    )
                    main_indices = torch.cat([lidar_indices, view_img_abs], dim=0)
                    main_sort_keys = self.compute_snake_sort_keys(main_coords, main_padding, self.grid_size_2d)
                    main_order = torch.argsort(main_sort_keys, dim=0)
                    ordered_parts.append(main_indices[main_order])

                sorted_indices_list.append(torch.cat(ordered_parts, dim=0))

            sorted_indices = torch.stack(sorted_indices_list, dim=0)
            debug_info = {
                "mode": mode,
                "main_view_indices": main_view_indices,
                "unified_coords": unified_coords,
                "unified_padding_mask": unified_padding_mask,
                "sort_keys": None,
            }
        else:
            raise ValueError("mode must be '2d' or '3d'")
        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, C)
        sorted_tokens = torch.gather(unified_tokens, 1, expanded_indices)
        sorted_padding_mask = torch.gather(unified_padding_mask, 1, sorted_indices)
        if return_debug:
            return sorted_tokens, sorted_indices, sorted_padding_mask, num_serialized_lidar, num_views, debug_info
        return sorted_tokens, sorted_indices, sorted_padding_mask, num_serialized_lidar, num_views
