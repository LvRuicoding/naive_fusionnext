from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

try:
    from mmdet.models import DETECTORS
    from mmdet3d.models.builder import build_head
    from mmdet3d.models.detectors.base import Base3DDetector
except ImportError:
    DETECTORS = None
    build_head = None
    Base3DDetector = nn.Module

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

from fusion_head import FusionNeXtSimple3DHead


def register_detector_module(cls):
    if DETECTORS is None:
        return cls
    return DETECTORS.register_module()(cls)


def quaternion_to_rotation_matrix_torch(
    quaternion: Sequence[float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    q = torch.as_tensor(quaternion, device=device, dtype=dtype)
    q = q / torch.linalg.norm(q)
    w, x, y, z = q.unbind()
    return torch.stack(
        [
            torch.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)]),
            torch.stack([2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)]),
            torch.stack([2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]),
        ]
    )


def build_homogeneous_transform_torch(
    translation: Sequence[float],
    rotation: Sequence[float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    transform = torch.eye(4, device=device, dtype=dtype)
    transform[:3, :3] = quaternion_to_rotation_matrix_torch(rotation, device, dtype)
    transform[:3, 3] = torch.as_tensor(translation, device=device, dtype=dtype)
    return transform


# ==========================================
# 1. 图像分支: 使用预训练 ResNet-50
# ==========================================
class ImageTokenizer(nn.Module):
    def __init__(self, embed_dim=256, image_weights=None):
        super().__init__()
        # 默认关闭权重下载；需要预训练权重时显式传入 "default" 或具体 weights 对象。
        if image_weights == "default":
            image_weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=image_weights)
        # 去掉最后的全局平均池化和全连接层，保留特征图输出
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        # ResNet-50 输出通道是 2048，投影到统一的 embed_dim
        self.proj = nn.Linear(2048, embed_dim)
        
        # 深度预测头
        self.depth_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1) 
        )

    def forward(self, img):
        """ img: (B, N_views, 3, H, W) """
        B, N_views, C, H, W = img.shape
        img_reshaped = img.reshape(B * N_views, C, H, W)
        
        # 提取特征: (B*N_views, 2048, H_feat, W_feat)
        features = self.backbone(img_reshaped)
        _, C_feat, H_feat, W_feat = features.shape
        
        # ==========================================
        # 分支 1：处理 Image Tokens
        # ==========================================
        # 展平为序列: (B*N_views, H_feat*W_feat, 2048)
        patch_tokens = features.flatten(2).permute(0, 2, 1)
        # 恢复 Batch 维度: (B, N_views * H_feat * W_feat, 2048)
        patch_tokens = patch_tokens.reshape(B, N_views * H_feat * W_feat, C_feat)
        
        # 线性投影到统一维度: (B, N_t, embed_dim)
        image_tokens = self.proj(patch_tokens)
        N_t = image_tokens.shape[1] # 记录当前图像分支产生的 Token 总数
        
        # ==========================================
        # 分支 2：处理 2D 坐标 [k, u, v]
        # ==========================================
        stride_h = H // H_feat
        stride_w = W // W_feat
        
        v_coords = torch.arange(H_feat, device=img.device) * stride_h + (stride_h // 2)
        u_coords = torch.arange(W_feat, device=img.device) * stride_w + (stride_w // 2)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing='ij')
        
        v_flat = v_grid.flatten()
        u_flat = u_grid.flatten()
        
        all_views_coords = []
        for k in range(N_views):
            k_tensor = torch.full_like(u_flat, k)
            kuv = torch.stack([k_tensor, u_flat, v_flat], dim=-1)
            all_views_coords.append(kuv)
            
        image_coords = torch.cat(all_views_coords, dim=0)
        image_coords = image_coords.unsqueeze(0).repeat(B, 1, 1).float()
        
        # ==========================================
        # ✨ 分支 3：新增的深度预测逻辑
        # ==========================================
        # 1. 预测深度图: (B*N_views, 1, H_feat, W_feat)
        depth_map = self.depth_head(features)
        
        # 2. 保证深度值为正数 (softplus 是平滑版的 ReLU，物理意义更稳定)
        depth_map = F.softplus(depth_map)
        
        # 3. 展平并调整维度，使其与 Tokens 数量对齐: (B, N_views * H_feat * W_feat, 1)
        # 保持最后有一个长度为 1 的维度，方便后续与 [k, u, v] 进行坐标运算
        depth_values = depth_map.reshape(B, N_t, 1)

        # 返回时带上 depth_values
        return image_tokens, image_coords, depth_values

# ==========================================
# 2. 点云分支: Voxel 特征提取
# ==========================================
class RealLidarTokenizer(nn.Module):
    def __init__(self, point_cloud_range, voxel_size, in_channels=5, embed_dim=256):
        super().__init__()
        self.in_channels = in_channels
        # 把范围和体素大小存下来，后续计算坐标要用
        self.register_buffer("pc_range", torch.tensor(point_cloud_range, dtype=torch.float32))
        self.register_buffer("voxel_size", torch.tensor(voxel_size, dtype=torch.float32))
        
        # 真正的 VFE (DynamicVFE) 本质上就是一个 Point-wise MLP
        self.vfe_mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

    def forward(self, points_list):
        """
        注意：真实场景下，不同 batch 样本的点云数量通常不一样。
        所以这里支持不定长输入：可以传入一个 List，里面每个元素都是 batch 内一个样本的点云 Tensor；
        也可以传入规整的 Tensor `(B, N_points, C)`，内部会沿 batch 维拆开后逐样本处理。
        points_list: List[torch.Tensor] or torch.Tensor
            - List 时，每个 Tensor 对应 batch 内一个样本的点云，形状为 (N_points, C)
            - Tensor 时，形状为 (B, N_points, C)
        默认 C=5，对应 [x, y, z, intensity, ring_index]
        """
        if isinstance(points_list, torch.Tensor):
            points_list = [points for points in points_list]

        device = points_list[0].device
        pc_range = self.pc_range.to(device)
        voxel_size = self.voxel_size.to(device)

        all_voxel_tokens = []
        all_voxel_coords = []
        
        # 逐 batch 样本处理点云；动态体素化通常就是按样本分别做的
        for points in points_list:
            if points.dim() != 2:
                raise ValueError("Each point cloud sample must have shape (N_points, C)")
            if points.shape[-1] != self.in_channels:
                raise ValueError(
                    f"Expected point features with {self.in_channels} channels, "
                    f"but got {points.shape[-1]}"
                )
            # ==========================================
            # 1. 坐标网格化：计算每个点属于哪个体素盒子
            # ==========================================
            # 网格索引 = 向下取整( (点坐标 - 边界最小值) / 体素大小 )
            grid_coords = torch.floor((points[:, :3] - pc_range[:3]) / voxel_size).long()
            
            # 过滤越界点：只保留落在 point_cloud_range 内的点
            # 使用 [min, max) 的半开区间，避免落在 max 边界上的点产生越界体素索引
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
                embed_dim = self.vfe_mlp[3].out_features  # Linear(64, embed_dim)
                all_voxel_tokens.append(points.new_zeros((0, embed_dim)))
                all_voxel_coords.append(points.new_zeros((0, 3)))
                continue
            
            # ==========================================
            # 2. 动态聚合：找到不重复的体素
            # ==========================================
            # torch.unique 会找出点云占据了哪些体素 (unq_coords)
            # 以及每个原始点对应哪个唯一体素 (inverse_indices)
            unq_coords, inverse_indices = torch.unique(grid_coords, dim=0, return_inverse=True)
            num_voxels = unq_coords.shape[0]
            
            # ==========================================
            # 3. 提取点特征并散射(Scatter)到体素中
            # ==========================================
            # 先对所有散乱的点过 MLP (N_points, embed_dim)
            point_feats = self.vfe_mlp(points)
            
            # 创建空的体素特征槽 (N_voxels, embed_dim)
            voxel_feats = point_feats.new_zeros((num_voxels, point_feats.shape[-1]))
            
            # 把每个点的特征，按 inverse_indices 丢进对应的体素槽里，取最大值(Max Pooling)
            # 这是 PyTorch 1.12 之后引入的原生 scatter_reduce 操作
            scatter_indices = inverse_indices.unsqueeze(1).expand(-1, point_feats.shape[-1])
            voxel_feats.scatter_reduce_(0, scatter_indices, point_feats, reduce="amax", include_self=False)
            
            # ==========================================
            # 4. 【核心联动】计算体素的真实 3D 物理坐标
            # ==========================================
            # 这里完美呼应了我们在图像分支写的 stride // 2！
            # 网格索引 * 体素大小 + 半个体素大小(感受野中心) + 边界平移量
            real_physical_coords = unq_coords.float() * voxel_size + (voxel_size / 2.0) + pc_range[:3]
            
            # 将当前样本的 voxel token 和坐标存起来
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

# ==========================================
# 3. 核心: 几何感知序列化 (空间排序)
# ==========================================
class GeometrySerializer(nn.Module):
    def __init__(self, grid_size_3d=1.0, grid_size_2d=16.0):
        super().__init__()
        # 3D 物理空间的网格大小 (比如 1 米一个格子)
        self.grid_size_3d = grid_size_3d
        # 2D 像素平面的网格大小 (比如 16x16 像素一个格子，也就是一个 Patch 的大小)
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

        img_coords_homo = torch.matmul(
            K.unsqueeze(2),
            cam_coords.unsqueeze(-1),
        ).squeeze(-1)

        depth = torch.clamp(img_coords_homo[..., 2:3], min=1e-5)
        uv = img_coords_homo[..., :2] / depth
        uv = self.apply_post_transform(
            uv,
            post_rots.unsqueeze(2),
            post_trans.unsqueeze(2),
        )

        camera_ids = torch.arange(num_views, device=lidar_coords.device, dtype=uv.dtype)
        camera_ids = camera_ids.view(1, num_views, 1, 1).expand(B, -1, N_lidar, -1)
        return torch.cat([camera_ids, uv], dim=-1)

    def unproject_image_to_3d(self, img_kuvd, K, T_c2w, post_rots=None, post_trans=None):
        """
        【模式：2D -> 3D】反投影：将 2D 像素和深度还原为 3D 物理坐标
        T_c2w: 相机到世界 (Camera to World) 的外参矩阵
        """
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

        return world_coords_homo[..., :3] # 返回 [x, y, z]

    def project_lidar_to_2d(self, lidar_coords, K, T_c2w, camera_idx=0, post_rots=None, post_trans=None):
        """
        【模式：3D -> 2D】正向投影：将 3D 雷达坐标投影到相机的 2D 像素平面
        lidar_coords: (B, N_lidar, 3) 真实的 3D [x, y, z]
        T_c2w: 相机到世界的外参，我们需要求逆变成 T_w2c (世界到相机)
        camera_idx: 当前相机的编号 k (多视角情况下的标识)
        """
        K, T_c2w = self.normalize_camera_params(K, T_c2w)
        post_rots, post_trans = self.normalize_post_transforms(post_rots, post_trans, K)
        K = K[:, camera_idx]
        T_c2w = T_c2w[:, camera_idx]
        post_rot = post_rots[:, camera_idx]
        post_tran = post_trans[:, camera_idx]

        # 1. 获取 世界 -> 相机 的外参矩阵 (求逆)
        T_w2c = torch.inverse(T_c2w)
        
        # 2. 构造齐次 3D 坐标 [x, y, z, 1]
        x, y, z = lidar_coords[..., 0], lidar_coords[..., 1], lidar_coords[..., 2]
        xyz1 = torch.stack([x, y, z, torch.ones_like(x)], dim=-1) # (B, N_lidar, 4)
        
        # 3. 乘以外参，转换到相机 3D 坐标系
        cam_coords_homo = torch.bmm(T_w2c, xyz1.transpose(1, 2)).transpose(1, 2)
        cam_coords = cam_coords_homo[..., :3] # [X_c, Y_c, Z_c]
        
        # 4. 乘以内参，转换到图像齐次坐标
        img_coords_homo = torch.bmm(K, cam_coords.transpose(1, 2)).transpose(1, 2)
        
        # 5. 透视除法：除以深度 Z_c 得到 [u, v]
        depth = img_coords_homo[..., 2:3]
        # 【工程防坑】雷达点可能在相机背后 (Z_c < 0)，除以负数会导致坐标翻转
        # 强制把深度 clamp 到一个小正数，避免除零和翻转，排在画面外围
        depth = torch.clamp(depth, min=1e-5) 
        uv = img_coords_homo[..., :2] / depth # (B, N_lidar, 2)
        uv = self.apply_post_transform(uv, post_rot.unsqueeze(1), post_tran.unsqueeze(1))
        
        # 6. 把相机编号 k 作为第一维拼进去，变成 [k, u, v]
        # 这样就能完美复用我们写的 [d0, d1, d2] 3D 蛇形排序！
        k_tensor = torch.full_like(uv[..., 0:1], camera_idx)
        kuv = torch.cat([k_tensor, uv], dim=-1) # (B, N_lidar, 3)
        
        return kuv

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
        mode='3d',
        lidar_padding_mask=None,
        img_padding_mask=None,
    ):
        """
        mode: '3d' 表示执行体素空间排序；'2d' 表示执行图像平面排序
        """
        B, N_lidar, C = lidar_tokens.shape
        _, N_img, _ = img_tokens.shape
        K, T_c2w = self.normalize_camera_params(K, T_c2w)
        if lidar_padding_mask is None:
            lidar_padding_mask = torch.zeros((B, N_lidar), dtype=torch.bool, device=lidar_tokens.device)
        if img_padding_mask is None:
            img_padding_mask = torch.zeros((B, N_img), dtype=torch.bool, device=img_tokens.device)

        if mode == '3d':
            # === 3D 排序模式：图像找雷达 ===
            img_coords_3d = self.unproject_image_to_3d(img_kuvd, K, T_c2w, post_rots, post_trans) # 得到 [x, y, z]
            serialized_lidar_tokens = lidar_tokens
            serialized_lidar_padding = lidar_padding_mask
            unified_coords = torch.cat([lidar_coords, img_coords_3d], dim=1)
            unified_tokens = torch.cat([serialized_lidar_tokens, img_tokens], dim=1)
            unified_padding_mask = torch.cat([serialized_lidar_padding, img_padding_mask], dim=1)
            grid_size = self.grid_size_3d
            num_serialized_lidar = N_lidar
            num_views = 1
        elif mode == '2d':
            # === 2D 排序模式：雷达找图像 ===
            num_views = K.shape[1]
            lidar_coords_2d = self.project_lidar_to_all_views(lidar_coords, K, T_c2w, post_rots, post_trans)
            serialized_lidar_tokens = (
                lidar_tokens.unsqueeze(1)
                .expand(-1, num_views, -1, -1)
                .reshape(B, num_views * N_lidar, C)
            )
            serialized_lidar_padding = (
                lidar_padding_mask.unsqueeze(1)
                .expand(-1, num_views, -1)
                .reshape(B, num_views * N_lidar)
            )
            # 将 img_kuvd [k, u, v, D] 转换为 [k, u, v] 以便对齐格式
            img_kuv = img_kuvd[..., :3]
            unified_coords = torch.cat([lidar_coords_2d.reshape(B, num_views * N_lidar, 3), img_kuv], dim=1)
            unified_tokens = torch.cat([serialized_lidar_tokens, img_tokens], dim=1)
            unified_padding_mask = torch.cat([serialized_lidar_padding, img_padding_mask], dim=1)
            grid_size = self.grid_size_2d
            num_serialized_lidar = num_views * N_lidar
        else:
            raise ValueError("mode must be '2d' or '3d'")

        # ==========================================
        # 核心排序引擎 (完全复用，不管是 [x,y,z] 还是 [k,u,v] 都适用)
        # ==========================================
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


class FlashWindowBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, window_size=256, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.use_flash = flash_attn_varlen_func is not None

        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.fallback_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        hidden_dim = embed_dim * mlp_ratio
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def get_flash_window_size(self):
        if self.window_size is None or self.window_size <= 0:
            return (-1, -1)
        radius = self.window_size // 2
        return (radius, radius)

    def apply_fallback_attention(self, x, padding_mask):
        outputs = []
        seq_len = x.shape[1]
        for start in range(0, seq_len, self.window_size):
            end = min(seq_len, start + self.window_size)
            chunk = x[:, start:end, :]
            chunk_padding_mask = None if padding_mask is None else padding_mask[:, start:end]

            if chunk_padding_mask is not None and torch.all(chunk_padding_mask):
                outputs.append(torch.zeros_like(chunk))
                continue

            chunk_out, _ = self.fallback_attn(
                chunk,
                chunk,
                chunk,
                key_padding_mask=chunk_padding_mask,
                need_weights=False,
            )
            outputs.append(chunk_out)

        return torch.cat(outputs, dim=1)

    def apply_flash_attention(self, x, padding_mask):
        B, S, C = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        flash_dtype = qkv.dtype if qkv.dtype in (torch.float16, torch.bfloat16) else torch.float16
        qkv = qkv.to(flash_dtype)
        q, k, v = qkv.unbind(dim=2)

        if padding_mask is None:
            valid_mask = torch.ones((B, S), dtype=torch.bool, device=x.device)
        else:
            valid_mask = ~padding_mask

        lengths = valid_mask.sum(dim=1).to(torch.int32)
        cu_seqlens = torch.cat(
            [torch.zeros(1, device=x.device, dtype=torch.int32), lengths.cumsum(0)]
        ).to(torch.int32)
        max_seqlen = int(lengths.max().item()) if lengths.numel() > 0 else 0

        packed_q = q[valid_mask]
        packed_k = k[valid_mask]
        packed_v = v[valid_mask]

        if packed_q.shape[0] == 0:
            context = q.new_zeros((B, S, self.num_heads, self.head_dim))
        else:
            packed_out = flash_attn_varlen_func(
                packed_q,
                packed_k,
                packed_v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                dropout_p=0.0,
                causal=False,
                window_size=self.get_flash_window_size(),
            )
            context = q.new_zeros((B, S, self.num_heads, self.head_dim))
            context[valid_mask] = packed_out

        return self.proj(context.reshape(B, S, C).to(x.dtype))

    def forward(self, x, padding_mask=None):
        residual = x
        x_norm = self.norm1(x)
        if self.use_flash and x.is_cuda:
            attn_out = self.apply_flash_attention(x_norm, padding_mask)
        else:
            attn_out = self.apply_fallback_attention(x_norm, padding_mask)
        x = residual + attn_out
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)

        x = x + self.mlp(self.norm2(x))
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
        return x


# ==========================================
# 4. LLM 风格的融合骨干网络 (3 层)
# ==========================================
class FusionNeXtBackbone(nn.Module):
    def __init__(self, embed_dim=256, num_layers=4, window_size=256): 
        # 这里默认设为 4 层，方便演示交替
        super().__init__()
        self.num_layers = num_layers
        self.window_size = window_size
        
        # 使用我们刚刚写好的终极版 Serializer
        self.serializer = GeometrySerializer()
        
        # 定义每一层的排序模式，这里采用论文实验中的严格交替：['2d', '3d', '2d', '3d']
        self.layer_modes = ['2d' if i % 2 == 0 else '3d' for i in range(num_layers)]
        
        self.blocks = nn.ModuleList([
            FlashWindowBlock(
                embed_dim=embed_dim,
                num_heads=8,
                mlp_ratio=4,
                window_size=window_size,
            )
            for _ in range(num_layers)
        ])

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
        # 维护当前特征的最新状态
        current_lidar_tokens = lidar_tokens
        current_img_tokens = img_tokens
        if lidar_padding_mask is None:
            lidar_padding_mask = torch.zeros((B, N_lidar), dtype=torch.bool, device=lidar_tokens.device)
        current_lidar_padding_mask = lidar_padding_mask
        current_img_padding_mask = torch.zeros((B, N_img), dtype=torch.bool, device=img_tokens.device)
        
        for i in range(self.num_layers):
            mode = self.layer_modes[i]
            
            # ==========================================
            # 步骤 A：获取当前模式下的序列和排序索引
            # ==========================================
            sorted_tokens, sorted_indices, sorted_padding_mask, num_serialized_lidar, num_views = self.serializer(
                current_lidar_tokens, lidar_coords, 
                current_img_tokens, img_kuvd, 
                K,
                T_c2w,
                post_rots=post_rots,
                post_trans=post_trans,
                mode=mode,
                lidar_padding_mask=current_lidar_padding_mask,
                img_padding_mask=current_img_padding_mask,
            )
            # ==========================================
            # 步骤 B：执行局部滑动窗口注意力交互
            # 此时的 sorted_tokens 已经完美具备了 2D 或 3D 的局部性
            # ==========================================
            attended_tokens = self.blocks[i](sorted_tokens, padding_mask=sorted_padding_mask)
            
            # ==========================================
            # 步骤 C：核心魔法 —— 解除排序，恢复原生顺序！
            # 对 sorted_indices 再次求 argsort，就能得到逆向恢复的索引
            # ==========================================
            inverse_indices = torch.argsort(sorted_indices, dim=1)
            expanded_inv_indices = inverse_indices.unsqueeze(-1).expand(-1, -1, C)
            
            # 将交互后的特征瞬间“洗回”原始的队伍顺序
            recovered_tokens = torch.gather(attended_tokens, 1, expanded_inv_indices)
            recovered_padding_mask = torch.gather(sorted_padding_mask, 1, inverse_indices)

            recovered_lidar_tokens = recovered_tokens[:, :num_serialized_lidar, :]
            recovered_lidar_padding_mask = recovered_padding_mask[:, :num_serialized_lidar]
            current_img_tokens = recovered_tokens[:, num_serialized_lidar:, :]
            current_img_padding_mask = recovered_padding_mask[:, num_serialized_lidar:]

            if mode == '2d' and num_views > 1:
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
            
            # ==========================================
            # 步骤 D：重新拆分回 LiDAR 和 Image 集合，为下一层的新模式排队做准备
            # ==========================================
            current_lidar_tokens = current_lidar_tokens[:, :N_lidar, :]
            current_lidar_padding_mask = current_lidar_padding_mask[:, :N_lidar]

        # 最终输出：将融合了多次 2D/3D 信息的特征拼接，丢给下游任务头
        final_unified_tokens = torch.cat([current_lidar_tokens, current_img_tokens], dim=1)
        
        return final_unified_tokens

# ==========================================
# 5. 顶层组装: 端到端前向传播
# ==========================================
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
        self.fusion_backbone = FusionNeXtBackbone(embed_dim, num_layers=3)

    def forward(self, img, voxels, K, T_c2w, post_rots=None, post_trans=None, return_dict=False):
        if K.dim() == 4 and K.shape[1] != img.shape[1]:
            raise ValueError("K must provide one intrinsic matrix per input camera view")
        if T_c2w.dim() == 4 and T_c2w.shape[1] != img.shape[1]:
            raise ValueError("T_c2w must provide one extrinsic matrix per input camera view")

        # 1. 各自提取特征
        img_tokens, img_coords, img_depths = self.image_branch(img)
        voxel_tokens, voxel_coords, voxel_padding_mask = self.lidar_branch(voxels)
        img_kuvd = torch.cat([img_coords, img_depths], dim=-1)
        
        # 2. 进入融合主干网络
        final_1d_sequence = self.fusion_backbone(
            voxel_tokens, voxel_coords,
            img_tokens, img_kuvd,
            K, T_c2w,
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


@register_detector_module
class FusionNeXt(Base3DDetector):
    def __init__(
        self,
        point_cloud_range,
        voxel_size,
        embed_dim=256,
        image_weights=None,
        lidar_in_channels=5,
        bbox_head=None,
        train_cfg=None,
        test_cfg=None,
    ):
        super().__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if bbox_head is None:
            bbox_head = dict(
                type="FusionNeXtSimple3DHead",
                num_classes=10,
                in_channels=embed_dim,
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range,
            )
        self.core = FusionNeXtMini(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            embed_dim=embed_dim,
            image_weights=image_weights,
            lidar_in_channels=lidar_in_channels,
        )
        self.bbox_head = self.build_bbox_head(bbox_head, embed_dim, voxel_size, point_cloud_range)

    def init_weights(self):
        # The image backbone is already initialized by torchvision. Only
        # initialize newly added linear layers here and skip verbose BaseModule logs.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def build_bbox_head(self, bbox_head_cfg, embed_dim, voxel_size, point_cloud_range):
        if isinstance(bbox_head_cfg, nn.Module):
            return bbox_head_cfg
        if not isinstance(bbox_head_cfg, dict):
            raise TypeError(f"bbox_head must be a dict or nn.Module, but got {type(bbox_head_cfg)}")

        cfg = dict(bbox_head_cfg)
        cfg.setdefault("in_channels", embed_dim)
        cfg.setdefault("voxel_size", voxel_size)
        cfg.setdefault("point_cloud_range", point_cloud_range)

        head_type = cfg.pop("type", "FusionNeXtSimple3DHead")
        if build_head is not None:
            return build_head(dict(type=head_type, **cfg))
        if head_type != "FusionNeXtSimple3DHead":
            raise ValueError(f"Unsupported bbox head without OpenMMLab registry: {head_type}")
        return FusionNeXtSimple3DHead(**cfg)

    def unpack_img_inputs(self, img_inputs):
        if not isinstance(img_inputs, (list, tuple)):
            raise TypeError(f"img_inputs must be a tuple/list, but got {type(img_inputs)}")
        if len(img_inputs) not in (6, 7):
            raise ValueError(f"Expected img_inputs with 6 or 7 items, but got {len(img_inputs)}")

        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans = img_inputs[:6]
        bda = img_inputs[6] if len(img_inputs) == 7 else None
        return imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda

    def get_lidar_to_global(self, img_metas, device, dtype):
        lidar_to_global = []
        for meta in img_metas:
            fusion_meta = meta.get("fusionnext_meta")
            if fusion_meta is None:
                raise KeyError(
                    "img_metas must contain 'fusionnext_meta'. "
                    "Use the helper pipeline in nuscenes_pipeline.py or include this meta manually."
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

    def prepare_fusion_inputs(self, points, img_inputs, img_metas):
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = self.unpack_img_inputs(img_inputs)
        device = imgs.device
        dtype = imgs.dtype

        if not isinstance(points, list):
            raise TypeError(f"points must be a list[Tensor] in OpenMMLab mode, but got {type(points)}")

        sensor2egos = sensor2egos.to(device=device, dtype=dtype)
        ego2globals = ego2globals.to(device=device, dtype=dtype)
        intrins = intrins.to(device=device, dtype=dtype)
        post_rots = post_rots.to(device=device, dtype=dtype)
        post_trans = post_trans.to(device=device, dtype=dtype)

        lidar_to_global = self.get_lidar_to_global(img_metas, device=device, dtype=dtype)
        global_to_lidar = torch.inverse(lidar_to_global).unsqueeze(1)
        cam_to_global = ego2globals @ sensor2egos
        cam_to_lidar = global_to_lidar @ cam_to_global

        if bda is not None:
            bda = bda.to(device=device, dtype=dtype)
            cam_to_lidar = bda.unsqueeze(1) @ cam_to_lidar

        return imgs, points, intrins, cam_to_lidar, post_rots, post_trans

    def extract_feat(self, points, img_inputs, img_metas, **kwargs):
        img, voxels, K, T_c2l, post_rots, post_trans = self.prepare_fusion_inputs(
            points,
            img_inputs,
            img_metas,
        )
        return self.core(
            img,
            voxels,
            K,
            T_c2l,
            post_rots=post_rots,
            post_trans=post_trans,
            return_dict=True,
        )

    def forward_train(
        self,
        points=None,
        img_metas=None,
        img_inputs=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        del kwargs
        features = self.extract_feat(points, img_inputs, img_metas)
        preds = self.bbox_head(
            features["fusion_tokens"],
            features["num_lidar_tokens"],
            features["lidar_coords"],
            features["unified_padding_mask"],
            features["lidar_padding_mask"],
        )
        return self.bbox_head.loss(
            preds,
            features["lidar_coords"],
            features["lidar_padding_mask"],
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
        )

    def simple_test(self, points, img_metas, img_inputs=None, **kwargs):
        del kwargs
        features = self.extract_feat(points, img_inputs, img_metas)
        preds = self.bbox_head(
            features["fusion_tokens"],
            features["num_lidar_tokens"],
            features["lidar_coords"],
            features["unified_padding_mask"],
            features["lidar_padding_mask"],
        )
        pred_instances = self.bbox_head.predict(
            preds,
            features["lidar_coords"],
            features["lidar_padding_mask"],
            img_metas=img_metas,
        )
        return [{"pts_bbox": result} for result in pred_instances]

    def aug_test(self, points, img_metas, img_inputs=None, **kwargs):
        raise NotImplementedError("FusionNeXt does not support test-time augmentation yet.")

    def forward_dummy(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        return self.extract_feat(points, img_inputs, img_metas, **kwargs)

# === 运行测试区 ===
if __name__ == "__main__":
    print("正在构建 FusionNeXt Mini 模型...")
    point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    voxel_size = [0.2, 0.2, 0.4]
    model = FusionNeXtMini(
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        embed_dim=256,
        image_weights=None,
        lidar_in_channels=5,
    )
    
    # 构造假数据
    B = 1
    N_views = 6
    H, W = 128, 128
    N_voxels = 256
    
    dummy_img = torch.randn(B, N_views, 3, H, W)
    dummy_voxels = torch.randn(B, N_voxels, 5)
    dummy_K = torch.eye(3).view(1, 1, 3, 3).repeat(B, N_views, 1, 1)
    dummy_T_c2w = torch.eye(4).view(1, 1, 4, 4).repeat(B, N_views, 1, 1)
    
    print(f"输入图像维度: {dummy_img.shape}")
    print(f"输入体素维度: {dummy_voxels.shape}")
    
    # 前向传播
    with torch.no_grad(): # 节省内存
        output_sequence = model(dummy_img, dummy_voxels, dummy_K, dummy_T_c2w)
        
    print(f"\n✅ 前向传播成功！")
    print(f"最终输出的 1D Token 序列维度: {output_sequence.shape}")
    print(f"(说明: Batch={B}, 序列长度=图像Tokens+体素Tokens, 特征维度=256)")
