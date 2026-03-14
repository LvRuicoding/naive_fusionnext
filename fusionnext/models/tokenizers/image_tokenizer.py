import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ImageTokenizer(nn.Module):
    def __init__(self, embed_dim=256, image_weights=None, patch_size=8):
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, but got {patch_size}")
        if image_weights == "default":
            image_weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=image_weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.patch_size = patch_size
        self.proj = nn.Linear(2048, embed_dim)
        self.depth_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
        )

    def forward(self, img):
        B, N_views, C, H, W = img.shape
        img_reshaped = img.reshape(B * N_views, C, H, W)

        features = self.backbone(img_reshaped)
        target_h = max(H // self.patch_size, 1)
        target_w = max(W // self.patch_size, 1)
        if features.shape[-2:] != (target_h, target_w):
            features = F.interpolate(features, size=(target_h, target_w), mode="bilinear", align_corners=False)
        _, C_feat, H_feat, W_feat = features.shape

        patch_tokens = features.flatten(2).permute(0, 2, 1)
        patch_tokens = patch_tokens.reshape(B, N_views * H_feat * W_feat, C_feat)

        image_tokens = self.proj(patch_tokens)
        N_t = image_tokens.shape[1]

        stride_h = self.patch_size
        stride_w = self.patch_size

        v_coords = torch.arange(H_feat, device=img.device) * stride_h + (stride_h // 2)
        u_coords = torch.arange(W_feat, device=img.device) * stride_w + (stride_w // 2)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")

        v_flat = v_grid.flatten()
        u_flat = u_grid.flatten()

        all_views_coords = []
        for k in range(N_views):
            k_tensor = torch.full_like(u_flat, k)
            kuv = torch.stack([k_tensor, u_flat, v_flat], dim=-1)
            all_views_coords.append(kuv)

        image_coords = torch.cat(all_views_coords, dim=0)
        image_coords = image_coords.unsqueeze(0).repeat(B, 1, 1).float()

        depth_map = self.depth_head(features)
        depth_map = F.softplus(depth_map)
        depth_values = depth_map.reshape(B, N_t, 1)
        return image_tokens, image_coords, depth_values
