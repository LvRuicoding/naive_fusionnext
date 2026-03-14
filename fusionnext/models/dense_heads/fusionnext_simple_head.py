from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.structures import get_box_type
from mmdet3d.registry import MODELS

try:
    from torchvision.ops import nms as torchvision_nms
except ImportError:
    torchvision_nms = None


def _to_label_tensor(labels, device: torch.device) -> torch.Tensor:
    if labels is None:
        return torch.zeros((0,), device=device, dtype=torch.long)
    if hasattr(labels, "to"):
        return labels.to(device=device, dtype=torch.long)
    return torch.as_tensor(labels, device=device, dtype=torch.long)


def _boxes_to_tensor(boxes, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if boxes is None:
        return torch.zeros((0, 7), device=device, dtype=dtype)

    if hasattr(boxes, "tensor"):
        tensor = boxes.tensor.to(device=device, dtype=dtype)
        if hasattr(boxes, "gravity_center"):
            centers = boxes.gravity_center.to(device=device, dtype=dtype)
            dims = tensor[:, 3:6]
            yaw = tensor[:, 6:7]
            return torch.cat([centers, dims, yaw], dim=-1)
        return tensor[:, :7]

    tensor = torch.as_tensor(boxes, device=device, dtype=dtype)
    if tensor.numel() == 0:
        return tensor.reshape(0, 7)
    return tensor[:, :7]


def _points_in_boxes(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((points.shape[0], 0), dtype=torch.bool, device=points.device)

    centers = boxes[:, :3]
    dims = boxes[:, 3:6].clamp_min(1e-3)
    yaws = boxes[:, 6]

    rel = points[:, None, :] - centers[None, :, :]
    cos_yaw = torch.cos(yaws)[None, :]
    sin_yaw = torch.sin(yaws)[None, :]

    local_x = rel[..., 0] * cos_yaw + rel[..., 1] * sin_yaw
    local_y = -rel[..., 0] * sin_yaw + rel[..., 1] * cos_yaw
    local_z = rel[..., 2]

    half_dims = dims[None, :, :] * 0.5
    return (
        (local_x.abs() <= half_dims[..., 0])
        & (local_y.abs() <= half_dims[..., 1])
        & (local_z.abs() <= half_dims[..., 2])
    )


@MODELS.register_module()
class FusionNeXtSimple3DHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        hidden_channels: int = 256,
        voxel_size: Sequence[float] = (0.2, 0.2, 0.4),
        point_cloud_range: Sequence[float] = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        loss_cls_weight: float = 1.0,
        loss_bbox_weight: float = 2.0,
        loss_dir_weight: float = 0.2,
        score_thr: float = 0.1,
        nms_thr: float = 0.2,
        max_num: int = 100,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_dir_weight = loss_dir_weight
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.max_num = max_num

        self.register_buffer("voxel_size", torch.tensor(voxel_size, dtype=torch.float32))
        self.register_buffer("point_cloud_range", torch.tensor(point_cloud_range, dtype=torch.float32))

        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.pred_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Linear(hidden_channels, num_classes + 1)
        self.reg_head = nn.Linear(hidden_channels, 8)
        self.dir_head = nn.Linear(hidden_channels, 2)

    def forward(
        self,
        unified_tokens: torch.Tensor,
        num_lidar_tokens: int,
        lidar_coords: torch.Tensor,
        unified_padding_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del lidar_coords
        feats = self.shared_mlp(unified_tokens)

        if unified_padding_mask is None:
            valid_mask = torch.ones(feats.shape[:2], dtype=feats.dtype, device=feats.device)
        else:
            valid_mask = (~unified_padding_mask).to(dtype=feats.dtype)
        valid_count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        global_context = (feats * valid_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_count.unsqueeze(-1)

        lidar_feats = feats[:, :num_lidar_tokens, :]
        expanded_context = global_context.expand(-1, num_lidar_tokens, -1)
        pred_feats = self.pred_mlp(torch.cat([lidar_feats, expanded_context], dim=-1))

        cls_logits = self.cls_head(pred_feats)
        reg_preds = self.reg_head(pred_feats)
        dir_logits = self.dir_head(pred_feats)

        if padding_mask is not None:
            cls_logits = cls_logits.masked_fill(padding_mask.unsqueeze(-1), 0)
            reg_preds = reg_preds.masked_fill(padding_mask.unsqueeze(-1), 0)
            dir_logits = dir_logits.masked_fill(padding_mask.unsqueeze(-1), 0)

        return {
            "cls_logits": cls_logits,
            "reg_preds": reg_preds,
            "dir_logits": dir_logits,
        }

    def _build_targets_single(
        self,
        coords: torch.Tensor,
        valid_mask: torch.Tensor,
        gt_boxes,
        gt_labels,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = coords.device
        dtype = coords.dtype
        num_tokens = coords.shape[0]

        cls_targets = torch.full((num_tokens,), self.num_classes, device=device, dtype=torch.long)
        reg_targets = torch.zeros((num_tokens, 8), device=device, dtype=dtype)
        dir_targets = torch.zeros((num_tokens,), device=device, dtype=torch.long)
        reg_weights = torch.zeros((num_tokens,), device=device, dtype=dtype)

        cls_targets[~valid_mask] = -1
        if valid_mask.sum() == 0:
            return cls_targets, reg_targets, dir_targets, reg_weights

        boxes = _boxes_to_tensor(gt_boxes, device=device, dtype=dtype)
        labels = _to_label_tensor(gt_labels, device=device)
        if boxes.numel() == 0 or labels.numel() == 0:
            return cls_targets, reg_targets, dir_targets, reg_weights

        valid_gt = (
            (labels >= 0)
            & (labels < self.num_classes)
            & (boxes[:, 3] > 0)
            & (boxes[:, 4] > 0)
            & (boxes[:, 5] > 0)
        )
        boxes = boxes[valid_gt]
        labels = labels[valid_gt]
        if boxes.numel() == 0:
            return cls_targets, reg_targets, dir_targets, reg_weights

        token_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)
        valid_coords = coords[token_indices]
        inside = _points_in_boxes(valid_coords, boxes)
        if inside.numel() == 0:
            return cls_targets, reg_targets, dir_targets, reg_weights

        volumes = boxes[:, 3] * boxes[:, 4] * boxes[:, 5]
        volumes = volumes.unsqueeze(0).expand(valid_coords.shape[0], -1)
        assign_volumes = torch.where(inside, volumes, torch.full_like(volumes, torch.inf))
        min_volume, min_indices = assign_volumes.min(dim=1)
        positive = torch.isfinite(min_volume)
        if positive.sum() == 0:
            return cls_targets, reg_targets, dir_targets, reg_weights

        pos_token_indices = token_indices[positive]
        assigned_boxes = boxes[min_indices[positive]]
        assigned_labels = labels[min_indices[positive]]

        centers = assigned_boxes[:, :3]
        dims = assigned_boxes[:, 3:6].clamp_min(1e-3)
        yaws = assigned_boxes[:, 6]

        cls_targets[pos_token_indices] = assigned_labels
        reg_targets[pos_token_indices, :3] = (centers - coords[pos_token_indices]) / self.voxel_size.to(dtype=dtype)
        reg_targets[pos_token_indices, 3:6] = torch.log(dims)
        reg_targets[pos_token_indices, 6] = torch.sin(yaws)
        reg_targets[pos_token_indices, 7] = torch.cos(yaws)
        dir_targets[pos_token_indices] = (torch.cos(yaws) < 0).long()
        reg_weights[pos_token_indices] = 1.0
        return cls_targets, reg_targets, dir_targets, reg_weights

    def loss(
        self,
        preds: Dict[str, torch.Tensor],
        lidar_coords: torch.Tensor,
        padding_mask: torch.Tensor,
        gt_bboxes_3d: Sequence,
        gt_labels_3d: Sequence,
    ) -> Dict[str, torch.Tensor]:
        cls_logits = preds["cls_logits"]
        reg_preds = preds["reg_preds"]
        dir_logits = preds["dir_logits"]

        batch_cls_targets = []
        batch_reg_targets = []
        batch_dir_targets = []
        batch_reg_weights = []

        for batch_idx in range(lidar_coords.shape[0]):
            targets = self._build_targets_single(
                coords=lidar_coords[batch_idx],
                valid_mask=~padding_mask[batch_idx],
                gt_boxes=gt_bboxes_3d[batch_idx],
                gt_labels=gt_labels_3d[batch_idx],
            )
            batch_cls_targets.append(targets[0])
            batch_reg_targets.append(targets[1])
            batch_dir_targets.append(targets[2])
            batch_reg_weights.append(targets[3])

        cls_targets = torch.stack(batch_cls_targets, dim=0)
        reg_targets = torch.stack(batch_reg_targets, dim=0)
        dir_targets = torch.stack(batch_dir_targets, dim=0)
        reg_weights = torch.stack(batch_reg_weights, dim=0)

        cls_loss = F.cross_entropy(
            cls_logits.reshape(-1, self.num_classes + 1),
            cls_targets.reshape(-1),
            ignore_index=-1,
        )

        positive_mask = reg_weights > 0
        if positive_mask.any():
            reg_loss = F.smooth_l1_loss(
                reg_preds[positive_mask],
                reg_targets[positive_mask],
                reduction="mean",
            )
            dir_loss = F.cross_entropy(
                dir_logits[positive_mask],
                dir_targets[positive_mask],
            )
        else:
            zero = cls_logits.sum() * 0.0
            reg_loss = zero
            dir_loss = zero

        loss = (
            self.loss_cls_weight * cls_loss
            + self.loss_bbox_weight * reg_loss
            + self.loss_dir_weight * dir_loss
        )
        return {
            "loss_cls": self.loss_cls_weight * cls_loss,
            "loss_bbox": self.loss_bbox_weight * reg_loss,
            "loss_dir": self.loss_dir_weight * dir_loss,
            "loss": loss,
        }

    def decode_boxes(self, reg_preds: torch.Tensor, lidar_coords: torch.Tensor) -> torch.Tensor:
        voxel_size = self.voxel_size.to(device=reg_preds.device, dtype=reg_preds.dtype)
        centers = lidar_coords + reg_preds[..., :3] * voxel_size
        dims = torch.exp(reg_preds[..., 3:6]).clamp_min(1e-3)
        yaws = torch.atan2(reg_preds[..., 6], reg_preds[..., 7]).unsqueeze(-1)
        return torch.cat([centers, dims, yaws], dim=-1)

    def _select_predictions_single(
        self,
        boxes: torch.Tensor,
        cls_scores: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores, labels = cls_scores.max(dim=-1)
        keep = valid_mask & (scores > self.score_thr)
        if keep.sum() == 0:
            return (
                boxes.new_zeros((0, 7)),
                boxes.new_zeros((0,)),
                boxes.new_zeros((0,), dtype=torch.long),
            )

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if boxes.shape[0] > self.max_num:
            topk_scores, topk_indices = scores.topk(self.max_num)
            boxes = boxes[topk_indices]
            labels = labels[topk_indices]
            scores = topk_scores

        if torchvision_nms is None or boxes.shape[0] <= 1:
            return boxes, scores, labels

        bev_boxes = torch.stack(
            [
                boxes[:, 0] - boxes[:, 3] * 0.5,
                boxes[:, 1] - boxes[:, 4] * 0.5,
                boxes[:, 0] + boxes[:, 3] * 0.5,
                boxes[:, 1] + boxes[:, 4] * 0.5,
            ],
            dim=-1,
        )
        keep_indices = torchvision_nms(bev_boxes, scores, self.nms_thr)
        keep_indices = keep_indices[: self.max_num]
        return boxes[keep_indices], scores[keep_indices], labels[keep_indices]

    def predict(
        self,
        preds: Dict[str, torch.Tensor],
        lidar_coords: torch.Tensor,
        padding_mask: torch.Tensor,
        img_metas: Optional[Sequence[Dict]] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        decoded_boxes = self.decode_boxes(preds["reg_preds"], lidar_coords)
        cls_scores = F.softmax(preds["cls_logits"], dim=-1)[..., : self.num_classes]

        results = []
        for batch_idx in range(decoded_boxes.shape[0]):
            boxes, scores, labels = self._select_predictions_single(
                boxes=decoded_boxes[batch_idx],
                cls_scores=cls_scores[batch_idx],
                valid_mask=~padding_mask[batch_idx],
            )
            box_output = boxes.detach().cpu()
            if img_metas is not None:
                box_type = img_metas[batch_idx].get("box_type_3d")
                if isinstance(box_type, str):
                    box_type = get_box_type(box_type)[0]
                if box_type is not None and boxes.numel() > 0:
                    box_tensor = boxes.detach().cpu().clone()
                    box_tensor[:, 2] = box_tensor[:, 2] - box_tensor[:, 5] * 0.5
                    box_output = box_type(box_tensor, box_dim=7)
            results.append(
                {
                    "boxes_3d": box_output,
                    "scores_3d": scores.detach().cpu(),
                    "labels_3d": labels.detach().cpu(),
                }
            )
        return results
