import torch.nn as nn

try:
    from mmdet.models import DETECTORS
    from mmdet3d.models.builder import build_head
    from mmdet3d.models.detectors.base import Base3DDetector
except ImportError:
    DETECTORS = None
    build_head = None
    Base3DDetector = nn.Module

from ..dense_heads import FusionNeXtSimple3DHead
from ..fusion_models import FusionNeXtMini
from ..utils import prepare_fusion_inputs


def register_detector_module(cls):
    if DETECTORS is None:
        return cls
    return DETECTORS.register_module()(cls)


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

    def extract_feat(self, points, img_inputs, img_metas, **kwargs):
        del kwargs
        img, voxels, K, T_c2l, post_rots, post_trans = prepare_fusion_inputs(points, img_inputs, img_metas)
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
