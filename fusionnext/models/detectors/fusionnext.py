import torch.nn as nn
from mmengine.structures import InstanceData

from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.registry import MODELS

from ..dense_heads import FusionNeXtSimple3DHead
from ..fusion_models import FusionNeXtMini
from ..utils import prepare_fusion_inputs


@MODELS.register_module()
class FusionNeXt(Base3DDetector):
    def __init__(
        self,
        point_cloud_range,
        voxel_size,
        embed_dim=256,
        image_weights=None,
        image_patch_size=8,
        lidar_in_channels=5,
        bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
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
            image_patch_size=image_patch_size,
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
        return MODELS.build(cfg)

    def extract_feat(self, points, img_inputs, img_metas, **kwargs):
        del kwargs
        img, voxels, intrins, cam_to_lidar, post_rots, post_trans = prepare_fusion_inputs(
            points,
            img_inputs,
            img_metas,
        )
        return self.core(
            img,
            voxels,
            intrins,
            cam_to_lidar,
            post_rots=post_rots,
            post_trans=post_trans,
            return_dict=True,
        )

    def _forward(self, inputs, data_samples=None, **kwargs):
        del kwargs
        img_metas = [] if data_samples is None else [sample.metainfo for sample in data_samples]
        features = self.extract_feat(inputs["points"], inputs["img_inputs"], img_metas)
        return self.bbox_head(
            features["fusion_tokens"],
            features["num_lidar_tokens"],
            features["lidar_coords"],
            features["unified_padding_mask"],
            features["lidar_padding_mask"],
        )

    def loss(self, inputs, data_samples, **kwargs):
        del kwargs
        img_metas = [sample.metainfo for sample in data_samples]
        features = self.extract_feat(inputs["points"], inputs["img_inputs"], img_metas)
        preds = self.bbox_head(
            features["fusion_tokens"],
            features["num_lidar_tokens"],
            features["lidar_coords"],
            features["unified_padding_mask"],
            features["lidar_padding_mask"],
        )

        gt_bboxes_3d = [sample.gt_instances_3d.bboxes_3d for sample in data_samples]
        gt_labels_3d = [sample.gt_instances_3d.labels_3d for sample in data_samples]
        return self.bbox_head.loss(
            preds,
            features["lidar_coords"],
            features["lidar_padding_mask"],
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
        )

    def predict(self, inputs, data_samples, **kwargs):
        del kwargs
        img_metas = [sample.metainfo for sample in data_samples]
        features = self.extract_feat(inputs["points"], inputs["img_inputs"], img_metas)
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

        instance_list = []
        for prediction in pred_instances:
            instance_list.append(
                InstanceData(
                    bboxes_3d=prediction["boxes_3d"],
                    scores_3d=prediction["scores_3d"],
                    labels_3d=prediction["labels_3d"],
                )
            )
        return self.add_pred_to_datasample(data_samples, data_instances_3d=instance_list)

    def aug_test(self, inputs, data_samples=None, **kwargs):
        raise NotImplementedError("FusionNeXt does not support test-time augmentation yet.")
