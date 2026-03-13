from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from pyquaternion import Quaternion

try:
    from mmdet3d.datasets.builder import PIPELINES
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    from mmdet3d.datasets.pipelines.loading import PrepareImageInputs
except ImportError as exc:  # pragma: no cover - intended for non-OpenMMLab envs
    raise ImportError(
        "nuscenes_pipeline.py requires the OpenMMLab BEVDet/mmdet3d environment."
    ) from exc


FUSIONNEXT_META_KEYS = (
    "filename",
    "ori_shape",
    "img_shape",
    "lidar2img",
    "depth2img",
    "cam2img",
    "pad_shape",
    "scale_factor",
    "flip",
    "pcd_horizontal_flip",
    "pcd_vertical_flip",
    "box_mode_3d",
    "box_type_3d",
    "img_norm_cfg",
    "pcd_trans",
    "sample_idx",
    "pcd_scale_factor",
    "pcd_rotation",
    "pcd_rotation_angle",
    "pts_filename",
    "transformation_3d_flow",
    "trans_mat",
    "affine_aug",
    "fusionnext_meta",
)


@PIPELINES.register_module()
class FusionNeXtPrepareMeta:
    """Attach compact lidar pose metadata for the FusionNeXt OpenMMLab wrapper."""

    def __call__(self, results: Dict) -> Dict:
        curr = results.get("curr")
        if curr is None:
            raise KeyError(
                "FusionNeXtPrepareMeta expects results['curr']. "
                "Set dataset img_info_prototype='bevdet'."
            )

        results["fusionnext_meta"] = {
            "lidar2ego_translation": curr["lidar2ego_translation"],
            "lidar2ego_rotation": curr["lidar2ego_rotation"],
            "ego2global_translation": curr["ego2global_translation"],
            "ego2global_rotation": curr["ego2global_rotation"],
        }
        return results


@PIPELINES.register_module()
class FusionNeXtPrepareImageInputs(PrepareImageInputs):
    """BEVDet image-input preparation that supports sensor2lidar-based infos.

    The local nuScenes-mini info files store camera extrinsics as
    ``sensor2lidar_*`` plus top-level ``lidar2ego_*`` and ``ego2global_*``.
    This wrapper converts them into the ``sensor2ego`` / ``ego2global`` pair
    expected by the original BEVDet image loading pipeline.
    """

    @staticmethod
    def _build_transform(rotation, translation) -> torch.Tensor:
        transform = torch.eye(4, dtype=torch.float32)
        rotation_array = np.asarray(rotation, dtype=np.float32)
        if rotation_array.shape == (3, 3):
            transform[:3, :3] = torch.from_numpy(rotation_array)
        else:
            quat = Quaternion(rotation_array.tolist())
            transform[:3, :3] = torch.from_numpy(quat.rotation_matrix.astype(np.float32))
        transform[:3, 3] = torch.as_tensor(translation, dtype=torch.float32)
        return transform

    def get_sensor_transforms(self, cam_info, cam_name):
        cam_record = cam_info["cams"][cam_name]
        if (
            "sensor2ego_rotation" in cam_record
            and "sensor2ego_translation" in cam_record
            and "ego2global_rotation" in cam_record
            and "ego2global_translation" in cam_record
        ):
            return super().get_sensor_transforms(cam_info, cam_name)

        if "sensor2lidar_rotation" not in cam_record or "sensor2lidar_translation" not in cam_record:
            raise KeyError(
                f"Camera record for {cam_name} does not contain supported extrinsic keys. "
                "Expected sensor2ego_* or sensor2lidar_* entries."
            )

        sensor2lidar = self._build_transform(
            cam_record["sensor2lidar_rotation"],
            cam_record["sensor2lidar_translation"],
        )
        lidar2ego = self._build_transform(
            cam_info["lidar2ego_rotation"],
            cam_info["lidar2ego_translation"],
        )
        ego2global = self._build_transform(
            cam_info["ego2global_rotation"],
            cam_info["ego2global_translation"],
        )
        sensor2ego = lidar2ego @ sensor2lidar
        return sensor2ego, ego2global


@PIPELINES.register_module()
class FusionNeXtLoadAnnotations:
    """Load nuScenes annotations directly from BEVDet-style info dicts."""

    def __init__(self, classes: Sequence[str], with_velocity: bool = True, use_valid_flag: bool = False):
        self.classes = tuple(classes)
        self.with_velocity = with_velocity
        self.use_valid_flag = use_valid_flag

    def __call__(self, results: Dict) -> Dict:
        info = results.get("curr")
        if info is None:
            raise KeyError(
                "FusionNeXtLoadAnnotations expects results['curr']. "
                "Set dataset img_info_prototype='bevdet'."
            )

        if self.use_valid_flag and "valid_flag" in info:
            mask = np.asarray(info["valid_flag"]).astype(bool)
        else:
            mask = np.asarray(info["num_lidar_pts"]) > 0

        gt_boxes = np.asarray(info["gt_boxes"], dtype=np.float32)[mask]
        gt_names = np.asarray(info["gt_names"], dtype=object)[mask]
        gt_labels = np.array(
            [self.classes.index(name) if name in self.classes else -1 for name in gt_names],
            dtype=np.int64,
        )

        if self.with_velocity and "gt_velocity" in info:
            gt_velocity = np.asarray(info["gt_velocity"], dtype=np.float32)[mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_boxes = np.concatenate([gt_boxes, gt_velocity], axis=-1)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_boxes,
            box_dim=gt_boxes.shape[-1] if gt_boxes.ndim == 2 else (9 if self.with_velocity else 7),
            origin=(0.5, 0.5, 0.5),
        )
        results["gt_bboxes_3d"] = gt_bboxes_3d
        results["gt_labels_3d"] = gt_labels
        results["gt_names"] = gt_names
        return results


@PIPELINES.register_module()
class FusionNeXtObjectNameFilter:
    """Filter GT boxes, labels and names together by class whitelist."""

    def __init__(self, classes: Sequence[str]):
        self.classes = tuple(classes)
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict: Dict) -> Dict:
        gt_labels_3d = np.asarray(input_dict["gt_labels_3d"])
        gt_bboxes_mask = np.array([label in self.labels for label in gt_labels_3d], dtype=np.bool_)
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = gt_labels_3d[gt_bboxes_mask]
        if "gt_names" in input_dict:
            input_dict["gt_names"] = np.asarray(input_dict["gt_names"], dtype=object)[gt_bboxes_mask]
        if "gt_bboxes_ignore" in input_dict:
            input_dict["gt_bboxes_ignore"] = input_dict["gt_bboxes_ignore"][gt_bboxes_mask]
        return input_dict


@PIPELINES.register_module()
class FusionNeXtObjectRangeFilter:
    """Object range filter that keeps gt_names aligned with boxes and labels."""

    def __init__(self, point_cloud_range: Sequence[float]):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict: Dict) -> Dict:
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = np.asarray(input_dict["gt_labels_3d"])

        bev_range = self.pcd_range[[0, 1, 3, 4]]
        mask = gt_bboxes_3d.in_range_bev(bev_range).numpy().astype(np.bool_)

        input_dict["gt_bboxes_3d"] = gt_bboxes_3d[mask]
        input_dict["gt_labels_3d"] = gt_labels_3d[mask]
        if "gt_names" in input_dict:
            input_dict["gt_names"] = np.asarray(input_dict["gt_names"], dtype=object)[mask]
        if "gt_bboxes_ignore" in input_dict:
            input_dict["gt_bboxes_ignore"] = input_dict["gt_bboxes_ignore"][mask]

        input_dict["gt_bboxes_3d"].limit_yaw(offset=0.5, period=2 * np.pi)
        return input_dict


def build_fusionnext_train_pipeline(
    data_config: Dict,
    class_names: Sequence[str],
    point_cloud_range: Sequence[float],
    bda_aug_conf: Dict,
    load_dim: int = 5,
    use_dim: int = 5,
    file_client_args: Optional[Dict] = None,
) -> list:
    if file_client_args is None:
        file_client_args = dict(backend="disk")

    return [
        dict(type="FusionNeXtPrepareImageInputs", is_train=True, data_config=data_config),
        dict(
            type="LoadPointsFromFile",
            coord_type="LIDAR",
            load_dim=load_dim,
            use_dim=use_dim,
            file_client_args=file_client_args,
        ),
        dict(type="FusionNeXtLoadAnnotations", classes=class_names),
        dict(type="BEVAug", bda_aug_conf=bda_aug_conf, classes=class_names),
        dict(type="FusionNeXtObjectRangeFilter", point_cloud_range=point_cloud_range),
        dict(type="FusionNeXtObjectNameFilter", classes=class_names),
        dict(type="FusionNeXtPrepareMeta"),
        dict(type="DefaultFormatBundle3D", class_names=class_names),
        dict(
            type="Collect3D",
            keys=["points", "img_inputs", "gt_bboxes_3d", "gt_labels_3d"],
            meta_keys=FUSIONNEXT_META_KEYS,
        ),
    ]


def build_fusionnext_test_pipeline(
    data_config: Dict,
    class_names: Sequence[str],
    bda_aug_conf: Dict,
    load_dim: int = 5,
    use_dim: int = 5,
    file_client_args: Optional[Dict] = None,
) -> list:
    if file_client_args is None:
        file_client_args = dict(backend="disk")

    return [
        dict(type="FusionNeXtPrepareImageInputs", data_config=data_config),
        dict(
            type="LoadPointsFromFile",
            coord_type="LIDAR",
            load_dim=load_dim,
            use_dim=use_dim,
            file_client_args=file_client_args,
        ),
        dict(type="FusionNeXtLoadAnnotations", classes=class_names),
        dict(type="BEVAug", bda_aug_conf=bda_aug_conf, classes=class_names, is_train=False),
        dict(type="FusionNeXtPrepareMeta"),
        dict(
            type="MultiScaleFlipAug3D",
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
                dict(
                    type="Collect3D",
                    keys=["points", "img_inputs"],
                    meta_keys=FUSIONNEXT_META_KEYS,
                ),
            ],
        ),
    ]


def build_fusionnext_dataset_cfg(
    ann_file: str,
    data_root: str,
    data_config: Dict,
    class_names: Sequence[str],
    point_cloud_range: Sequence[float],
    bda_aug_conf: Dict,
    test_mode: bool = False,
    load_dim: int = 5,
    use_dim: int = 5,
    file_client_args: Optional[Dict] = None,
) -> Dict:
    pipeline = (
        build_fusionnext_test_pipeline(
            data_config=data_config,
            class_names=class_names,
            bda_aug_conf=bda_aug_conf,
            load_dim=load_dim,
            use_dim=use_dim,
            file_client_args=file_client_args,
        )
        if test_mode
        else build_fusionnext_train_pipeline(
            data_config=data_config,
            class_names=class_names,
            point_cloud_range=point_cloud_range,
            bda_aug_conf=bda_aug_conf,
            load_dim=load_dim,
            use_dim=use_dim,
            file_client_args=file_client_args,
        )
    )

    return dict(
        type="NuScenesDataset",
        data_root=data_root,
        ann_file=ann_file,
        pipeline=pipeline,
        classes=class_names,
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False,
        ),
        img_info_prototype="bevdet",
        test_mode=test_mode,
        box_type_3d="LiDAR",
    )


def build_fusionnext_model_cfg(
    point_cloud_range: Sequence[float],
    voxel_size: Sequence[float],
    embed_dim: int = 256,
    image_weights: Optional[str] = None,
    lidar_in_channels: int = 5,
    bbox_head: Optional[Dict] = None,
    train_cfg: Optional[Dict] = None,
    test_cfg: Optional[Dict] = None,
) -> Dict:
    model_cfg = dict(
        type="FusionNeXt",
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        embed_dim=embed_dim,
        image_weights=image_weights,
        lidar_in_channels=lidar_in_channels,
        train_cfg=train_cfg,
        test_cfg=test_cfg,
    )
    if bbox_head is not None:
        model_cfg["bbox_head"] = bbox_head
    return model_cfg


def build_fusionnext_config_snippet(
    data_root: str,
    train_ann_file: str,
    val_ann_file: str,
    data_config: Dict,
    class_names: Sequence[str],
    point_cloud_range: Sequence[float],
    voxel_size: Sequence[float],
    bda_aug_conf: Dict,
    bbox_head: Optional[Dict] = None,
    train_cfg: Optional[Dict] = None,
    test_cfg: Optional[Dict] = None,
) -> Dict:
    return dict(
        model=build_fusionnext_model_cfg(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        ),
        data=dict(
            train=build_fusionnext_dataset_cfg(
                ann_file=train_ann_file,
                data_root=data_root,
                data_config=data_config,
                class_names=class_names,
                point_cloud_range=point_cloud_range,
                bda_aug_conf=bda_aug_conf,
                test_mode=False,
            ),
            val=build_fusionnext_dataset_cfg(
                ann_file=val_ann_file,
                data_root=data_root,
                data_config=data_config,
                class_names=class_names,
                point_cloud_range=point_cloud_range,
                bda_aug_conf=bda_aug_conf,
                test_mode=True,
            ),
            test=build_fusionnext_dataset_cfg(
                ann_file=val_ann_file,
                data_root=data_root,
                data_config=data_config,
                class_names=class_names,
                point_cloud_range=point_cloud_range,
                bda_aug_conf=bda_aug_conf,
                test_mode=True,
            ),
        ),
    )
