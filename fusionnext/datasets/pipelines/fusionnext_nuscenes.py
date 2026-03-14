from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmcv import BaseTransform
from mmcv.image.photometric import imnormalize
from mmengine.structures import InstanceData
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.datasets.transforms.formating import Pack3DDetInputs, to_tensor
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes, PointData
from mmdet3d.structures.points import BasePoints


FUSIONNEXT_META_KEYS = (
    "sample_idx",
    "token",
    "timestamp",
    "lidar_path",
    "box_type_3d",
    "box_mode_3d",
    "fusionnext_meta",
)


def normalize_img(img: Image.Image) -> torch.Tensor:
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    array = imnormalize(np.array(img), mean, std, to_rgb=True)
    return torch.tensor(array).float().permute(2, 0, 1).contiguous()


@TRANSFORMS.register_module()
class FusionNeXtPrepareMeta(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        curr = results.get("curr")
        if curr is None:
            raise KeyError("FusionNeXtPrepareMeta expects results['curr'].")

        results["fusionnext_meta"] = {
            "lidar2ego_translation": curr["lidar2ego_translation"],
            "lidar2ego_rotation": curr["lidar2ego_rotation"],
            "ego2global_translation": curr["ego2global_translation"],
            "ego2global_rotation": curr["ego2global_rotation"],
        }
        results["token"] = curr["token"]
        return results


@TRANSFORMS.register_module()
class FusionNeXtPrepareImageInputs(BaseTransform):
    def __init__(self, data_config: Dict, is_train: bool = False):
        self.data_config = data_config
        self.is_train = is_train

    def get_rot(self, angle: float) -> torch.Tensor:
        return torch.tensor(
            [
                [np.cos(angle), np.sin(angle)],
                [-np.sin(angle), np.cos(angle)],
            ],
            dtype=torch.float32,
        )

    def choose_cams(self):
        cams = self.data_config["cams"]
        if self.is_train and self.data_config["Ncams"] < len(cams):
            return np.random.choice(cams, self.data_config["Ncams"], replace=False)
        return cams

    def sample_augmentation(self, height: int, width: int):
        final_h, final_w = self.data_config["input_size"]
        if self.is_train:
            resize = float(final_w) / float(width)
            resize += np.random.uniform(*self.data_config["resize"])
            resize_dims = (int(width * resize), int(height * resize))
            new_w, new_h = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config["crop_h"])) * new_h) - final_h
            crop_w = int(np.random.uniform(0, max(0, new_w - final_w)))
            crop = (crop_w, crop_h, crop_w + final_w, crop_h + final_h)
            flip = bool(self.data_config["flip"] and np.random.choice([0, 1]))
            rotate = float(np.random.uniform(*self.data_config["rot"]))
        else:
            resize = float(final_w) / float(width) + self.data_config.get("resize_test", 0.0)
            resize_dims = (int(width * resize), int(height * resize))
            new_w, new_h = resize_dims
            crop_h = int((1 - np.mean(self.data_config["crop_h"])) * new_h) - final_h
            crop_w = int(max(0, new_w - final_w) / 2)
            crop = (crop_w, crop_h, crop_w + final_w, crop_h + final_h)
            flip = False
            rotate = 0.0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(
        self,
        img: Image.Image,
        post_rot: torch.Tensor,
        post_tran: torch.Tensor,
        resize: float,
        resize_dims: Tuple[int, int],
        crop: Tuple[int, int, int, int],
        flip: bool,
        rotate: float,
    ) -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        post_rot = post_rot * resize
        post_tran = post_tran - torch.tensor(crop[:2], dtype=torch.float32)
        if flip:
            transform = torch.tensor([[-1, 0], [0, 1]], dtype=torch.float32)
            bias = torch.tensor([crop[2] - crop[0], 0], dtype=torch.float32)
            post_rot = transform @ post_rot
            post_tran = transform @ post_tran + bias

        rotate_mat = self.get_rot(rotate / 180.0 * np.pi)
        bias = torch.tensor([crop[2] - crop[0], crop[3] - crop[1]], dtype=torch.float32) / 2.0
        bias = rotate_mat @ (-bias) + bias
        post_rot = rotate_mat @ post_rot
        post_tran = rotate_mat @ post_tran + bias
        return img, post_rot, post_tran

    def _build_transform(self, rotation, translation) -> torch.Tensor:
        transform = torch.eye(4, dtype=torch.float32)
        rotation_array = np.asarray(rotation, dtype=np.float32)
        if rotation_array.shape == (3, 3):
            transform[:3, :3] = torch.from_numpy(rotation_array)
        else:
            quat = Quaternion(rotation_array.tolist())
            transform[:3, :3] = torch.from_numpy(quat.rotation_matrix.astype(np.float32))
        transform[:3, 3] = torch.as_tensor(translation, dtype=torch.float32)
        return transform

    def get_sensor_transforms(self, cam_info: Dict, cam_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        cam_record = cam_info["cams"][cam_name]
        if (
            "sensor2ego_rotation" in cam_record
            and "sensor2ego_translation" in cam_record
            and "ego2global_rotation" in cam_record
            and "ego2global_translation" in cam_record
        ):
            sensor2ego = self._build_transform(
                cam_record["sensor2ego_rotation"],
                cam_record["sensor2ego_translation"],
            )
            ego2global = self._build_transform(
                cam_record["ego2global_rotation"],
                cam_record["ego2global_translation"],
            )
            return sensor2ego, ego2global

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
        return lidar2ego @ sensor2lidar, ego2global

    def transform(self, results: Dict) -> Dict:
        curr = results["curr"]
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        filenames = []

        for cam_name in self.choose_cams():
            cam_data = curr["cams"][cam_name]
            img = Image.open(cam_data["data_path"]).convert("RGB")
            post_rot = torch.eye(2, dtype=torch.float32)
            post_tran = torch.zeros(2, dtype=torch.float32)

            sensor2ego, ego2global = self.get_sensor_transforms(curr, cam_name)
            intrin = torch.tensor(cam_data["cam_intrinsic"], dtype=torch.float32)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(img.height, img.width)
            img, post_rot2, post_tran2 = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )

            post_rot_3d = torch.eye(3, dtype=torch.float32)
            post_tran_3d = torch.zeros(3, dtype=torch.float32)
            post_rot_3d[:2, :2] = post_rot2
            post_tran_3d[:2] = post_tran2

            imgs.append(normalize_img(img))
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            intrins.append(intrin)
            post_rots.append(post_rot_3d)
            post_trans.append(post_tran_3d)
            filenames.append(cam_data["data_path"])

        img_shape = tuple(imgs[0].shape[-2:])
        results["img_inputs"] = (
            torch.stack(imgs, dim=0),
            torch.stack(sensor2egos, dim=0),
            torch.stack(ego2globals, dim=0),
            torch.stack(intrins, dim=0),
            torch.stack(post_rots, dim=0),
            torch.stack(post_trans, dim=0),
        )
        results["img_path"] = filenames
        results["img_shape"] = img_shape
        results["ori_shape"] = img_shape
        results["pad_shape"] = img_shape
        results["scale_factor"] = 1.0
        results["img_norm_cfg"] = dict(
            mean=np.array([123.675, 116.28, 103.53], dtype=np.float32),
            std=np.array([58.395, 57.12, 57.375], dtype=np.float32),
            to_rgb=True,
        )
        return results


@TRANSFORMS.register_module()
class FusionNeXtLoadAnnotations(BaseTransform):
    def __init__(self, classes: Sequence[str], with_velocity: bool = True, use_valid_flag: bool = False):
        self.classes = tuple(classes)
        self.with_velocity = with_velocity
        self.use_valid_flag = use_valid_flag

    def transform(self, results: Dict) -> Dict:
        info = results.get("curr")
        if info is None:
            raise KeyError("FusionNeXtLoadAnnotations expects results['curr'].")

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


@TRANSFORMS.register_module()
class FusionNeXtObjectNameFilter(BaseTransform):
    def __init__(self, classes: Sequence[str]):
        self.classes = tuple(classes)
        self.labels = list(range(len(self.classes)))

    def transform(self, input_dict: Dict) -> Dict:
        gt_labels_3d = np.asarray(input_dict["gt_labels_3d"])
        gt_bboxes_mask = np.array([label in self.labels for label in gt_labels_3d], dtype=np.bool_)
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = gt_labels_3d[gt_bboxes_mask]
        if "gt_names" in input_dict:
            input_dict["gt_names"] = np.asarray(input_dict["gt_names"], dtype=object)[gt_bboxes_mask]
        return input_dict


@TRANSFORMS.register_module()
class FusionNeXtObjectRangeFilter(BaseTransform):
    def __init__(self, point_cloud_range: Sequence[float]):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def transform(self, input_dict: Dict) -> Dict:
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = np.asarray(input_dict["gt_labels_3d"])

        bev_range = self.pcd_range[[0, 1, 3, 4]]
        mask = gt_bboxes_3d.in_range_bev(bev_range).numpy().astype(np.bool_)
        input_dict["gt_bboxes_3d"] = gt_bboxes_3d[mask]
        input_dict["gt_labels_3d"] = gt_labels_3d[mask]
        if "gt_names" in input_dict:
            input_dict["gt_names"] = np.asarray(input_dict["gt_names"], dtype=object)[mask]

        input_dict["gt_bboxes_3d"].limit_yaw(offset=0.5, period=2 * np.pi)
        return input_dict


@TRANSFORMS.register_module()
class PackFusionDetInputs(Pack3DDetInputs):
    INPUTS_KEYS = ["points", "img", "img_inputs"]

    def pack_single_results(self, results: dict) -> dict:
        if "points" in results and isinstance(results["points"], BasePoints):
            results["points"] = results["points"].tensor

        for key in [
            "proposals",
            "gt_bboxes",
            "gt_bboxes_ignore",
            "gt_labels",
            "gt_bboxes_labels",
            "attr_labels",
            "pts_instance_mask",
            "pts_semantic_mask",
            "centers_2d",
            "depths",
            "gt_labels_3d",
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])

        if "gt_bboxes_3d" in results and not isinstance(results["gt_bboxes_3d"], LiDARInstance3DBoxes):
            results["gt_bboxes_3d"] = to_tensor(results["gt_bboxes_3d"])

        data_sample = Det3DDataSample()
        gt_instances_3d = InstanceData()
        gt_instances = InstanceData()
        gt_pts_seg = PointData()

        data_metas = {}
        for key in self.meta_keys:
            if key in results:
                data_metas[key] = results[key]
        data_sample.set_metainfo(data_metas)

        inputs = {}
        for key in self.keys:
            if key not in results:
                continue
            if key in self.INPUTS_KEYS:
                inputs[key] = results[key]
            elif key in self.INSTANCEDATA_3D_KEYS:
                gt_instances_3d[self._remove_prefix(key)] = results[key]
            elif key in self.INSTANCEDATA_2D_KEYS:
                if key == "gt_bboxes_labels":
                    gt_instances["labels"] = results[key]
                else:
                    gt_instances[self._remove_prefix(key)] = results[key]
            elif key in self.SEG_KEYS:
                gt_pts_seg[self._remove_prefix(key)] = results[key]
            else:
                raise NotImplementedError(f"Unsupported key in PackFusionDetInputs: {key}")

        data_sample.gt_instances_3d = gt_instances_3d
        data_sample.gt_instances = gt_instances
        data_sample.gt_pts_seg = gt_pts_seg
        data_sample.eval_ann_info = None

        return {
            "inputs": inputs,
            "data_samples": data_sample,
        }


def build_fusionnext_train_pipeline(
    data_config: Dict,
    class_names: Sequence[str],
    point_cloud_range: Sequence[float],
    load_dim: int = 5,
    use_dim: int = 5,
    backend_args: Optional[Dict] = None,
) -> list:
    return [
        dict(type="FusionNeXtPrepareImageInputs", is_train=True, data_config=data_config),
        dict(
            type="LoadPointsFromFile",
            coord_type="LIDAR",
            load_dim=load_dim,
            use_dim=use_dim,
            backend_args=backend_args,
        ),
        dict(type="FusionNeXtLoadAnnotations", classes=class_names),
        dict(type="FusionNeXtObjectRangeFilter", point_cloud_range=point_cloud_range),
        dict(type="FusionNeXtObjectNameFilter", classes=class_names),
        dict(type="FusionNeXtPrepareMeta"),
        dict(
            type="PackFusionDetInputs",
            keys=["points", "img_inputs", "gt_bboxes_3d", "gt_labels_3d"],
            meta_keys=FUSIONNEXT_META_KEYS,
        ),
    ]


def build_fusionnext_test_pipeline(
    data_config: Dict,
    class_names: Sequence[str],
    load_dim: int = 5,
    use_dim: int = 5,
    backend_args: Optional[Dict] = None,
) -> list:
    return [
        dict(type="FusionNeXtPrepareImageInputs", data_config=data_config),
        dict(
            type="LoadPointsFromFile",
            coord_type="LIDAR",
            load_dim=load_dim,
            use_dim=use_dim,
            backend_args=backend_args,
        ),
        dict(type="FusionNeXtPrepareMeta"),
        dict(
            type="PackFusionDetInputs",
            keys=["points", "img_inputs"],
            meta_keys=FUSIONNEXT_META_KEYS,
        ),
    ]


def build_fusionnext_dataset_cfg(
    ann_file: str,
    data_root: str,
    data_config: Dict,
    class_names: Sequence[str],
    point_cloud_range: Sequence[float],
    test_mode: bool = False,
    load_dim: int = 5,
    use_dim: int = 5,
    backend_args: Optional[Dict] = None,
) -> Dict:
    pipeline = (
        build_fusionnext_test_pipeline(
            data_config=data_config,
            class_names=class_names,
            load_dim=load_dim,
            use_dim=use_dim,
            backend_args=backend_args,
        )
        if test_mode
        else build_fusionnext_train_pipeline(
            data_config=data_config,
            class_names=class_names,
            point_cloud_range=point_cloud_range,
            load_dim=load_dim,
            use_dim=use_dim,
            backend_args=backend_args,
        )
    )

    return dict(
        type="FusionNuScenesDataset",
        data_root=data_root,
        ann_file=ann_file,
        pipeline=pipeline,
        metainfo=dict(classes=list(class_names)),
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False,
        ),
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
        data_preprocessor=dict(type="FusionDet3DDataPreprocessor"),
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
        train_dataloader=dict(
            dataset=build_fusionnext_dataset_cfg(
                ann_file=train_ann_file,
                data_root=data_root,
                data_config=data_config,
                class_names=class_names,
                point_cloud_range=point_cloud_range,
                test_mode=False,
            )
        ),
        val_dataloader=dict(
            dataset=build_fusionnext_dataset_cfg(
                ann_file=val_ann_file,
                data_root=data_root,
                data_config=data_config,
                class_names=class_names,
                point_cloud_range=point_cloud_range,
                test_mode=True,
            )
        ),
        test_dataloader=dict(
            dataset=build_fusionnext_dataset_cfg(
                ann_file=val_ann_file,
                data_root=data_root,
                data_config=data_config,
                class_names=class_names,
                point_cloud_range=point_cloud_range,
                test_mode=True,
            )
        ),
    )
