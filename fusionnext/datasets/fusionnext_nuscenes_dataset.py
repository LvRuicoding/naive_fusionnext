import copy
import os
from typing import Callable, List, Optional, Sequence, Union

from mmengine.dataset import BaseDataset
from mmengine.fileio import load

from mmdet3d.registry import DATASETS
from mmdet3d.structures import get_box_type


@DATASETS.register_module()
class FusionNuScenesDataset(BaseDataset):
    METAINFO = {
        "classes": (
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ),
        "version": "v1.0-mini",
    }

    def __init__(
        self,
        ann_file: str,
        data_root: str = "",
        metainfo: Optional[dict] = None,
        pipeline: List[Union[dict, Callable]] = [],
        modality: Optional[dict] = None,
        box_type_3d: str = "LiDAR",
        filter_empty_gt: bool = True,
        test_mode: bool = False,
        use_valid_flag: bool = False,
        load_interval: int = 1,
        serialize_data: bool = True,
        **kwargs,
    ) -> None:
        self.modality = modality or dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False,
        )
        self.filter_empty_gt = filter_empty_gt
        self.use_valid_flag = use_valid_flag
        self.load_interval = load_interval
        self.box_type_3d_name = box_type_3d
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.box_mode_3d_name = getattr(self.box_mode_3d, "name", str(self.box_mode_3d))
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            metainfo=metainfo,
            pipeline=pipeline,
            test_mode=test_mode,
            serialize_data=serialize_data,
            **kwargs,
        )
        self._metainfo["box_type_3d"] = self.box_type_3d_name
        self._metainfo["box_mode_3d"] = self.box_mode_3d_name

    def _join_data_path(self, path: str) -> str:
        if os.path.isabs(path) or not self.data_root:
            return path
        normalized = os.path.normpath(path)
        if normalized.startswith(f"data{os.sep}"):
            data_root_norm = os.path.normpath(self.data_root)
            parent = os.path.dirname(data_root_norm)
            if os.path.basename(parent) == "data":
                return os.path.join(os.path.dirname(parent), normalized)
        return os.path.join(self.data_root, normalized)

    def load_data_list(self) -> List[dict]:
        annotations = load(self.ann_file)
        infos = annotations["infos"]
        if self.load_interval > 1:
            infos = infos[:: self.load_interval]

        metadata = annotations.get("metadata", {})
        if "version" in metadata:
            self._metainfo.setdefault("version", metadata["version"])

        data_list = []
        for info in infos:
            parsed = copy.deepcopy(info)
            parsed["lidar_path"] = self._join_data_path(parsed["lidar_path"])
            for cam_info in parsed.get("cams", {}).values():
                cam_info["data_path"] = self._join_data_path(cam_info["data_path"])

            data_list.append(
                dict(
                    token=parsed["token"],
                    sample_idx=parsed["token"],
                    timestamp=parsed["timestamp"] / 1e6,
                    curr=parsed,
                    lidar_points=dict(lidar_path=parsed["lidar_path"]),
                    lidar_path=parsed["lidar_path"],
                    box_type_3d=self.box_type_3d_name,
                    box_mode_3d=self.box_mode_3d_name,
                )
            )
        return data_list

    def filter_data(self) -> List[dict]:
        if self.test_mode or not self.filter_empty_gt:
            return self.data_list

        valid_classes = set(self.metainfo["classes"])
        filtered = []
        for data_info in self.data_list:
            curr = data_info["curr"]
            if self.use_valid_flag and "valid_flag" in curr:
                mask = curr["valid_flag"]
            else:
                mask = curr["num_lidar_pts"] > 0

            gt_names = curr["gt_names"][mask]
            if any(name in valid_classes for name in gt_names):
                filtered.append(data_info)
        return filtered

    def get_cat_ids(self, idx: int) -> List[int]:
        data_info = self.get_data_info(idx)
        curr = data_info["curr"]
        if self.use_valid_flag and "valid_flag" in curr:
            mask = curr["valid_flag"]
        else:
            mask = curr["num_lidar_pts"] > 0

        cat_ids = []
        class_names = self.metainfo["classes"]
        for name in set(curr["gt_names"][mask]):
            if name in class_names:
                cat_ids.append(class_names.index(name))
        return cat_ids
