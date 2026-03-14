from typing import Sequence

import torch
from mmengine.model import BaseDataPreprocessor

from mmdet3d.registry import MODELS


@MODELS.register_module()
class FusionDet3DDataPreprocessor(BaseDataPreprocessor):
    def forward(self, data: dict, training: bool = False) -> dict:
        del training
        data = self.cast_data(data)
        inputs = data["inputs"]
        batch_inputs = {}

        if "points" in inputs:
            batch_inputs["points"] = inputs["points"]

        if "img_inputs" in inputs:
            batch_inputs["img_inputs"] = self._stack_img_inputs(inputs["img_inputs"])

        return {
            "inputs": batch_inputs,
            "data_samples": data.get("data_samples", None),
        }

    def _stack_img_inputs(self, img_inputs):
        if not isinstance(img_inputs, Sequence):
            raise TypeError(f"img_inputs must be a sequence, but got {type(img_inputs)}")

        stacked = []
        for field in img_inputs:
            if isinstance(field, torch.Tensor):
                stacked.append(field)
            elif isinstance(field, Sequence) and field and isinstance(field[0], torch.Tensor):
                stacked.append(torch.stack(list(field), dim=0))
            else:
                raise TypeError(f"Unsupported img_inputs field type: {type(field)}")
        return tuple(stacked)
