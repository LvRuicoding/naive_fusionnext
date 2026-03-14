from typing import Sequence

import torch


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
