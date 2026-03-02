"""Image transforms for mimic-video: 2x2 camera concat and normalization."""

import torch
import torch.nn.functional as F
from typing import List


def concat_cameras_2x2(
    images: List[torch.Tensor],
    target_height: int = 480,
    target_width: int = 640,
) -> torch.Tensor:
    """Stack 4 camera images into a 2x2 grid and resize back to target resolution.

    Args:
        images: List of 4 tensors, each [C, H, W] or [T, C, H, W].
        target_height: Output height after resize.
        target_width: Output width after resize.

    Returns:
        Tensor of shape [C, target_height, target_width] or [T, C, target_height, target_width].
    """
    assert len(images) == 4, f"Expected 4 cameras, got {len(images)}"

    has_time = images[0].ndim == 4
    if has_time:
        # [T, C, H, W] for each camera -> 2x2 grid -> resize per frame
        top = torch.cat([images[0], images[1]], dim=-1)   # [T, C, H, 2W]
        bottom = torch.cat([images[2], images[3]], dim=-1) # [T, C, H, 2W]
        grid = torch.cat([top, bottom], dim=-2)            # [T, C, 2H, 2W]

        T, C = grid.shape[:2]
        # Batch resize: treat T*C as batch dim for F.interpolate
        grid_flat = grid.reshape(T * C, grid.shape[2], grid.shape[3]).unsqueeze(1)  # [T*C, 1, 2H, 2W]
        grid_resized = F.interpolate(
            grid_flat, size=(target_height, target_width), mode="bilinear", align_corners=False
        )  # [T*C, 1, H, W]
        return grid_resized.reshape(T, C, target_height, target_width)
    else:
        # [C, H, W] for each camera
        top = torch.cat([images[0], images[1]], dim=-1)   # [C, H, 2W]
        bottom = torch.cat([images[2], images[3]], dim=-1) # [C, H, 2W]
        grid = torch.cat([top, bottom], dim=-2)            # [C, 2H, 2W]

        # Resize to target
        grid = grid.unsqueeze(0)  # [1, C, 2H, 2W]
        grid = F.interpolate(
            grid, size=(target_height, target_width), mode="bilinear", align_corners=False
        )
        return grid.squeeze(0)  # [C, H, W]


def normalize_to_neg1_pos1(images: torch.Tensor) -> torch.Tensor:
    """Normalize images from [0, 1] or [0, 255] to [-1, 1].

    Args:
        images: Tensor of pixel values.

    Returns:
        Normalized tensor in [-1, 1].
    """
    if images.max() > 1.0:
        images = images.float() / 255.0
    return images * 2.0 - 1.0


def denormalize_from_neg1_pos1(images: torch.Tensor) -> torch.Tensor:
    """Denormalize images from [-1, 1] to [0, 1].

    Args:
        images: Tensor in [-1, 1].

    Returns:
        Tensor in [0, 1].
    """
    return (images + 1.0) / 2.0
