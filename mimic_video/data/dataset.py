"""MimicVideoDataset wrapping LeRobot for the mimic-video pipeline."""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple

from mimic_video.data.transforms import concat_cameras_2x2, normalize_to_neg1_pos1


class MimicVideoDataset(Dataset):
    """Dataset for mimic-video training.

    Loads episodes from a LeRobot dataset and provides:
    - 17 consecutive video frames (concatenated 2x2 from 4 cameras)
    - Proprioception (16-dim)
    - Action chunk (16 future steps x 16-dim)
    - Cached T5 text embedding (if available)
    """

    # Proprioception keys in order: left_arm(7) + right_arm(7) + left_gripper(1) + right_gripper(1)
    STATE_KEYS = [
        "observation.state",  # This should contain the full state vector
    ]

    ACTION_KEY = "action"

    def __init__(
        self,
        repo_id: str,
        camera_names: list,
        num_pixel_frames: int = 17,
        action_chunk_size: int = 16,
        action_dim: int = 16,
        proprio_dim: int = 16,
        target_height: int = 480,
        target_width: int = 640,
        episode_indices: Optional[list] = None,
        precomputed_dir: Optional[str] = None,
        action_stats: Optional[Dict[str, torch.Tensor]] = None,
        fps: int = 30,
    ):
        """Initialize the dataset.

        Args:
            repo_id: HuggingFace dataset repository ID.
            camera_names: List of 4 camera observation keys.
            num_pixel_frames: Number of consecutive pixel frames to return (17).
            action_chunk_size: Number of future action steps (16).
            action_dim: Dimension of action vector (16).
            proprio_dim: Dimension of proprioception vector (16).
            target_height: Target height after 2x2 concat + resize.
            target_width: Target width after 2x2 concat + resize.
            episode_indices: List of episode indices to use (for train/val split).
            precomputed_dir: Directory with precomputed T5 embeddings and VAE latents.
            action_stats: Dict with 'mean' and 'std' tensors for action normalization.
            fps: Frames per second of the dataset.
        """
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = repo_id
        self.camera_names = camera_names
        self.num_pixel_frames = num_pixel_frames
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.target_height = target_height
        self.target_width = target_width
        self.precomputed_dir = precomputed_dir
        self.fps = fps

        # Build delta_timestamps for LeRobot
        # We need: num_pixel_frames past frames + action_chunk_size future actions
        # Frame delta: [0, 1/fps, 2/fps, ..., (num_pixel_frames-1)/fps]
        frame_deltas = [i / fps for i in range(num_pixel_frames)]
        action_deltas = [(i + 1) / fps for i in range(action_chunk_size)]

        delta_timestamps = {}
        for cam_name in camera_names:
            delta_timestamps[cam_name] = frame_deltas
        delta_timestamps["observation.state"] = [0.0]  # Current state only
        delta_timestamps["action"] = action_deltas

        self.lerobot_dataset = LeRobotDataset(
            repo_id=repo_id,
            delta_timestamps=delta_timestamps,
        )

        # Build valid sample indices
        # Each sample needs: num_pixel_frames consecutive frames + action_chunk_size future frames
        self._build_valid_indices(episode_indices)

        # Action normalization stats
        if action_stats is not None:
            self.action_mean = action_stats["mean"]
            self.action_std = action_stats["std"]
        else:
            self.action_mean = None
            self.action_std = None

        # Load precomputed T5 embedding if available
        self.t5_embedding = None
        if precomputed_dir and os.path.exists(os.path.join(precomputed_dir, "t5_embedding.pt")):
            self.t5_embedding = torch.load(
                os.path.join(precomputed_dir, "t5_embedding.pt"), map_location="cpu", weights_only=True
            )

    def _build_valid_indices(self, episode_indices: Optional[list] = None):
        """Build list of valid (episode_idx, frame_idx) pairs.

        A sample is valid if there are enough consecutive frames for video
        and enough future frames for the action chunk.
        """
        self.valid_indices = []

        # Get episode info from the dataset
        episodes = self.lerobot_dataset.episode_data_index

        for ep_idx in range(len(episodes["from"])):
            if episode_indices is not None and ep_idx not in episode_indices:
                continue

            ep_start = episodes["from"][ep_idx].item()
            ep_end = episodes["to"][ep_idx].item()
            ep_len = ep_end - ep_start

            # Need num_pixel_frames for video + action_chunk_size for actions after the last video frame
            min_frames_needed = self.num_pixel_frames + self.action_chunk_size
            if ep_len < min_frames_needed:
                continue

            # Valid start frames within the episode
            for frame_offset in range(ep_len - min_frames_needed + 1):
                global_idx = ep_start + frame_offset
                self.valid_indices.append(global_idx)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def compute_action_stats(self) -> Dict[str, torch.Tensor]:
        """Compute mean and std of actions across the dataset for normalization."""
        all_actions = []
        for idx in range(min(len(self), 5000)):  # Sample up to 5000 for efficiency
            sample = self.lerobot_dataset[self.valid_indices[idx]]
            action = sample["action"]  # [action_chunk_size, action_dim]
            if isinstance(action, torch.Tensor):
                all_actions.append(action)
            else:
                all_actions.append(torch.tensor(action, dtype=torch.float32))

        all_actions = torch.cat(all_actions, dim=0)  # [N*chunk, action_dim]
        mean = all_actions.mean(dim=0)
        std = all_actions.std(dim=0).clamp(min=1e-6)

        self.action_mean = mean
        self.action_std = std
        return {"mean": mean, "std": std}

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize actions to zero mean, unit variance."""
        if self.action_mean is None:
            return actions
        device = actions.device
        return (actions - self.action_mean.to(device)) / self.action_std.to(device)

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Denormalize actions back to original scale."""
        if self.action_mean is None:
            return actions
        device = actions.device
        return actions * self.action_std.to(device) + self.action_mean.to(device)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample.

        Returns:
            Dict with keys:
            - "video": [T, C, H, W] concatenated 2x2 camera frames, normalized to [-1, 1]
            - "proprio": [proprio_dim] proprioception vector
            - "actions": [action_chunk_size, action_dim] normalized action chunk
            - "t5_embedding": [seq_len, hidden_dim] T5 text embedding (if available)
        """
        global_idx = self.valid_indices[idx]
        sample = self.lerobot_dataset[global_idx]

        # Extract and concatenate camera images
        # Each camera gives [T, C, H, W] for the requested delta_timestamps
        camera_frames = []
        for cam_name in self.camera_names:
            frames = sample[cam_name]  # [T, C, H, W]
            if frames.ndim == 3:
                frames = frames.unsqueeze(0)  # Add time dim if missing
            camera_frames.append(frames)

        # Concatenate cameras in 2x2 grid: each frame gets all 4 cameras concatenated
        # Result: [T, C, H, W]
        video = concat_cameras_2x2(camera_frames, self.target_height, self.target_width)
        video = normalize_to_neg1_pos1(video)  # [T, C, H, W] in [-1, 1]

        # Proprioception
        proprio = sample["observation.state"]  # [1, state_dim] or [state_dim]
        if isinstance(proprio, torch.Tensor):
            proprio = proprio.squeeze(0)
        else:
            proprio = torch.tensor(proprio, dtype=torch.float32).squeeze(0)
        # Take first proprio_dim dimensions
        proprio = proprio[: self.proprio_dim].float()

        # Actions
        actions = sample["action"]  # [action_chunk_size, action_dim]
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32)
        actions = actions[:, : self.action_dim].float()

        # Normalize actions
        actions = self.normalize_actions(actions)

        result = {
            "video": video,
            "proprio": proprio,
            "actions": actions,
        }

        # Add T5 embedding if available
        if self.t5_embedding is not None:
            result["t5_embedding"] = self.t5_embedding

        return result
