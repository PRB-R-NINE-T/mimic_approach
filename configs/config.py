"""Dataclass-based configs for mimic-video training."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    repo_id: str = "pierre818191/UnitreeBagClose"
    num_episodes: int = 200
    train_episodes: int = 180
    val_episodes: int = 20

    # Frame dimensions (per camera)
    camera_height: int = 480
    camera_width: int = 640
    camera_names: List[str] = field(
        default_factory=lambda: [
            "observation.images.cam_left_high",
            "observation.images.cam_right_high",
            "observation.images.cam_left_wrist",
        ]
    )

    # Video frames: 17 pixel frames -> 5 latent frames (2 cond + 3 pred)
    num_pixel_frames: int = 17
    num_latent_frames: int = 5
    num_cond_latent_frames: int = 2
    num_pred_latent_frames: int = 3
    fps: int = 30

    # State/Action feature keys (will be concatenated)
    state_keys: List[str] = field(
        default_factory=lambda: [
            "observation.left_arm",
            "observation.right_arm",
            "observation.left_gripper",
            "observation.right_gripper",
        ]
    )
    action_keys: List[str] = field(
        default_factory=lambda: [
            "action.left_arm",
            "action.right_arm",
            "action.left_gripper",
            "action.right_gripper",
        ]
    )

    # Actions
    action_chunk_size: int = 16
    action_dim: int = 16  # left_arm(7) + right_arm(7) + left_gripper(1) + right_gripper(1)
    proprio_dim: int = 16

    # Proprioception masking probability during training
    proprio_mask_prob: float = 0.1

    # Text prompt for the task
    task_prompt: str = "Organize and tidy the items on the table."

    # Precomputed embeddings path
    precomputed_dir: str = "precomputed/"


@dataclass
class ModelConfig:
    # Cosmos model
    cosmos_model_id: str = "nvidia/Cosmos-Predict2-2B-Video2World"

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
        ]
    )

    # Hidden state extraction
    hidden_state_layer: int = 19  # Layer k=19
    hidden_state_pool: str = "none"  # "mean" (5 tokens) or "none" (all ~6000 tokens)

    # Action decoder
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 8
    decoder_num_heads: int = 8
    decoder_mlp_ratio: int = 4
    backbone_hidden_dim: int = 2048  # Cosmos transformer hidden dim

    # VAE latent channels
    vae_latent_channels: int = 16


@dataclass
class Stage1Config:
    lr: float = 1e-4
    warmup_steps: int = 10
    weight_decay: float = 0.1
    grad_clip: float = 10.0
    total_steps: int = 100
    batch_size: int = 256  # effective batch size via accumulation
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 256

    # LR schedule: constant after warmup
    lr_schedule: str = "constant"

    # Mixed precision
    dtype: str = "bf16"
    gradient_checkpointing: bool = True

    # Logging
    log_every: int = 10
    save_every: int = 50
    output_dir: str = "checkpoints/stage1"
    wandb_project: str = "mimic-video"
    wandb_run_name: str = "stage1-lora"


@dataclass
class Stage2Config:
    lr: float = 1e-4
    warmup_steps: int = 10
    weight_decay: float = 0.1
    grad_clip: float = 10.0
    total_steps: int = 100
    batch_size: int = 32  # effective batch size
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 32

    # LR schedule: linear decay after warmup
    lr_schedule: str = "linear_decay"

    # Mixed precision
    dtype: str = "bf16"
    gradient_checkpointing: bool = True

    # Action flow matching tau sampling
    # pi0-style: U^(1/power) where U~Uniform(0,1)
    tau_power: float = 0.999

    # Logging
    log_every: int = 10
    save_every: int = 50
    output_dir: str = "checkpoints/stage2"
    wandb_project: str = "mimic-video"
    wandb_run_name: str = "stage2-action-decoder"

    # Stage 1 checkpoint to load
    stage1_checkpoint: str = "checkpoints/stage1/final"
