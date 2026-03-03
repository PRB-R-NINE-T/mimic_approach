"""Entry point for Stage 1: LoRA finetuning of the video backbone.

Usage:
    python scripts/train_stage1.py [--resume CHECKPOINT_PATH]
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DataConfig, ModelConfig, Stage1Config
from mimic_video.data.dataset import MimicVideoDataset
from mimic_video.models.video_backbone import CosmosVideoBackbone
from mimic_video.training.stage1_trainer import Stage1Trainer


def main():
    parser = argparse.ArgumentParser(description="Stage 1: LoRA finetuning")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precomputed_dir", type=str, default="precomputed/")
    args = parser.parse_args()

    data_config = DataConfig()
    model_config = ModelConfig()
    train_config = Stage1Config()

    # Load precomputed T5 embedding
    t5_path = os.path.join(args.precomputed_dir, "t5_embedding.pt")
    if os.path.exists(t5_path):
        print(f"Loading precomputed T5 embedding from {t5_path}")
        t5_embedding = torch.load(t5_path, map_location="cpu", weights_only=True)
    else:
        print("WARNING: No precomputed T5 embedding found. Run precompute_embeddings.py first.")
        t5_embedding = None

    # Create dataset
    print("Loading dataset...")
    train_episodes = list(range(data_config.train_episodes))

    train_dataset = MimicVideoDataset(
        repo_id=data_config.repo_id,
        camera_names=data_config.camera_names,
        state_keys=data_config.state_keys,
        action_keys=data_config.action_keys,
        num_pixel_frames=data_config.num_pixel_frames,
        action_chunk_size=data_config.action_chunk_size,
        action_dim=data_config.action_dim,
        proprio_dim=data_config.proprio_dim,
        target_height=data_config.camera_height,
        target_width=data_config.camera_width,
        episode_indices=train_episodes,
        precomputed_dir=args.precomputed_dir,
    )

    print(f"Train dataset: {len(train_dataset)} samples")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.micro_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Load model
    print("Loading Cosmos video backbone...")
    backbone = CosmosVideoBackbone(
        model_id=model_config.cosmos_model_id,
        lora_rank=model_config.lora_rank,
        lora_alpha=model_config.lora_alpha,
        lora_target_modules=model_config.lora_target_modules,
        hidden_state_layer=model_config.hidden_state_layer,
        dtype=torch.bfloat16,
        device=args.device,
    )

    # Enable gradient checkpointing for memory efficiency
    backbone.transformer.base_model.model.enable_gradient_checkpointing()

    # Move transformer to GPU, offload VAE/T5 to CPU
    backbone.transformer.to(args.device)
    backbone.offload_vae_and_text_encoder("cpu")

    # Print trainable parameters
    trainable = sum(p.numel() for p in backbone.transformer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone.transformer.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Create trainer
    trainer = Stage1Trainer(
        backbone=backbone,
        train_dataloader=train_dataloader,
        lr=train_config.lr,
        warmup_steps=train_config.warmup_steps,
        weight_decay=train_config.weight_decay,
        grad_clip=train_config.grad_clip,
        total_steps=train_config.total_steps,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        lr_schedule=train_config.lr_schedule,
        dtype=train_config.dtype,
        output_dir=train_config.output_dir,
        log_every=train_config.log_every,
        save_every=train_config.save_every,
        wandb_project=train_config.wandb_project,
        wandb_run_name=train_config.wandb_run_name,
        precomputed_t5_embedding=t5_embedding,
        num_cond_latent_frames=data_config.num_cond_latent_frames,
        device=args.device,
    )

    # Resume if requested
    if args.resume:
        print(f"Resuming from {args.resume}")
        step = trainer._load_checkpoint(args.resume)
        print(f"Resumed at step {step}")

    # Train
    print("Starting Stage 1 training...")
    trainer.train()


if __name__ == "__main__":
    main()
