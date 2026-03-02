"""Entry point for Stage 2: Action decoder training with frozen backbone.

Usage:
    python scripts/train_stage2.py [--stage1_checkpoint CHECKPOINT_PATH] [--resume CHECKPOINT_PATH]
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DataConfig, ModelConfig, Stage2Config
from mimic_video.data.dataset import MimicVideoDataset
from mimic_video.models.video_backbone import CosmosVideoBackbone
from mimic_video.models.action_decoder import ActionDecoderDiT
from mimic_video.training.stage2_trainer import Stage2Trainer


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Action decoder training")
    parser.add_argument(
        "--stage1_checkpoint", type=str, default=None,
        help="Path to Stage 1 LoRA checkpoint (default: from config)"
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to Stage 2 checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precomputed_dir", type=str, default="precomputed/")
    args = parser.parse_args()

    data_config = DataConfig()
    model_config = ModelConfig()
    train_config = Stage2Config()

    stage1_path = args.stage1_checkpoint or train_config.stage1_checkpoint

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
        num_pixel_frames=data_config.num_pixel_frames,
        action_chunk_size=data_config.action_chunk_size,
        action_dim=data_config.action_dim,
        proprio_dim=data_config.proprio_dim,
        target_height=data_config.camera_height,
        target_width=data_config.camera_width,
        episode_indices=train_episodes,
        precomputed_dir=args.precomputed_dir,
    )

    # Compute or load action stats
    stats_path = os.path.join(args.precomputed_dir, "action_stats.pt")
    if os.path.exists(stats_path):
        print(f"Loading action stats from {stats_path}")
        action_stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        train_dataset.action_mean = action_stats["mean"]
        train_dataset.action_std = action_stats["std"]
    else:
        print("Computing action statistics...")
        action_stats = train_dataset.compute_action_stats()
        os.makedirs(args.precomputed_dir, exist_ok=True)
        torch.save(action_stats, stats_path)
        print(f"Saved action stats to {stats_path}")

    print(f"Train dataset: {len(train_dataset)} samples")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.micro_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Load backbone with Stage 1 LoRA weights
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

    # Load Stage 1 LoRA weights
    if os.path.exists(stage1_path):
        print(f"Loading Stage 1 LoRA weights from {stage1_path}")
        backbone.load_lora(stage1_path)
    else:
        print(f"WARNING: Stage 1 checkpoint not found at {stage1_path}. Using base model.")

    backbone.transformer.to(args.device)
    backbone.offload_vae_and_text_encoder("cpu")

    # Create action decoder
    print("Creating action decoder...")
    action_decoder = ActionDecoderDiT(
        action_dim=data_config.action_dim,
        proprio_dim=data_config.proprio_dim,
        hidden_dim=model_config.decoder_hidden_dim,
        num_layers=model_config.decoder_num_layers,
        num_heads=model_config.decoder_num_heads,
        mlp_ratio=model_config.decoder_mlp_ratio,
        backbone_hidden_dim=backbone.hidden_dim,  # Auto-detected from loaded model
        action_chunk_size=data_config.action_chunk_size,
        proprio_mask_prob=data_config.proprio_mask_prob,
    )

    # Print parameter counts
    decoder_params = sum(p.numel() for p in action_decoder.parameters())
    print(f"Action decoder parameters: {decoder_params:,} (~{decoder_params / 1e6:.1f}M)")

    # Create trainer
    trainer = Stage2Trainer(
        backbone=backbone,
        action_decoder=action_decoder,
        train_dataloader=train_dataloader,
        lr=train_config.lr,
        warmup_steps=train_config.warmup_steps,
        weight_decay=train_config.weight_decay,
        grad_clip=train_config.grad_clip,
        total_steps=train_config.total_steps,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        lr_schedule=train_config.lr_schedule,
        tau_power=train_config.tau_power,
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
    print("Starting Stage 2 training...")
    trainer.train()


if __name__ == "__main__":
    main()
