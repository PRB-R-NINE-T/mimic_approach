"""Stage 1 Trainer: LoRA finetuning of the Cosmos video backbone.

Trains the video backbone with LoRA to predict future video frames
using flow matching, conditioned on past frames and text embeddings.
"""

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm

from mimic_video.models.video_backbone import CosmosVideoBackbone
from mimic_video.models.flow_matching import FlowMatchingScheduler


class Stage1Trainer:
    """Stage 1: LoRA finetuning of the video backbone for video prediction."""

    def __init__(
        self,
        backbone: CosmosVideoBackbone,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        warmup_steps: int = 1000,
        weight_decay: float = 0.1,
        grad_clip: float = 10.0,
        total_steps: int = 27000,
        gradient_accumulation_steps: int = 256,
        lr_schedule: str = "constant",
        dtype: str = "bf16",
        output_dir: str = "checkpoints/stage1",
        log_every: int = 10,
        save_every: int = 1000,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        precomputed_t5_embedding: Optional[torch.Tensor] = None,
        num_cond_latent_frames: int = 2,
        device: str = "cuda",
    ):
        self.backbone = backbone
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.total_steps = total_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_schedule = lr_schedule
        self.output_dir = output_dir
        self.log_every = log_every
        self.save_every = save_every
        self.num_cond_latent_frames = num_cond_latent_frames
        self.device = device
        self.precomputed_t5_embedding = precomputed_t5_embedding

        self.compute_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        self.fm = FlowMatchingScheduler()

        # Enable gradient checkpointing
        if hasattr(backbone.transformer, 'base_model'):
            backbone.transformer.base_model.model.gradient_checkpointing = True
        else:
            backbone.transformer.gradient_checkpointing = True

        # Freeze VAE and text encoder
        for param in backbone.vae.parameters():
            param.requires_grad = False
        if backbone.text_encoder is not None:
            for param in backbone.text_encoder.parameters():
                param.requires_grad = False

        # Setup optimizer (only LoRA params)
        trainable_params = [p for p in backbone.transformer.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Setup LR scheduler
        self.lr_scheduler = self._build_lr_scheduler()

        # Wandb logging
        self.use_wandb = wandb_project is not None
        if self.use_wandb:
            import wandb
            wandb.init(project=wandb_project, name=wandb_run_name)

        os.makedirs(output_dir, exist_ok=True)

    def _build_lr_scheduler(self):
        """Build learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            # Constant LR after warmup for stage 1
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Dict with "video" [B, T, C, H, W] frames in [-1, 1].

        Returns:
            Dict with loss metrics.
        """
        video = batch["video"]  # [B, T, C, H, W]
        B = video.shape[0]

        # Rearrange to [B, C, T, H, W] for VAE
        video = video.permute(0, 2, 1, 3, 4).to(self.device)

        # Encode with VAE (frozen, no grad)
        with torch.no_grad():
            self.backbone.move_vae_to(self.device)
            z_0 = self.backbone.encode_video(video)  # [B, C_lat, T_lat, H_lat, W_lat]
            self.backbone.offload_vae_and_text_encoder("cpu")

        # Split into conditioning and prediction
        z_cond = z_0[:, :, :self.num_cond_latent_frames]   # [B, C, T_cond, H, W]
        z_pred = z_0[:, :, self.num_cond_latent_frames:]   # [B, C, T_pred, H, W]

        # Sample noise and timesteps
        eps_v = torch.randn_like(z_pred)
        tau_v = self.fm.sample_tau_video(B, device=z_pred.device)  # [B]

        # Create noisy latents: z_tau = (1-tau)*z_pred + tau*eps
        z_noisy = self.fm.interpolate(z_pred, eps_v, tau_v)

        # Get T5 text embedding
        if self.precomputed_t5_embedding is not None:
            t5_emb = self.precomputed_t5_embedding.to(self.device, dtype=self.compute_dtype)
            if t5_emb.shape[0] == 1:
                t5_emb = t5_emb.expand(B, -1, -1)
        elif "t5_embedding" in batch:
            t5_emb = batch["t5_embedding"].to(self.device, dtype=self.compute_dtype)
        else:
            raise ValueError("No T5 embedding available. Either precompute or include in batch.")

        # Forward through transformer (LoRA active)
        with torch.amp.autocast("cuda", dtype=self.compute_dtype):
            raw_output, _ = self.backbone.forward_transformer(
                z_noisy=z_noisy,
                z_cond=z_cond,
                tau_v=tau_v,
                encoder_hidden_states=t5_emb,
            )

        # The raw output of the Cosmos network IS the velocity v = eps - x_0
        # We only compute loss on the prediction frames (not conditioning frames)
        T_cond = self.num_cond_latent_frames
        velocity_pred = raw_output[:, :, T_cond:]  # [B, C, T_pred, H, W]
        velocity_target = self.fm.velocity_target(z_pred, eps_v)  # [B, C, T_pred, H, W]

        loss = self.fm.compute_loss(velocity_pred.float(), velocity_target.float())

        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps

        loss.backward()

        return {"loss": loss.item() * self.gradient_accumulation_steps}

    def train(self):
        """Run the full training loop."""
        self.backbone.transformer.train()
        self.backbone.vae.eval()

        data_iter = iter(self.train_dataloader)
        running_loss = 0.0
        global_step = 0

        pbar = tqdm(total=self.total_steps, desc="Stage 1 Training")

        while global_step < self.total_steps:
            self.optimizer.zero_grad()

            # Gradient accumulation
            for micro_step in range(self.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)

                metrics = self.train_step(batch)
                running_loss += metrics["loss"]

            # Clip gradients
            trainable_params = [p for p in self.backbone.transformer.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, self.grad_clip)

            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()

            global_step += 1
            avg_loss = running_loss / self.gradient_accumulation_steps
            running_loss = 0.0

            # Logging
            if global_step % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/step": global_step,
                    })

            # Save checkpoint
            if global_step % self.save_every == 0:
                self._save_checkpoint(global_step)

            pbar.update(1)

        pbar.close()

        # Save final checkpoint
        self._save_checkpoint(global_step, is_final=True)
        print(f"Stage 1 training complete. Final checkpoint saved to {self.output_dir}")

    def _save_checkpoint(self, step: int, is_final: bool = False):
        """Save LoRA checkpoint."""
        suffix = "final" if is_final else f"step_{step}"
        save_path = os.path.join(self.output_dir, suffix)
        os.makedirs(save_path, exist_ok=True)

        self.backbone.save_lora(save_path)

        # Save optimizer state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "step": step,
        }, os.path.join(save_path, "training_state.pt"))

        print(f"Checkpoint saved to {save_path}")

    def _load_checkpoint(self, path: str):
        """Load a checkpoint."""
        self.backbone.load_lora(path)

        state_path = os.path.join(path, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            self.optimizer.load_state_dict(state["optimizer"])
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
            return state["step"]
        return 0
