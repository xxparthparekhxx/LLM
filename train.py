"""
Advanced training pipeline for language model
Features:
- Gradient accumulation
- Mixed precision training (CUDA AMP / TPU bfloat16)
- Learning rate scheduling
- Checkpointing
- Wandb integration (optional)
- Early stopping
- TPU support via PyTorch/XLA
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast as cuda_autocast
import os
import json
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any
import time
from datetime import datetime
import signal
import sys

from model import LanguageModel, ModelConfig
from data_utils import TextDataset, create_dataloader, split_dataset
from tokenizer import SimpleTokenizer, BPETokenizer


class Trainer:
    """Advanced trainer for language model with TPU support"""

    def __init__(
        self,
        model: LanguageModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = "cuda",
        use_wandb: bool = False,
    ):
        self.config = config
        self.use_wandb = use_wandb
        
        # Detect if using TPU
        self.is_tpu = device == "tpu"
        
        if self.is_tpu:
            # Set up TPU environment
            os.environ['PJRT_DEVICE'] = 'TPU'
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.parallel_loader as pl
            
            self.xm = xm
            self.pl = pl
            self.device = xm.xla_device()
            print(f"Using TPU device: {self.device}")
        else:
            self.xm = None
            self.pl = None
            self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer - use model's production-ready optimizer configuration
        device_type = "cuda" if (not self.is_tpu and self.device == "cuda") else "cpu"
        self.optimizer = self.model.configure_optimizers(
            weight_decay=config.get("weight_decay", 0.1),
            learning_rate=config.get("learning_rate", 3e-4),
            betas=config.get("betas", (0.9, 0.95)),
            device_type=device_type,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("max_epochs", 10),
            eta_min=config.get("min_lr", 1e-6),
        )

        # Mixed precision training
        self.use_amp = config.get("use_amp", True)
        
        if self.is_tpu:
            # TPUs use bfloat16 natively, no need for GradScaler
            self.scaler = None
        else:
            self.use_amp = self.use_amp and self.device == "cuda"
            self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.current_epoch = 0
        self.current_batch = 0  # Track batch within epoch
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.last_checkpoint_time = time.time()  # For hourly checkpoints
        self.interrupted = False  # For Ctrl+C handling

        # Checkpoint directory
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)

        # Initialize wandb if requested
        if self.use_wandb:
            try:
                import wandb

                wandb.init(
                    project=config.get("wandb_project", "llm-training"),
                    config=config,
                    name=config.get(
                        "run_name", f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                    ),
                )
            except ImportError:
                print("Warning: wandb not installed, continuing without it")
                self.use_wandb = False
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully by saving checkpoint"""
        print("\n\n⚠️  Keyboard interrupt detected!")
        print("Saving checkpoint before exit...")
        self.interrupted = True
        self.save_checkpoint(is_best=False, reason="interrupt")
        print("✓ Checkpoint saved! You can resume training later.")
        sys.exit(0)

    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Wrap dataloader for TPU
        if self.is_tpu:
            train_loader = self.pl.ParallelLoader(
                self.train_loader, [self.device]
            ).per_device_loader(self.device)
        else:
            train_loader = self.train_loader

        total_batches = len(train_loader)
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {self.current_epoch + 1}",
            initial=self.current_batch,
            total=total_batches
        )

        for batch_idx, (x, y) in enumerate(pbar):
            # Skip batches if resuming from checkpoint
            if batch_idx < self.current_batch:
                continue
            
            self.current_batch = batch_idx
            if not self.is_tpu:
                x, y = x.to(self.device), y.to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                if self.is_tpu:
                    # TPU: use XLA autocast with bfloat16
                    with torch.autocast(device_type='xla', dtype=torch.bfloat16):
                        logits, loss, _ = self.model(x, y)  # Ignore KV cache during training
                else:
                    # CUDA: use standard autocast
                    with cuda_autocast():
                        logits, loss, _ = self.model(x, y)  # Ignore KV cache during training
                        
                loss = loss / self.config.get("gradient_accumulation_steps", 1)
            else:
                logits, loss, _ = self.model(x, y)  # Ignore KV cache during training
                loss = loss / self.config.get("gradient_accumulation_steps", 1)

            # Backward pass
            if self.use_amp and not self.is_tpu:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.get("gradient_accumulation_steps", 1) == 0:
                # Gradient clipping
                if self.use_amp and not self.is_tpu:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.get("max_grad_norm", 1.0)
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.get("max_grad_norm", 1.0)
                    )
                    if self.is_tpu:
                        # Use XLA optimizer step
                        self.xm.optimizer_step(self.optimizer)
                    else:
                        self.optimizer.step()

                self.optimizer.zero_grad()
                
                # Mark step for TPU to execute graph
                if self.is_tpu:
                    self.xm.mark_step()
                
                self.global_step += 1
            
            # Mark step after backward for gradient accumulation
            elif self.is_tpu:
                self.xm.mark_step()

            total_loss += loss.item() * self.config.get(
                "gradient_accumulation_steps", 1
            )
            num_batches += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f'{loss.item() * self.config.get("gradient_accumulation_steps", 1):.4f}',
                    "lr": f"{self.scheduler.get_last_lr()[0]:.6f}",
                }
            )

            # Log to wandb
            if (
                self.use_wandb
                and self.global_step % self.config.get("log_interval", 100) == 0
            ):
                import wandb

                wandb.log(
                    {
                        "train/loss": loss.item()
                        * self.config.get("gradient_accumulation_steps", 1),
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/step": self.global_step,
                        "train/epoch_progress": f"{batch_idx}/{total_batches}",
                    }
                )
            
            # Hourly checkpoint saving (for Colab)
            current_time = time.time()
            if current_time - self.last_checkpoint_time >= 3600:  # 1 hour
                print(f"\n⏰ Hourly checkpoint at batch {batch_idx}/{total_batches}")
                self.save_checkpoint(is_best=False, reason="hourly")
                self.last_checkpoint_time = current_time

        # Reset batch counter at end of epoch
        self.current_batch = 0
        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Wrap dataloader for TPU
        if self.is_tpu:
            val_loader = self.pl.ParallelLoader(
                self.val_loader, [self.device]
            ).per_device_loader(self.device)
        else:
            val_loader = self.val_loader

        pbar = tqdm(val_loader, desc="Validating")

        for x, y in pbar:
            if not self.is_tpu:
                x, y = x.to(self.device), y.to(self.device)

            if self.use_amp:
                if self.is_tpu:
                    # TPU: use XLA autocast with bfloat16
                    with torch.autocast(device_type='xla', dtype=torch.bfloat16):
                        _, loss, _ = self.model(x, y)  # Ignore logits and KV cache
                else:
                    with cuda_autocast():
                        _, loss, _ = self.model(x, y)  # Ignore logits and KV cache
            else:
                _, loss, _ = self.model(x, y)  # Ignore logits and KV cache

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Mark step after validation for TPU
        if self.is_tpu:
            self.xm.mark_step()

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, is_best: bool = False, reason: str = "periodic"):
        """Save model checkpoint with full training state"""
        checkpoint = {
            "epoch": self.current_epoch,
            "batch": self.current_batch,  # Save batch progress
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
            "reason": reason,  # Why was this checkpoint saved
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Use XLA save for TPU, regular torch.save otherwise
        if self.is_tpu:
            # Synchronize before saving on TPU
            self.xm.mark_step()
            # For TPU, use xm.save which handles XLA tensors properly
            self.xm.save(checkpoint, str(self.checkpoint_dir / "latest.pt"))
            if is_best:
                self.xm.save(checkpoint, str(self.checkpoint_dir / "best.pt"))
                print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
            if (self.current_epoch + 1) % self.config.get("save_interval", 5) == 0:
                self.xm.save(
                    checkpoint, 
                    str(self.checkpoint_dir / f"epoch_{self.current_epoch + 1}.pt")
                )
        else:
            # Save latest checkpoint
            torch.save(checkpoint, self.checkpoint_dir / "latest.pt")

            # Save best checkpoint
            if is_best:
                torch.save(checkpoint, self.checkpoint_dir / "best.pt")
                print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
            
            # Save reason-specific checkpoints
            if reason == "hourly":
                checkpoint_name = f"hourly_epoch{self.current_epoch}_batch{self.current_batch}.pt"
                torch.save(checkpoint, self.checkpoint_dir / checkpoint_name)
                print(f"✓ Hourly checkpoint saved: {checkpoint_name}")
            elif reason == "interrupt":
                checkpoint_name = "interrupt_recovery.pt"
                torch.save(checkpoint, self.checkpoint_dir / checkpoint_name)
                print(f"✓ Interrupt recovery checkpoint saved")

            # Save periodic epoch checkpoint
            if (self.current_epoch + 1) % self.config.get("save_interval", 1) == 0 and reason == "periodic":
                torch.save(
                    checkpoint, self.checkpoint_dir / f"epoch_{self.current_epoch + 1}.pt"
                )

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint with full training state"""
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.current_batch = checkpoint.get("batch", 0)  # Resume from exact batch
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.patience_counter = checkpoint.get("patience_counter", 0)

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        reason = checkpoint.get("reason", "unknown")
        timestamp = checkpoint.get("timestamp", "unknown")
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}, batch {self.current_batch}")
        print(f"  Checkpoint reason: {reason}, saved at: {timestamp}")
        print(f"  Best val loss: {self.best_val_loss:.4f}, Global step: {self.global_step}")

    def train(self):
        """Main training loop"""
        max_epochs = self.config.get("max_epochs", 10)
        early_stop_patience = self.config.get("early_stop_patience", None)

        print(f"\n{'='*60}")
        print(f"Starting training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Device Type: {'TPU' if self.is_tpu else 'GPU/CPU'}")
        print(f"Model parameters: {self.model.get_num_params() / 1e6:.2f}M")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print(f"Max epochs: {max_epochs}")
        print(f"Mixed precision: {self.use_amp} ({'bfloat16' if self.is_tpu else 'float16'})")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(max_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Update learning rate
            self.scheduler.step()

            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{max_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")

            # Log to wandb
            if self.use_wandb:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/epoch_loss": train_loss,
                        "val/loss": val_loss,
                        "val/best_loss": self.best_val_loss,
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                    }
                )

            # Save checkpoint
            self.save_checkpoint(is_best=is_best, reason="epoch_end")

            # Early stopping
            if early_stop_patience and self.patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                self.save_checkpoint(is_best=False, reason="early_stop")
                break
            
            # Check if interrupted
            if self.interrupted:
                print("\nTraining interrupted by user")
                break

        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total time: {elapsed_time / 3600:.2f} hours")
        print(f"{'='*60}")


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description="Train language model")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data file or directory",
    )
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", 
        choices=["cuda", "cpu", "tpu"],
        help="Device to use (cuda/cpu/tpu)"
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Use wandb for logging"
    )

    args = parser.parse_args()

    # Load or create config
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        # Optimized config for Colab T4 GPU (15GB VRAM)
        # Target: ~570M parameters for safe training
        config = {
            "model": {
                "context_length": 1024,  # Reduced to save memory
                "n_layers": 24,  # Keep depth
                "n_heads": 16,  # Standard head count
                "n_kv_heads": 4,  # GQA 4x for efficient inference & smaller KV cache
                "n_embd": 1280,  # Reduced from 1536 (~570M params total)
                "dropout": 0.1,
                "use_gradient_checkpointing": True,  # CRITICAL for memory
            },
            "training": {
                "batch_size": 1,  # Minimum for T4 GPU
                "gradient_accumulation_steps": 64,  # Increased to maintain effective batch
                "max_epochs": 10,
                "learning_rate": 6e-4,  # Higher LR for larger models
                "weight_decay": 0.1,
                "max_grad_norm": 1.0,
                "use_amp": True,  # Mixed precision (FP16) essential
                "early_stop_patience": 3,
                "save_interval": 1,  # Save every epoch
            },
            "data": {
                "block_size": 1024,  # Match context_length
                "stride": 512,  # 50% overlap
            },
        }

    device = args.device

    # Validate TPU availability
    if device == "tpu":
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            print("✓ TPU libraries available")
        except ImportError:
            print("Error: TPU requested but torch_xla not installed")
            print("Install with: pip install torch_xla")
            return

    # Load data
    print("Loading data...")
    from data_utils import load_text_file, load_directory

    if os.path.isfile(args.data):
        texts = load_text_file(args.data)
    else:
        texts = load_directory(args.data)

    print(f"Loaded {len(texts)} texts")

    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.train(texts)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create dataset
    print("Creating dataset...")
    dataset = TextDataset(
        texts,
        tokenizer,
        block_size=config["data"]["block_size"],
        stride=config["data"].get("stride", config["data"]["block_size"]),
    )

    train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.9)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create data loaders
    # For TPU, reduce num_workers to avoid issues
    num_workers = 0 if device == "tpu" else 0
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    # Create model
    print("Creating model...")
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, **config["model"])
    model = LanguageModel(model_config)

    # Create trainer
    training_config = {
        **config["training"],
        "checkpoint_dir": "checkpoints",
        "wandb_project": "llm-training",
        "run_name": f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    }

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        training_config,
        device=device,
        use_wandb=args.use_wandb,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
