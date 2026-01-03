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
# Disable tokenizers parallelism to avoid deadlocks in forked processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Handle GradScaler deprecation
try:
    from torch.amp import GradScaler, autocast as cuda_autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast as cuda_autocast

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
from data_utils import load_text_file, load_directory
from data_utils_optimized import StreamingTextDataset

from profiler import TrainingProfiler, RICH_AVAILABLE
if RICH_AVAILABLE:
    from rich.live import Live


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
        enable_profiling: bool = False,
    ):
        self.config = config
        self.use_wandb = use_wandb
        self.enable_profiling = enable_profiling
        
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
            # Initialize GradScaler with 'cuda' device if using new API, or default for old
            if self.use_amp:
                try:
                    self.scaler = GradScaler('cuda')
                except TypeError:
                    self.scaler = GradScaler()
            else:
                self.scaler = None

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
        
        # Profiler
        self.profiler = TrainingProfiler()
        
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

        try:
            total_batches = len(train_loader)
        except TypeError:
            total_batches = None
        
        # Skip to current batch if resuming
        if self.current_batch > 0:
            if total_batches:
                print(f"\nResuming from batch {self.current_batch}/{total_batches}")
            else:
                print(f"\nResuming from batch {self.current_batch}")
                
            # Create iterator and skip ahead
            train_iter = iter(train_loader)
            for _ in tqdm(range(self.current_batch), desc="Skipping batches"):
                try:
                    next(train_iter)
                except StopIteration:
                    break
            # Use iterator directly (don't convert to list to save RAM)
            train_loader = train_iter
            start_batch = self.current_batch
        else:
            start_batch = 0
        
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {self.current_epoch + 1}",
            initial=start_batch,
            total=total_batches
        )

        # Live display context
        live_ctx = Live(self.profiler.generate_table(), refresh_per_second=4) if (RICH_AVAILABLE and self.enable_profiling) else None
        
        if live_ctx:
            live_ctx.start()
            
        try:
            self.profiler.start("Data Loading")
            
            for batch_idx, (x, y) in enumerate(pbar, start=start_batch):
                self.profiler.stop("Data Loading")
                self.current_batch = batch_idx  # Update current batch for checkpoints
                
                if not self.is_tpu:
                    x, y = x.to(self.device), y.to(self.device)
    
                # Forward pass with mixed precision
                self.profiler.start("Forward Pass")
                if self.use_amp:
                    if self.is_tpu:
                        # TPU: use XLA autocast with bfloat16
                        with torch.autocast(device_type='xla', dtype=torch.bfloat16):
                            logits, loss, _ = self.model(x, y)  # Ignore KV cache during training
                    else:
                        # CUDA: use standard autocast, preferring bfloat16 if available (Ampere+)
                        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                        with cuda_autocast("cuda", dtype=dtype):
                            logits, loss, _ = self.model(x, y)  # Ignore KV cache during training
                            
                    loss = loss / self.config.get("gradient_accumulation_steps", 1)
                else:
                    logits, loss, _ = self.model(x, y)  # Ignore KV cache during training
                    loss = loss / self.config.get("gradient_accumulation_steps", 1)
                self.profiler.stop("Forward Pass")
    
                # Backward pass
                self.profiler.start("Backward Pass")
                if self.use_amp and not self.is_tpu:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                self.profiler.stop("Backward Pass")
    
                # Gradient accumulation
                if (batch_idx + 1) % self.config.get("gradient_accumulation_steps", 1) == 0:
                    self.profiler.start("Optimizer Step")
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
                    self.profiler.stop("Optimizer Step")
                
                # Mark step after backward for gradient accumulation
                elif self.is_tpu:
                    self.xm.mark_step()
    
                total_loss += loss.item() * self.config.get(
                    "gradient_accumulation_steps", 1
                )
                num_batches += 1
                
                self.profiler.step()
    
                # Update progress bar
                if RICH_AVAILABLE and live_ctx:
                    live_ctx.update(self.profiler.generate_table())
                else:
                    pbar.set_postfix(
                        {
                            "loss": f'{loss.item() * self.config.get("gradient_accumulation_steps", 1):.4f}',
                            "lr": f"{self.scheduler.get_last_lr()[0]:.6f}",
                        }
                    )
                
                # Start timing next data load
                self.profiler.start("Data Loading")

        finally:
            if live_ctx:
                live_ctx.stop()

            # Log to wandb
            if (
                self.use_wandb
                and self.global_step % self.config.get("log_interval", 100) == 0
            ):
                import wandb
                
                progress_str = f"{batch_idx}/{total_batches}" if total_batches else f"{batch_idx}"

                wandb.log(
                    {
                        "train/loss": loss.item()
                        * self.config.get("gradient_accumulation_steps", 1),
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/step": self.global_step,
                        "train/epoch_progress": progress_str,
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

        try:
            total_batches = len(val_loader)
        except TypeError:
            total_batches = None
            
        pbar = tqdm(val_loader, desc="Validating", total=total_batches)

        for x, y in pbar:
            if not self.is_tpu:
                x, y = x.to(self.device), y.to(self.device)

            if self.use_amp:
                if self.is_tpu:
                    # TPU: use XLA autocast with bfloat16
                    with torch.autocast(device_type='xla', dtype=torch.bfloat16):
                        _, loss, _ = self.model(x, y)  # Ignore logits and KV cache
                else:
                    with cuda_autocast("cuda"):
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
                checkpoint_name = f"interrupt_epoch{self.current_epoch}_batch{self.current_batch}.pt"
                torch.save(checkpoint, self.checkpoint_dir / checkpoint_name)
                print(f"✓ Interrupt checkpoint saved: {checkpoint_name}")
            elif reason == "early_stop":
                checkpoint_name = f"early_stop_epoch{self.current_epoch}.pt"
                torch.save(checkpoint, self.checkpoint_dir / checkpoint_name)
                print(f"✓ Early stop checkpoint saved: {checkpoint_name}")

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
        
        # Try to infer epoch/batch from filename if they seem wrong (e.g. 0)
        # This helps if the user renamed the file or if saving logic had a bug
        try:
            import re
            filename = os.path.basename(checkpoint_path)
            # Look for patterns like "epoch1_batch2000" or "epoch_1_batch_2000"
            match = re.search(r"epoch[_]?(\d+)[_]?batch[_]?(\d+)", filename)
            if match:
                filename_epoch = int(match.group(1))
                filename_batch = int(match.group(2))
                
                if filename_batch != self.current_batch:
                    print(f"⚠️  Warning: Checkpoint content batch ({self.current_batch}) matches filename batch ({filename_batch}) mismatch.")
                    print(f"   Trusting filename and updating current_batch to {filename_batch}")
                    self.current_batch = filename_batch
                    
                if filename_epoch != self.current_epoch:
                    print(f"⚠️  Warning: Checkpoint content epoch ({self.current_epoch}) matches filename epoch ({filename_epoch}) mismatch.")
                    print(f"   Trusting filename and updating current_epoch to {filename_epoch}")
                    self.current_epoch = filename_epoch
        except Exception as e:
            print(f"Warning: Could not infer batch/epoch from filename: {e}")

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
        try:
            print(f"Training batches: {len(self.train_loader)}")
        except TypeError:
            print("Training batches: Unknown (Streaming)")
            
        try:
            print(f"Validation batches: {len(self.val_loader)}")
        except TypeError:
            print("Validation batches: Unknown (Streaming)")
        print(f"Max epochs: {max_epochs}")
        print(f"Mixed precision: {self.use_amp} ({'bfloat16' if self.is_tpu else 'float16'})")
        print(f"{'='*60}\n")
        
        # Reset checkpoint timer to avoid immediate save if setup took long
        self.last_checkpoint_time = time.time()

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
    parser.add_argument("--use-wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--tokenizer", type=str, help="Path to pre-trained tokenizer JSON")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode (slower, for massive datasets)")
    parser.add_argument("--profile", action="store_true", help="Enable rich profiling output")

    args = parser.parse_args()

    # Load or create config
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        # Optimized for FREE T4 GPU (15GB) with resume headroom
        # Target: ~700M params, ~13.5GB fresh, ~14GB resume
        config = {
            "model": {
                "context_length": 2048,  # Keep long context
                "n_layers": 24,  # Keep depth
                "n_heads": 16,  # Standard head count
                "n_kv_heads": 4,  # GQA 4x for efficiency
                "n_embd": 1408,  # Reduced from 1536 (~700M params)
                "dropout": 0.1,
                "use_gradient_checkpointing": True,  # CRITICAL for memory
            },
            "training": {
                "batch_size": 1,  # Minimum for free GPU
                "gradient_accumulation_steps": 32,  # Effective batch = 32
                "max_epochs": 10,
                "learning_rate": 6e-4,  # Higher LR for larger models
                "weight_decay": 0.1,
                "max_grad_norm": 1.0,
                "use_amp": True,  # Mixed precision (FP16) essential
                "early_stop_patience": 3,
                "save_interval": 1,  # Save every epoch
            },
            "data": {
                "block_size": 2048,  # Match context_length
                "stride": 1024,  # 50% overlap for better learning
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

    # 1. Initialize Tokenizer (Must be done before dataset creation)
    print("Initializing tokenizer...")
    # Use BPETokenizer for better quality
    tokenizer = BPETokenizer(vocab_size=config["model"].get("vocab_size", 50257))
    
    if args.tokenizer and os.path.exists(args.tokenizer):
        print(f"Loading tokenizer from {args.tokenizer}...")
        tokenizer.load(args.tokenizer)
        print(f"Vocabulary size: {tokenizer.vocab_size}")
    else:
        # Need to train tokenizer
        print("No pre-trained tokenizer provided. Training from scratch...")
        
        # Check if streaming
        if "/" in args.data and not os.path.exists(args.data):
             print(f"Training tokenizer on sample from streaming dataset: {args.data}")
             # Load a small sample to train tokenizer
             try:
                 from datasets import load_dataset
                 
                 # Handle subsets like "HuggingFaceFW/fineweb-edu/sample-10BT"
                 if "/" in args.data and len(args.data.split("/")) > 2:
                     parts = args.data.split("/")
                     repo = "/".join(parts[:2])
                     subset = "/".join(parts[2:])
                     print(f"Loading dataset: {repo}, subset: {subset}")
                     ds_sample = load_dataset(repo, subset, split="train", streaming=True).take(10000)
                 else:
                     ds_sample = load_dataset(args.data, split="train", streaming=True).take(10000)
                     
                 sample_texts = [item["text"] for item in ds_sample if item["text"]]
                 print(f"Collected {len(sample_texts)} samples for tokenizer training")
                 tokenizer.train(sample_texts)
                 
                 # Clean up memory
                 del ds_sample
                 del sample_texts
                 import gc
                 gc.collect()
                 
             except Exception as e:
                 print(f"Error training tokenizer on stream: {e}")
                 print("Please provide a pre-trained tokenizer with --tokenizer")
                 return
        else:
            # Standard file loading for tokenizer training
            pass
            
        from data_utils import load_text_file, load_directory
        if os.path.isfile(args.data):
            # Optimization: If file is large (>100MB), sample it for tokenizer training
            file_size = os.path.getsize(args.data)
            if file_size > 100 * 1024 * 1024:  # 100MB
                print(f"File is large ({file_size / 1024 / 1024:.2f} MB). Sampling for tokenizer training...")
                with open(args.data, 'r', encoding='utf-8') as f:
                    # Read first 10MB or 50k lines
                    sample_lines = []
                    for _ in range(50000):
                        line = f.readline()
                        if not line: break
                        sample_lines.append(line)
                texts = sample_lines
                print(f"Sampled {len(texts)} lines for tokenizer training")
            else:
                texts = load_text_file(args.data)
        else:
            texts = load_directory(args.data)
        
        if hasattr(tokenizer, 'train'):
             tokenizer.train(texts)
            
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        
        # Save tokenizer
        tokenizer.save("tokenizer.json")
        
        # Also save to checkpoint directory for portability
        ckpt_dir = config["training"].get("checkpoint_dir", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_tokenizer_path = os.path.join(ckpt_dir, "tokenizer.json")
        tokenizer.save(ckpt_tokenizer_path)

    # 2. Load Data & Create Dataset
    print("Loading data...")
    
    # Check if args.data is a HuggingFace dataset (contains '/')
    if "/" in args.data and not os.path.exists(args.data):
        print(f"Detected HuggingFace dataset: {args.data}")
        
        if args.stream:
            # Use streaming mode (slower, for massive datasets)
            print("Using STREAMING mode (slower)")
            from data_utils_optimized import StreamingTextDataset
            train_dataset = StreamingTextDataset(
                [args.data], 
                tokenizer, 
                block_size=config["data"]["block_size"],
                split="train"
            )
            # For streaming, we can't easily split validation, so we use the same stream or a different split
            try:
                val_dataset = StreamingTextDataset(
                    [args.data], 
                    tokenizer, 
                    block_size=config["data"]["block_size"],
                    split="validation" # Try to load validation split
                )
            except:
                print("Warning: No validation split found, using train split for validation (not ideal)")
                val_dataset = train_dataset
        else:
            # Use cached mode (FAST, recommended)
            print("Using CACHED mode (fast, downloads once)")
            from cached_dataset import CachedHFDataset
            train_dataset = CachedHFDataset(
                args.data,
                tokenizer,
                block_size=config["data"]["block_size"],
                split="train",
                max_samples=config["data"].get("max_samples", None)  # Use config limit or all samples
            )
            # Try to load validation split
            try:
                val_dataset = CachedHFDataset(
                    args.data,
                    tokenizer,
                    block_size=config["data"]["block_size"],
                    split="validation",
                    max_samples=min(10000, config["data"].get("max_samples", 10000))  # Limit val set size
                )
            except:
                print("Warning: No validation split found, using train split for validation (not ideal)")
                # Use a subset of train for validation
                from torch.utils.data import Subset
                import numpy as np
                val_size = min(10000, len(train_dataset) // 10)
                val_indices = np.random.choice(len(train_dataset), val_size, replace=False)
                val_dataset = Subset(train_dataset, val_indices)
            
        dataset = None # Marker that we are using streaming
    else:
        # Standard file loading
        # We already loaded 'texts' above if we trained tokenizer, but let's be safe and reload or reuse
        # Optimization: Reuse 'texts' if we have them to avoid double loading
        if 'texts' not in locals():
            from data_utils import load_text_file, load_directory  # Ensure imports are available
            if os.path.isfile(args.data):
                texts = load_text_file(args.data)
            else:
                texts = load_directory(args.data)
        
        print(f"Creating dataset with {len(texts)} texts...")
        dataset = TextDataset(
            texts,
            tokenizer,
            block_size=config["data"]["block_size"],
            stride=config["data"].get("stride", config["data"]["block_size"]),
            lazy=False,  # Optimization: Pre-tokenize everything for speed since we fit in RAM
        )
        train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.9)

    # Create data loaders
    # For TPU, reduce num_workers to avoid issues
    num_workers = 0 if device == "tpu" else 1
    
    # Streaming datasets don't support shuffle=True in DataLoader
    # Safe check for streaming dataset type
    try:
        from data_utils_optimized import StreamingTextDataset
        is_streaming = isinstance(train_dataset, StreamingTextDataset)
    except ImportError:
        is_streaming = False
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=not is_streaming, # Disable shuffle for streaming (it's internal)
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
    # Ensure vocab size matches tokenizer
    config["model"]["vocab_size"] = tokenizer.vocab_size
    model_config = ModelConfig(**config["model"])
    model = LanguageModel(model_config)

    # Create trainer
    training_config = {
        **config["training"],
        "wandb_project": "llm-training",
        "run_name": f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        "model": config["model"],
        "data": config["data"],
    }

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        training_config,
        device=device,
        use_wandb=args.use_wandb,
        enable_profiling=args.profile,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        # Verify tokenizer match
        if trainer.model.token_embedding.weight.shape[0] != tokenizer.vocab_size:
            print(f"⚠️ Warning: Model vocab size ({trainer.model.token_embedding.weight.shape[0]}) != Tokenizer vocab size ({tokenizer.vocab_size})")
            print("   This might cause errors. Consider rebuilding tokenizer or model.")

    # Train
    trainer.train()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
