"""
Advanced training pipeline for language model
Features:
- Gradient accumulation
- Mixed precision training
- Learning rate scheduling
- Checkpointing
- Wandb integration (optional)
- Early stopping
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
import json
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any
import time
from datetime import datetime

from model import LanguageModel, ModelConfig
from data_utils import TextDataset, create_dataloader, split_dataset
from tokenizer import SimpleTokenizer, BPETokenizer


class Trainer:
    """Advanced trainer for language model"""
    
    def __init__(
        self,
        model: LanguageModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = 'cuda',
        use_wandb: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 3e-4),
            betas=config.get('betas', (0.9, 0.95)),
            weight_decay=config.get('weight_decay', 0.1),
            eps=config.get('eps', 1e-8)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('max_epochs', 10),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True) and device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if requested
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.get('wandb_project', 'llm-training'),
                    config=config,
                    name=config.get('run_name', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                )
            except ImportError:
                print("Warning: wandb not installed, continuing without it")
                self.use_wandb = False
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    logits, loss = self.model(x, y)
                    loss = loss / self.config.get('gradient_accumulation_steps', 1)
            else:
                logits, loss = self.model(x, y)
                loss = loss / self.config.get('gradient_accumulation_steps', 1)
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('max_grad_norm', 1.0)
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('max_grad_norm', 1.0)
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            total_loss += loss.item() * self.config.get('gradient_accumulation_steps', 1)
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item() * self.config.get("gradient_accumulation_steps", 1):.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
            
            # Log to wandb
            if self.use_wandb and self.global_step % self.config.get('log_interval', 100) == 0:
                import wandb
                wandb.log({
                    'train/loss': loss.item() * self.config.get('gradient_accumulation_steps', 1),
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    'train/step': self.global_step
                })
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc="Validating")
        
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            if self.use_amp:
                with autocast():
                    _, loss = self.model(x, y)
            else:
                _, loss = self.model(x, y)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
        
        # Save periodic checkpoint
        if (self.current_epoch + 1) % self.config.get('save_interval', 5) == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{self.current_epoch + 1}.pt')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop"""
        max_epochs = self.config.get('max_epochs', 10)
        early_stop_patience = self.config.get('early_stop_patience', None)
        
        print(f"\n{'='*60}")
        print(f"Starting training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.get_num_params() / 1e6:.2f}M")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print(f"Max epochs: {max_epochs}")
        print(f"Mixed precision: {self.use_amp}")
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
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_loss,
                    'val/loss': val_loss,
                    'val/best_loss': self.best_val_loss,
                    'train/learning_rate': self.scheduler.get_last_lr()[0]
                })
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if early_stop_patience and self.patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
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
    
    parser = argparse.ArgumentParser(description='Train language model')
    parser.add_argument('--data', type=str, required=True, help='Path to training data file or directory')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--use-wandb', action='store_true', help='Use wandb for logging')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default config
        config = {
            'model': {
                # 'vocab_size': 10000,
                'context_length': 512,
                'n_layers': 6,
                'n_heads': 8,
                'n_embd': 512,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 8,
                'gradient_accumulation_steps': 4,
                'max_epochs': 10,
                'learning_rate': 3e-4,
                'weight_decay': 0.1,
                'max_grad_norm': 1.0,
                'use_amp': True,
                'early_stop_patience': 3,
                'save_interval': 5
            },
            'data': {
                'block_size': 512,
                'stride': 256
            }
        }
    
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    
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
        block_size=config['data']['block_size'],
        stride=config['data'].get('stride', config['data']['block_size'])
    )
    
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.9)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        **config['model']
    )
    model = LanguageModel(model_config)
    
    # Create trainer
    training_config = {
        **config['training'],
        'checkpoint_dir': 'checkpoints',
        'wandb_project': 'llm-training',
        'run_name': f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        training_config,
        device=device,
        use_wandb=args.use_wandb
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

