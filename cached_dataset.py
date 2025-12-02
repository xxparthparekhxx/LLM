"""
Cached dataset implementation for fast HuggingFace dataset loading.
Downloads once, uses local cache for subsequent runs.
"""

import torch
from torch.utils.data import Dataset
from typing import List
from datasets import load_dataset

class CachedHFDataset(Dataset):
    """
    Dataset that downloads and caches HuggingFace datasets.
    Much faster than streaming for repeated training runs.
    """
    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        block_size: int,
        split: str = "train",
        text_column: str = "text",
        max_samples: int = None,
        cache_dir: str = None
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_column = text_column
        
        print(f"Loading dataset: {dataset_name} ({split})...")
        print("Note: First run will download and cache. Subsequent runs will be instant!")
        
        # Handle subsets like "HuggingFaceFW/fineweb-edu/sample-10BT"
        if "/" in dataset_name and len(dataset_name.split("/")) > 2:
            parts = dataset_name.split("/")
            repo = "/".join(parts[:2])
            subset = "/".join(parts[2:])
            self.dataset = load_dataset(
                repo, 
                subset, 
                split=split,
                cache_dir=cache_dir
            )
        else:
            self.dataset = load_dataset(
                dataset_name, 
                split=split,
                cache_dir=cache_dir
            )
        
        # Limit samples if specified
        if max_samples and len(self.dataset) > max_samples:
            print(f"Limiting to {max_samples} samples (out of {len(self.dataset)})")
            self.dataset = self.dataset.select(range(max_samples))
        
        print(f"Dataset loaded: {len(self.dataset)} samples")
        
        # Pre-tokenize and cache
        print("Pre-tokenizing dataset (this may take a few minutes on first run)...")
        self._prepare_sequences()
        
    def _prepare_sequences(self):
        """Pre-tokenize all texts and create sequences"""
        self.sequences = []
        
        # Process in batches for speed
        batch_size = 1000
        for i in range(0, len(self.dataset), batch_size):
            batch_end = min(i + batch_size, len(self.dataset))
            batch = self.dataset[i:batch_end]
            
            # Get texts
            if isinstance(batch[self.text_column], list):
                texts = batch[self.text_column]
            else:
                texts = [batch[self.text_column]]
            
            # Batch encode
            if hasattr(self.tokenizer, 'encode_batch'):
                batch_tokens = self.tokenizer.encode_batch(texts)
            else:
                batch_tokens = [self.tokenizer.encode(text) for text in texts]
            
            # Create sequences from tokens
            for tokens in batch_tokens:
                # Split into block_size chunks
                for j in range(0, len(tokens) - self.block_size, self.block_size):
                    seq = tokens[j:j + self.block_size + 1]
                    if len(seq) == self.block_size + 1:
                        self.sequences.append(seq)
            
            if (i + batch_size) % 10000 == 0:
                print(f"  Processed {min(i + batch_size, len(self.dataset))}/{len(self.dataset)} samples, {len(self.sequences)} sequences created")
        
        print(f"Pre-tokenization complete: {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y
