"""
Data processing utilities for language model training
Fixed with lazy loading and efficient processing
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List, Optional, Tuple
import os
import json
from pathlib import Path
import numpy as np

try:
    from datasets import load_dataset, interleave_datasets, IterableDataset as HFIterableDataset
except ImportError:
    print("Warning: 'datasets' library not found. Streaming datasets will not work.")

class StreamingTextDataset(IterableDataset):
    """
    Dataset for streaming text from HuggingFace datasets without loading into RAM.
    Optimized with batch encoding for faster tokenization.
    """
    def __init__(
        self,
        dataset_names: List[str],
        tokenizer,
        block_size: int,
        split: str = "train",
        probs: Optional[List[float]] = None,
        seed: int = 42,
        buffer_size: int = 10000,
        text_column: str = "text",
        batch_encode_size: int = 32  # Batch size for tokenization
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.text_column = text_column
        self.batch_encode_size = batch_encode_size
        
        # Load datasets in streaming mode
        self.datasets = []
        for name in dataset_names:
            print(f"Loading streaming dataset: {name} ({split})...")
            # Handle subsets like "HuggingFaceFW/fineweb-edu/sample-10BT"
            if "/" in name and len(name.split("/")) > 2:
                parts = name.split("/")
                repo = "/".join(parts[:2])
                subset = "/".join(parts[2:])
                ds = load_dataset(repo, subset, split=split, streaming=True)
            else:
                ds = load_dataset(name, split=split, streaming=True)
            self.datasets.append(ds)
            
        if len(self.datasets) > 1:
            self.dataset = interleave_datasets(self.datasets, probabilities=probs, seed=seed)
        else:
            self.dataset = self.datasets[0]
            
        self.dataset = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)
        
    def __iter__(self):
        iterator = iter(self.dataset)
        token_buffer = []
        text_batch = []
        
        for item in iterator:
            try:
                text = item[self.text_column]
                if not text: 
                    continue
                
                # Accumulate texts for batch encoding
                text_batch.append(text)
                
                # When batch is full, encode all texts at once
                if len(text_batch) >= self.batch_encode_size:
                    # Batch encode for speed
                    if hasattr(self.tokenizer, 'encode_batch'):
                        batch_tokens = self.tokenizer.encode_batch(text_batch)
                        for tokens in batch_tokens:
                            token_buffer.extend(tokens)
                    else:
                        # Fallback to one-by-one
                        for txt in text_batch:
                            token_buffer.extend(self.tokenizer.encode(txt))
                    
                    text_batch = []
                    
                    # Yield sequences from buffer
                    while len(token_buffer) >= self.block_size + 1:
                        yield torch.tensor(token_buffer[:self.block_size], dtype=torch.long), torch.tensor(token_buffer[1:self.block_size+1], dtype=torch.long)
                        token_buffer = token_buffer[self.block_size:]
                        
            except: 
                continue
        
        # Process remaining texts in batch
        if text_batch:
            if hasattr(self.tokenizer, 'encode_batch'):
                batch_tokens = self.tokenizer.encode_batch(text_batch)
                for tokens in batch_tokens:
                    token_buffer.extend(tokens)
            else:
                for txt in text_batch:
                    token_buffer.extend(self.tokenizer.encode(txt))
        
        # Yield remaining sequences
        while len(token_buffer) >= self.block_size + 1:
            yield torch.tensor(token_buffer[:self.block_size], dtype=torch.long), torch.tensor(token_buffer[1:self.block_size+1], dtype=torch.long)
            token_buffer = token_buffer[self.block_size:]
