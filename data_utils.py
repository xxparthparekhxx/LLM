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
        text_column: str = "text"
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.text_column = text_column
        
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
        for item in iterator:
            try:
                text = item[self.text_column]
                if not text: continue
                token_buffer.extend(self.tokenizer.encode(text))
                while len(token_buffer) >= self.block_size + 1:
                    yield torch.tensor(token_buffer[:self.block_size], dtype=torch.long), torch.tensor(token_buffer[1:self.block_size+1], dtype=torch.long)
                    token_buffer = token_buffer[self.block_size:]
            except: continue
            
class TextDataset(Dataset):
    """Dataset for language modeling with lazy tokenization"""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        block_size: int,
        stride: Optional[int] = None,
        lazy: bool = True
    ):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer instance with encode method
            block_size: Maximum sequence length
            stride: Stride for sliding window (if None, uses block_size)
            lazy: If True, tokenize on-the-fly (much faster initialization)
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size
        self.lazy = lazy
        
        if not lazy:
            # Old behavior: tokenize everything upfront (SLOW for large datasets)
            print("Tokenizing all texts (this may take a while)...")
            self.tokens = []
            for i, text in enumerate(texts):
                if i % 1000 == 0:
                    print(f"  Tokenized {i}/{len(texts)} texts...")
                tokens = tokenizer.encode(text)
                self.tokens.extend(tokens)
            
            # Create sequences with sliding window
            self.sequences = []
            for i in range(0, len(self.tokens) - block_size, self.stride):
                seq = self.tokens[i:i + block_size + 1]  # +1 for target
                if len(seq) == block_size + 1:
                    self.sequences.append(seq)
            
            self.tokens = None  # Free memory
        else:
            # New behavior: lazy tokenization (FAST)
            # Just store text indices and compute sequences on-demand
            self.sequences = None
            self.tokens = None
            
            # Pre-compute approximate number of sequences
            # Estimate: avg 4 chars per token
            total_chars = sum(len(t) for t in texts[:min(100, len(texts))])
            avg_chars = total_chars / min(100, len(texts))
            avg_tokens_per_text = avg_chars / 4
            
            # Each text can produce roughly (tokens - block_size) / stride sequences
            self._estimated_len = max(1, int(len(texts) * avg_tokens_per_text / stride))
    
    def __len__(self):
        if self.sequences is not None:
            return len(self.sequences)
        return self._estimated_len
    
    def __getitem__(self, idx):
        if self.sequences is not None:
            # Non-lazy mode
            seq = self.sequences[idx]
            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            return x, y
        else:
            # Lazy mode: tokenize on-demand
            # Map global idx to text and position within text
            text_idx = idx % len(self.texts)
            text = self.texts[text_idx]
            
            # Tokenize this text
            tokens = self.tokenizer.encode(text)
            
            # Get a random window from this text
            if len(tokens) <= self.block_size:
                # Pad if too short
                seq = tokens + [self.tokenizer.pad_token_id] * (self.block_size + 1 - len(tokens))
            else:
                # Random crop
                max_start = len(tokens) - self.block_size - 1
                start = torch.randint(0, max(1, max_start), (1,)).item()
                seq = tokens[start:start + self.block_size + 1]
            
            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            return x, y


class MemoryMappedDataset(Dataset):
    """
    Memory-mapped dataset for very large corpora
    Tokenizes once, saves to disk, then loads via mmap (fastest for repeated use)
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        block_size: int,
        cache_dir: str = '.cache',
        force_reprocess: bool = False
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cache filename based on content hash
        cache_key = f"tokens_{len(texts)}_{block_size}"
        self.cache_file = self.cache_dir / f"{cache_key}.npy"
        
        if self.cache_file.exists() and not force_reprocess:
            print(f"Loading cached tokens from {self.cache_file}...")
            self.tokens = np.load(str(self.cache_file), mmap_mode='r')
        else:
            print(f"Processing and caching tokens to {self.cache_file}...")
            all_tokens = []
            for i, text in enumerate(texts):
                if i % 1000 == 0:
                    print(f"  Processed {i}/{len(texts)} texts...")
                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)
            
            # Save to disk
            tokens_array = np.array(all_tokens, dtype=np.uint16)  # uint16 for vocab < 65536
            np.save(str(self.cache_file), tokens_array)
            
            # Load as memory-mapped
            self.tokens = np.load(str(self.cache_file), mmap_mode='r')
            print(f"Cached {len(self.tokens)} tokens")
        
        # Calculate number of sequences
        self.n_sequences = max(1, (len(self.tokens) - block_size) // block_size)
    
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size + 1
        
        if end_idx > len(self.tokens):
            # Wrap around
            seq = np.concatenate([
                self.tokens[start_idx:],
                self.tokens[:end_idx - len(self.tokens)]
            ])
        else:
            seq = self.tokens[start_idx:end_idx]
        
        x = torch.from_numpy(seq[:-1].astype(np.int64))
        y = torch.from_numpy(seq[1:].astype(np.int64))
        return x, y


def load_text_file(filepath: str) -> List[str]:
    """Load texts from a file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newlines (paragraphs)
    texts = [t.strip() for t in content.split('\n\n') if t.strip()]
    if not texts:
        # Fall back to single newlines
        texts = [t.strip() for t in content.split('\n') if t.strip()]
    
    return texts


def load_jsonl(filepath: str, text_key: str = 'text') -> List[str]:
    """Load texts from JSONL file"""
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if text_key in data:
                texts.append(data[text_key])
    return texts


def load_directory(directory: str, extensions: List[str] = None) -> List[str]:
    """Load all text files from a directory"""
    if extensions is None:
        extensions = ['.txt', '.md', '.py', '.json']
    
    texts = []
    directory = Path(directory)
    
    for ext in extensions:
        for filepath in directory.rglob(f'*{ext}'):
            try:
                if ext == '.json':
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'text' in data:
                            texts.append(data['text'])
                        elif isinstance(data, list):
                            texts.extend([str(item) for item in data])
                else:
                    file_texts = load_text_file(str(filepath))
                    texts.extend(file_texts)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
    
    return texts


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """Create a DataLoader with sensible defaults"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )


def split_dataset(dataset: Dataset, train_ratio: float = 0.9) -> Tuple[Dataset, Dataset]:
    """Split dataset into train and validation sets"""
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset


class DataCollator:
    """Collate function for batching"""
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate a batch of sequences"""
        xs, ys = zip(*batch)
        
        # All sequences should be same length in TextDataset
        # but this handles edge cases
        max_len = max(x.size(0) for x in xs)
        
        x_batch = []
        y_batch = []
        
        for x, y in zip(xs, ys):
            pad_len = max_len - x.size(0)
            if pad_len > 0:
                x = torch.cat([x, torch.full((pad_len,), self.pad_token_id, dtype=x.dtype)])
                y = torch.cat([y, torch.full((pad_len,), -1, dtype=y.dtype)])
            x_batch.append(x)
            y_batch.append(y)
        
        return torch.stack(x_batch), torch.stack(y_batch)


if __name__ == "__main__":
    # Test dataset
    from tokenizer import SimpleTokenizer
    
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world! This is a test.",
        "Machine learning is fascinating."
    ] * 1000  # More texts to test lazy loading
    
    tokenizer = SimpleTokenizer()
    tokenizer.train(texts)
    
    print("\n=== Testing Lazy Dataset (FAST) ===")
    import time
    start = time.time()
    dataset = TextDataset(texts, tokenizer, block_size=32, stride=16, lazy=True)
    print(f"Lazy dataset created in {time.time() - start:.2f}s")
    print(f"Dataset size: {len(dataset)}")
    
    print("\n=== Testing Non-Lazy Dataset (SLOW) ===")
    start = time.time()
    dataset2 = TextDataset(texts[:100], tokenizer, block_size=32, stride=16, lazy=False)
    print(f"Non-lazy dataset created in {time.time() - start:.2f}s")
    print(f"Dataset size: {len(dataset2)}")
    
    dataloader = create_dataloader(dataset, batch_size=4, shuffle=False, num_workers=0)
    
    print("\n=== Testing DataLoader ===")
    for i, (x, y) in enumerate(dataloader):
        print(f"Batch {i}: x shape={x.shape}, y shape={y.shape}")
        if i >= 2:
            break