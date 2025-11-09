"""
Data processing utilities for language model training
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple
import os
import json
from pathlib import Path


class TextDataset(Dataset):
    """Dataset for language modeling"""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        block_size: int,
        stride: Optional[int] = None
    ):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer instance with encode method
            block_size: Maximum sequence length
            stride: Stride for sliding window (if None, uses block_size)
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size
        
        # Tokenize all texts
        self.tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            self.tokens.extend(tokens)
        
        # Create sequences with sliding window
        self.sequences = []
        for i in range(0, len(self.tokens) - block_size + 1, self.stride):
            seq = self.tokens[i:i + block_size + 1]  # +1 for target
            if len(seq) == block_size + 1:
                self.sequences.append(seq)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


def load_text_file(filepath: str) -> List[str]:
    """Load texts from a file (one per line or paragraph)"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newlines (paragraphs) or single newlines
    texts = [t.strip() for t in content.split('\n\n') if t.strip()]
    if not texts:
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
        drop_last=True  # Drop last incomplete batch
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
        
        # Pad to same length
        max_len = max(x.size(0) for x in xs)
        
        x_batch = []
        y_batch = []
        
        for x, y in zip(xs, ys):
            pad_len = max_len - x.size(0)
            if pad_len > 0:
                x = torch.cat([x, torch.full((pad_len,), self.pad_token_id, dtype=x.dtype)])
                y = torch.cat([y, torch.full((pad_len,), -1, dtype=y.dtype)])  # -1 for ignore_index
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
    ] * 10
    
    tokenizer = SimpleTokenizer()
    tokenizer.train(texts)
    
    dataset = TextDataset(texts, tokenizer, block_size=32, stride=16)
    print(f"Dataset size: {len(dataset)}")
    
    dataloader = create_dataloader(dataset, batch_size=2, shuffle=False)
    
    for i, (x, y) in enumerate(dataloader):
        print(f"Batch {i}: x shape={x.shape}, y shape={y.shape}")
        if i >= 2:
            break

