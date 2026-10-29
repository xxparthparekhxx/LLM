"""Fixed-length training windows from the flat uint16 token files written by prepare_data.py"""

from pathlib import Path

import numpy as np
import torch


class TokenData:
    """The file is cut into non-overlapping windows of block_size + 1 tokens and
    shuffled once per epoch with a seeded permutation, so every token is used
    once per epoch and batch i is identical on every run (exact resume).
    """

    def __init__(self, path: Path, block_size: int, seed: int = 0):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.seed = seed
        self.num_windows = (len(self.data) - 1) // block_size
        if self.num_windows < 1:
            raise ValueError(f"{path} has only {len(self.data)} tokens, need more than block_size={block_size}")
        self._perms = {}

    def __len__(self):
        return len(self.data)

    def _window(self, w: int) -> np.ndarray:
        start = w * self.block_size
        return self.data[start : start + self.block_size + 1]

    def _perm(self, epoch: int) -> np.ndarray:
        if epoch not in self._perms:
            if len(self._perms) > 1:  # keep at most this epoch and the next
                self._perms.pop(min(self._perms))
            self._perms[epoch] = np.random.default_rng(self.seed + epoch).permutation(self.num_windows)
        return self._perms[epoch]

    def batch(self, index: int, batch_size: int):
        """The index-th batch of the shuffled stream, as (x, y) int64 tensors"""
        n = self.num_windows
        first = index * batch_size
        windows = [self._window(self._perm(i // n)[i % n]) for i in range(first, first + batch_size)]
        chunk = torch.from_numpy(np.stack(windows).astype(np.int64))
        return chunk[:, :-1], chunk[:, 1:]

    def eval_batches(self, batch_size: int, max_tokens: int):
        """Fixed, unshuffled windows from the start of the file, for comparable val loss"""
        n = min(self.num_windows, max(1, max_tokens // self.block_size))
        for first in range(0, n, batch_size):
            ws = range(first, min(first + batch_size, n))
            chunk = torch.from_numpy(np.stack([self._window(w) for w in ws]).astype(np.int64))
            yield chunk[:, :-1], chunk[:, 1:]
