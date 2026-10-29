"""
Engram: conditional memory via hashed n-gram lookup (DeepSeek, arXiv 2601.07372)

Before selected blocks, each token's suffix n-grams (of normalized token ids)
are hashed into large embedding tables. The retrieved vectors are gated by the
current hidden state and added to the residual stream. The lookup costs O(1)
per token and depends only on token ids.

Follows the paper's Eqs. 1-4 and DeepSeek's reference code (engram_demo_v1.py):
multiplicative-XOR hashing, one prime-sized table per (order, head), a
signed-square-root gate, and one gate per residual stream under mHC (Eq. 6).
The short causal convolution is left out, as in DeepSeek-V4.1-Flash §2.4.2.
"""

import re
import unicodedata
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .tokenizer import EOT


def _normalize(text: str) -> str:
    """NFKC, strip accents, lowercase, collapse whitespace (the paper's App. C normalizer)"""
    t = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", text))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = re.sub(r"[ \t\r\n]+", " ", t)
    stripped = t.strip()
    return stripped if stripped else (" " if t else "")


@lru_cache(maxsize=4)
def compression_table(vocab_size: int) -> tuple[torch.Tensor, int]:
    """Map each token id to a normalized id, so " Apple", "apple", "APPLE" share hash keys.

    Returns (table of shape (vocab_size,), number of distinct ids). Vocabularies
    smaller than GPT-2's (test models) map to themselves.
    """
    if vocab_size <= EOT:
        return torch.arange(vocab_size), vocab_size
    from .tokenizer import get_tokenizer

    enc = get_tokenizer()
    keys = {}
    table = []
    for i in range(vocab_size):
        if i <= EOT:
            raw = enc.decode_single_token_bytes(i)
            text = raw.decode("utf-8", errors="replace")
            # Partial UTF-8 byte sequences keep their raw bytes as the key
            key = ("raw", raw) if "�" in text else ("text", _normalize(text))
        else:
            key = ("id", i)  # padding / chat-special ids stay distinct
        table.append(keys.setdefault(key, len(keys)))
    return torch.tensor(table), len(keys)


def _primes_from(start: int):
    n = max(2, start)
    while True:
        if n < 4 or all(n % p for p in range(2, int(n**0.5) + 1)):
            yield n
        n += 1


class PrimeAllocator:
    """Hands out distinct primes >= a target, so no two tables share a modulus"""

    def __init__(self):
        self.used = set()

    def take(self, target: int) -> int:
        for p in _primes_from(target):
            if p not in self.used:
                self.used.add(p)
                return p


class Engram(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int, primes: PrimeAllocator, n_branches: int):
        super().__init__()
        self.orders = list(cfg.engram_orders)
        self.max_order = max(self.orders)
        self.k = cfg.engram_heads
        d, dh = cfg.d_model, cfg.engram_head_dim
        self.d, self.n_branches = d, n_branches
        _, n_ids = compression_table(cfg.vocab_size)

        # Odd multipliers per layer, shared by every order and head; products can't overflow int64
        rng = np.random.default_rng(10007 * layer)
        r_max = ((2**63 - 1) // n_ids) // 2
        self.register_buffer("mults", torch.tensor(2 * rng.integers(0, r_max, self.max_order) + 1),
                             persistent=False)
        moduli = [primes.take(cfg.engram_rows) for _ in self.orders for _ in range(self.k)]
        offsets = np.concatenate(([0], np.cumsum(moduli)[:-1]))
        self.register_buffer("moduli", torch.tensor(moduli), persistent=False)
        self.register_buffer("offsets", torch.tensor(offsets), persistent=False)

        d_mem = len(self.orders) * self.k * dh
        self.table = nn.Embedding(int(sum(moduli)), dh)
        self.w_k = nn.Linear(d_mem, n_branches * d, bias=False)
        self.w_v = nn.Linear(d_mem, d, bias=False)
        self.h_norm = nn.Parameter(torch.ones(n_branches, d))
        self.k_norm = nn.Parameter(torch.ones(n_branches, d))

    def reset_parameters(self):
        nn.init.normal_(self.table.weight, std=1.0)
        nn.init.normal_(self.w_k.weight, std=0.02)
        nn.init.zeros_(self.w_v.weight)  # starts as the identity
        nn.init.ones_(self.h_norm)
        nn.init.ones_(self.k_norm)

    def hash_ids(self, ids: torch.Tensor, t: int) -> torch.Tensor:
        """ids: (B, max_order - 1 + T) compressed ids with left context -> (B, T, tables) row indices"""
        ctx = self.max_order - 1
        mixed = []
        for n in self.orders:
            mix = torch.zeros_like(ids[:, ctx:])
            for j in range(n):
                mix = mix ^ (ids[:, ctx - j : ctx - j + t] * self.mults[j])
            mixed.append(mix[..., None].expand(-1, -1, self.k))
        mix = torch.cat(mixed, dim=-1)  # (B, T, orders * k)
        return mix % self.moduli + self.offsets

    def forward(self, x, ids):
        """x: (B, T, d) or (B, T, n, d) residual stream; ids from left_context_ids()"""
        t = x.size(1)
        e = self.table(self.hash_ids(ids, t)).flatten(2)  # (B, T, d_mem)
        v = self.w_v(e)  # (B, T, d)
        k = self.w_k(e).unflatten(-1, (self.n_branches, self.d))  # (B, T, n, d)
        xs = x if x.ndim == 4 else x[:, :, None]
        h = F.rms_norm(xs, (self.d,)) * self.h_norm
        k = F.rms_norm(k, (self.d,)) * self.k_norm
        score = (h * k).sum(dim=-1, keepdim=True) / self.d**0.5
        score = score.sign() * score.abs().clamp(min=1e-6).sqrt()  # as in the reference code
        y = torch.sigmoid(score) * v[:, :, None]  # (B, T, n, d)
        return x + (y if x.ndim == 4 else y[:, :, 0])


def left_context_ids(idx: torch.Tensor, compress: torch.Tensor, history, ctx: int):
    """Compressed ids for idx (B, T) with ctx tokens of left context.

    history holds earlier token ids when decoding with a KV cache; positions
    before the sequence start use <|endoftext|>, which also starts every document.
    """
    if history is None:
        left = idx.new_full((idx.size(0), ctx), min(EOT, compress.numel() - 1))
    else:
        left = history[:, -ctx:]
        if left.size(1) < ctx:
            pad = idx.new_full((idx.size(0), ctx - left.size(1)), min(EOT, compress.numel() - 1))
            left = torch.cat((pad, left), dim=1)
    return compress[torch.cat((left, idx), dim=1)]
