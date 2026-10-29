"""
Attention: grouped-query (GQA) and DeepSeek's multi-head latent attention (MLA)

Both use rotary positions, optional QK-norm, an optional sigmoid output gate,
and an optional document mask so packed sequences don't attend across
<|endoftext|> boundaries.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


def rope_tables(dim: int, length: int, theta: float):
    inv_freq = 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    freqs = torch.outer(torch.arange(length, dtype=torch.float32), inv_freq)
    return freqs.cos(), freqs.sin()  # (length, dim / 2)


def apply_rope(x, cos, sin):
    """Rotate (..., T, D) by position; cos and sin are (T, D/2)"""
    x1, x2 = x.float().chunk(2, dim=-1)
    out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.type_as(x)


class KVCache:
    """Preallocated per-layer cache for generation. Cached tensors keep time on dim -2."""

    def __init__(self, max_len: int):
        self.max_len = max_len
        self.pos = 0
        self.layers = {}
        self.tokens = None  # recent token ids, for Engram's n-gram context

    def update(self, layer: int, *new: torch.Tensor):
        t = new[0].size(-2)
        if layer not in self.layers:
            self.layers[layer] = [x.new_empty(*x.shape[:-2], self.max_len, x.size(-1)) for x in new]
        bufs = self.layers[layer]
        for buf, x in zip(bufs, new):
            buf[..., self.pos : self.pos + t, :] = x
        return [buf[..., : self.pos + t, :] for buf in bufs]


@dataclass
class Context:
    """Per-forward state shared by every block"""

    rope: tuple  # (cos, sin) for the positions being processed
    cache: KVCache | None = None
    mask: torch.Tensor | None = None  # (B, 1, T, T) bool, True = may attend
    engram_ids: torch.Tensor | None = None  # compressed token ids with left context
    aux_losses: list = field(default_factory=list)


def document_mask(idx: torch.Tensor, eot: int) -> torch.Tensor:
    """Causal mask that also blocks attention across documents.

    Every document starts with <|endoftext|>, so a running count of EOT tokens
    gives each position its document id.
    """
    doc = (idx == eot).cumsum(dim=1)
    same_doc = doc[:, :, None] == doc[:, None, :]
    t = idx.size(1)
    causal = torch.ones(t, t, dtype=torch.bool, device=idx.device).tril()
    return (same_doc & causal)[:, None]


def attend(q, k, v, mask=None):
    """SDPA for (B, H, Tq, D) queries over (B, H, Tk, D) keys; causal unless decoding one token"""
    tq, tk = q.size(-2), k.size(-2)
    if mask is not None:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    if tq == 1:
        return F.scaled_dot_product_attention(q, k, v)
    if tq == tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    # Queries are the last tq positions of the cached sequence
    causal = torch.ones(tq, tk, dtype=torch.bool, device=q.device).tril(diagonal=tk - tq)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=causal)


def maybe_norm(enabled: bool, dim: int, eps: float):
    return RMSNorm(dim, eps) if enabled else nn.Identity()


class Attention(nn.Module):
    """Grouped-query attention"""

    def __init__(self, cfg: ModelConfig, layer: int):
        super().__init__()
        self.layer = layer
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        d = cfg.d_model
        self.q_proj = nn.Linear(d, self.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.hd, d, bias=False)
        self.gate = nn.Linear(d, self.n_heads * self.hd, bias=False) if cfg.attn_gate else None
        self.q_norm = maybe_norm(cfg.qk_norm, self.hd, cfg.norm_eps)
        self.k_norm = maybe_norm(cfg.qk_norm, self.hd, cfg.norm_eps)

    def forward(self, x, ctx: Context):
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, t, self.n_heads, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, self.n_kv, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, *ctx.rope), apply_rope(k, *ctx.rope)
        if ctx.cache is not None:
            k, v = ctx.cache.update(self.layer, k, v)
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = attend(q, k, v, ctx.mask).transpose(1, 2).reshape(b, t, -1)
        if self.gate is not None:
            y = y * torch.sigmoid(self.gate(x))
        return self.o_proj(y)


class MLAttention(nn.Module):
    """Multi-head latent attention (DeepSeek-V2 §2.1, V3 Eq. 1-11)

    Keys and values are up-projected from one small latent c_kv per token, and
    a separate RoPE key of mla_rope_dim dims is shared by all heads. The KV
    cache holds only the latent and the rope key: mla_kv_rank + mla_rope_dim
    numbers per token, versus 2 * n_kv_heads * head_dim for GQA.

    QK-norm here normalizes the content and rope parts separately, so the
    cached rope key never depends on the up-projected keys.
    """

    def __init__(self, cfg: ModelConfig, layer: int):
        super().__init__()
        self.layer = layer
        self.n_heads, self.hd = cfg.n_heads, cfg.head_dim
        self.rank, self.rd = cfg.mla_kv_rank, cfg.mla_rope_dim
        d, h, eps = cfg.d_model, cfg.n_heads, cfg.norm_eps
        q_out = h * (self.hd + self.rd)
        if cfg.mla_q_rank:
            self.q_down = nn.Linear(d, cfg.mla_q_rank, bias=False)
            self.q_down_norm = RMSNorm(cfg.mla_q_rank, eps)
            self.q_up = nn.Linear(cfg.mla_q_rank, q_out, bias=False)
        else:
            self.q_proj = nn.Linear(d, q_out, bias=False)
        self.kv_down = nn.Linear(d, self.rank + self.rd, bias=False)  # [c_kv | rope key]
        self.kv_norm = RMSNorm(self.rank, eps)
        self.kv_up = nn.Linear(self.rank, h * 2 * self.hd, bias=False)  # per head [key | value]
        self.o_proj = nn.Linear(h * self.hd, d, bias=False)
        self.gate = nn.Linear(d, h * self.hd, bias=False) if cfg.attn_gate else None
        self.q_norm = maybe_norm(cfg.qk_norm, self.hd, eps)
        self.q_rope_norm = maybe_norm(cfg.qk_norm, self.rd, eps)
        self.k_norm = maybe_norm(cfg.qk_norm, self.hd, eps)
        self.k_rope_norm = maybe_norm(cfg.qk_norm, self.rd, eps)

    def forward(self, x, ctx: Context):
        b, t, _ = x.shape
        h, hd, rd = self.n_heads, self.hd, self.rd
        q = self.q_up(self.q_down_norm(self.q_down(x))) if hasattr(self, "q_up") else self.q_proj(x)
        q_nope, q_rope = q.view(b, t, h, hd + rd).split([hd, rd], dim=-1)
        q_nope, q_rope = self.q_norm(q_nope), self.q_rope_norm(q_rope)

        c_kv, k_rope = self.kv_down(x).split([self.rank, rd], dim=-1)
        c_kv = self.kv_norm(c_kv)  # (B, T, rank)
        k_rope = apply_rope(self.k_rope_norm(k_rope)[:, None], *ctx.rope)  # (B, 1, T, rd)
        q_rope = apply_rope(q_rope.transpose(1, 2), *ctx.rope)  # (B, H, T, rd)
        if ctx.cache is not None:
            c_kv, k_rope = ctx.cache.update(self.layer, c_kv, k_rope)

        tk = c_kv.size(1)
        k_nope, v = self.kv_up(c_kv).view(b, tk, h, 2 * hd).split([hd, hd], dim=-1)
        k_nope = self.k_norm(k_nope)
        q = torch.cat((q_nope.transpose(1, 2), q_rope), dim=-1)
        k = torch.cat((k_nope.transpose(1, 2), k_rope.expand(b, h, tk, rd)), dim=-1)
        y = attend(q, k, v.transpose(1, 2), ctx.mask).transpose(1, 2).reshape(b, t, -1)
        if self.gate is not None:
            y = y * torch.sigmoid(self.gate(x))
        return self.o_proj(y)
