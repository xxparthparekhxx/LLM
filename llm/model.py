"""
Decoder-only transformer

The base model is a modern dense design: pre-norm RMSNorm, rotary positions,
grouped-query attention with QK-norm and a sigmoid output gate, SwiGLU,
tied embeddings and tanh logit soft-capping. No biases, no dropout.

DeepSeek components, each switched on in ModelConfig:
  attention="mla"     multi-head latent attention         (V2, V3)
  moe=True            shared + routed experts, loss-free balancing (V3, V4)
  hc_streams=4        manifold-constrained hyper-connections (mHC, V4)
  engram_layers=[1]   hashed n-gram memory                 (Engram, V4.1)
  mtp_depth=1         multi-token prediction               (V3)
  doc_masking=True    no attention across documents        (V4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import Attention, Context, KVCache, MLAttention, RMSNorm, document_mask, rope_tables
from .config import ModelConfig
from .engram import Engram, PrimeAllocator, compression_table, left_context_ids
from .hyper import HyperConnection, HyperHead, expand_streams
from .moe import MoE, SwiGLU
from .tokenizer import EOT


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int, primes: PrimeAllocator | None = None,
                 allow_extras: bool = True):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = (MLAttention if cfg.attention == "mla" else Attention)(cfg, layer)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        use_moe = allow_extras and cfg.moe and layer >= cfg.moe_dense_layers
        self.mlp = MoE(cfg) if use_moe else SwiGLU(cfg.d_model, cfg.ffn_dim, cfg.swiglu_clamp)
        n = cfg.hc_streams if allow_extras else 1
        self.hc_attn = HyperConnection(cfg.d_model, n, layer, cfg.hc_sinkhorn_iters) if n > 1 else None
        self.hc_mlp = HyperConnection(cfg.d_model, n, layer, cfg.hc_sinkhorn_iters) if n > 1 else None
        use_engram = allow_extras and layer in cfg.engram_layers
        self.engram = Engram(cfg, layer, primes, n) if use_engram else None

    def forward(self, x, ctx: Context):
        if self.engram is not None:
            x = self.engram(x, ctx.engram_ids)

        def attn(u):
            return self.attn(self.attn_norm(u), ctx)

        def mlp(u):
            u = self.mlp_norm(u)
            return self.mlp(u, ctx.aux_losses) if isinstance(self.mlp, MoE) else self.mlp(u)

        if self.hc_attn is None:
            x = x + attn(x)
            return x + mlp(x)
        x = self.hc_attn(x, attn)
        return self.hc_mlp(x, mlp)


class MTPModule(nn.Module):
    """Predicts token t+2 from the trunk's state at t and the embedding of token t+1 (V3 Eq. 21-24)"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.e_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.block = Block(cfg, cfg.n_layers, allow_extras=False)
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(self, h, next_emb, ctx: Context):
        x = self.proj(torch.cat((self.h_norm(h), self.e_norm(next_emb)), dim=-1))
        return self.norm(self.block(x, ctx))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        primes = PrimeAllocator()
        self.blocks = nn.ModuleList(Block(cfg, i, primes) for i in range(cfg.n_layers))
        self.hc_head = HyperHead(cfg.d_model, cfg.hc_streams, cfg.n_layers) if cfg.hc_streams > 1 else None
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.mtp = MTPModule(cfg) if cfg.mtp_depth else None

        cos, sin = rope_tables(cfg.rope_dim, cfg.context_length, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        if cfg.engram_layers:
            self.register_buffer("engram_compress", compression_table(cfg.vocab_size)[0], persistent=False)

        # Set by the trainer:
        self.ce_chunk: int | None = None  # tokens per loss chunk; keeps (tokens x vocab) logits small
        self.mtp_weight: float = 0.3  # lambda in V3 Eq. 25
        self.moe_bias_rate: float = cfg.moe_bias_rate
        self.reset_parameters()

    def reset_parameters(self):
        for name, p in self.named_parameters():
            if p.ndim < 2 or ".hc_" in name or name.startswith("hc_head"):
                continue  # norm gains at 1; mHC keeps its own init
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.zeros_(p)  # every block starts as the identity
            else:
                nn.init.normal_(p, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, (Engram, MoE)):
                m.reset_parameters()

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def active_params(self) -> int:
        """Parameters used per token: unrouted experts and unused Engram rows don't count"""
        n = self.get_num_params()
        for m in self.modules():
            if isinstance(m, MoE):
                n -= (m.n_experts - m.top_k) * m.expert_params()
            elif isinstance(m, Engram):
                n -= m.table.weight.numel() - m.moduli.numel() * m.table.weight.size(1)
        return n

    def logits(self, h):
        weight = self.embed.weight if self.head is None else self.head.weight
        logits = F.linear(h, weight).float()
        cap = self.config.logit_softcap
        return cap * torch.tanh(logits / cap) if cap else logits

    def forward(self, idx, targets=None, cache: KVCache | None = None):
        """Logits (B, T, V) without targets, or (loss, metrics) with them.

        With a cache, only the last position's logits are returned.
        """
        cfg = self.config
        t = idx.size(1)
        start = cache.pos if cache is not None else 0
        assert start + t <= cfg.context_length, "sequence longer than context_length"
        ctx = Context(rope=(self.rope_cos[start : start + t], self.rope_sin[start : start + t]), cache=cache)
        if cfg.doc_masking and cache is None:
            ctx.mask = document_mask(idx, EOT)
        if cfg.engram_layers:
            history = cache.tokens if cache is not None else None
            ctx.engram_ids = left_context_ids(idx, self.engram_compress, history,
                                              max(cfg.engram_orders) - 1)
            if cache is not None:
                keep = max(cfg.engram_orders) - 1
                cache.tokens = idx[:, -keep:] if history is None else torch.cat((history, idx), 1)[:, -keep:]

        x = self.embed(idx)
        if self.hc_head is not None:
            x = expand_streams(x, cfg.hc_streams)
        for block in self.blocks:
            x = block(x, ctx)
        if self.hc_head is not None:
            x = self.hc_head(x)
        if cache is not None:
            cache.pos += t
        h = self.norm(x)

        if targets is None:
            return self.logits(h[:, -1:] if cache is not None else h)

        lm_loss = self.lm_loss(h, targets)
        loss = lm_loss
        metrics = {"lm_loss": lm_loss.detach()}
        if self.mtp is not None and self.training:
            # Position t sees tokens <= t+1 (its own target) and predicts t+2
            mtp_h = self.mtp(x[:, :-1], self.embed(targets[:, :-1].clamp(min=0)),
                             Context(rope=(ctx.rope[0][:-1], ctx.rope[1][:-1]),
                                     mask=None if ctx.mask is None else ctx.mask[:, :, :-1, :-1]))
            mtp_loss = self.lm_loss(mtp_h, targets[:, 1:])
            loss = loss + self.mtp_weight * mtp_loss
            metrics["mtp_loss"] = mtp_loss.detach()
        if ctx.aux_losses:
            aux = torch.stack(ctx.aux_losses).sum()
            loss = loss + aux
            metrics["moe_aux"] = aux.detach()
        return loss, metrics

    def lm_loss(self, h, targets):
        h, targets = h.flatten(0, 1), targets.flatten()
        if not self.ce_chunk or h.size(0) <= self.ce_chunk:
            return F.cross_entropy(self.logits(h), targets, ignore_index=-1)
        total = h.new_zeros((), dtype=torch.float32)
        for hs, ts in zip(h.split(self.ce_chunk), targets.split(self.ce_chunk)):
            total = total + checkpoint(self._ce_sum, hs, ts, use_reentrant=False)
        return total / (targets != -1).sum().clamp(min=1)

    def _ce_sum(self, h, targets):
        return F.cross_entropy(self.logits(h), targets, ignore_index=-1, reduction="sum")

    @torch.no_grad()
    def after_optimizer_step(self):
        """Auxiliary-loss-free MoE balancing: one bias update per optimizer step"""
        for m in self.modules():
            if isinstance(m, MoE):
                m.update_bias(self.moe_bias_rate)

    def optimizer_groups(self, tcfg: dict) -> list[dict]:
        """Muon for hidden weight matrices; AdamW for embeddings, router, mHC maps, norms and gains.

        Learning rates and weight decay follow tcfg. With muon_mode "deepseek" the
        Muon update is rescaled to AdamW's RMS (V4 §2.4), so one learning rate
        (adam_lr) drives both optimizers.
        """
        groups = {k: [] for k in ("muon", "embed", "engram", "decay", "other")}
        for name, p in self.named_parameters():
            if name.startswith(("embed.", "head.")):
                groups["embed"].append(p)
            elif name.endswith("table.weight"):
                groups["engram"].append(p)
            elif name.endswith((".h_norm", ".k_norm")):
                groups["other"].append(p)  # Engram's per-stream norm gains
            elif ".hc_" in name or name.startswith("hc_head"):
                groups["decay" if name.endswith("phi") else "other"].append(p)
            elif p.ndim >= 2 and not name.endswith("router"):
                groups["muon"].append(p)
            elif name.endswith("router"):
                groups["decay"].append(p)
            else:
                groups["other"].append(p)

        mode = tcfg.get("muon_mode", "keller")
        wd = tcfg.get("weight_decay", 0.0)
        adam_lr = tcfg.get("adam_lr", 3e-3)
        embed_lr = tcfg.get("embed_lr", adam_lr)
        betas = tuple(tcfg.get("adam_betas", (0.9, 0.95)))
        muon = dict(use_muon=True, momentum=tcfg.get("muon_momentum", 0.95), weight_decay=wd)
        if mode == "deepseek":
            muon.update(lr=adam_lr, rms_scale=tcfg.get("muon_rms", 0.18), ns_mode="hybrid")
        else:
            muon.update(lr=tcfg.get("muon_lr", 0.02))
        adam = dict(use_muon=False, betas=betas, eps=tcfg.get("adam_eps", 1e-10))
        out = [
            dict(params=groups["muon"], **muon),
            dict(params=groups["embed"], lr=embed_lr, weight_decay=0.0, **adam),
            dict(params=groups["engram"], lr=tcfg.get("engram_lr", 5 * embed_lr), weight_decay=0.0, **adam),
            dict(params=groups["decay"], lr=adam_lr, weight_decay=wd, **adam),
            dict(params=groups["other"], lr=adam_lr, weight_decay=0.0, **adam),
        ]
        return [g for g in out if g["params"]]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int | None = 50,
                 top_p: float = 1.0, min_p: float = 0.0, repetition_penalty: float = 1.0,
                 stop_tokens: tuple[int, ...] = ()):
        """Sample up to max_new_tokens after idx (B, T). Returns (B, T + generated)."""
        ctx_len = self.config.context_length
        cache = KVCache(ctx_len)
        logits = self(idx[:, -(ctx_len - 1):], cache=cache)[:, -1]
        out = idx
        stop = torch.tensor(stop_tokens, device=idx.device) if stop_tokens else None
        for _ in range(max_new_tokens):
            nxt = sample_next(logits, out, temperature, top_k, top_p, min_p, repetition_penalty)
            if stop is not None and bool(torch.isin(nxt, stop).all()):
                break
            out = torch.cat((out, nxt), dim=1)
            if cache.pos >= ctx_len:
                # Context full: slide the window and re-prefill from the latest half
                cache = KVCache(ctx_len)
                logits = self(out[:, -(ctx_len // 2):], cache=cache)[:, -1]
            else:
                logits = self(nxt, cache=cache)[:, -1]
        return out


def sample_next(logits, previous, temperature=0.8, top_k=None, top_p=1.0, min_p=0.0, repetition_penalty=1.0):
    """Next token per row from (B, V) logits; temperature 0 is greedy"""
    logits = logits.float().clone()
    if repetition_penalty != 1.0:
        # CTRL-style: push already-seen tokens down, whichever side of zero they are on
        seen = torch.gather(logits, 1, previous)
        logits.scatter_(1, previous, torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty))
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if min_p > 0:
        # min-p: drop tokens below min_p times the top token's probability
        probs = probs.masked_fill(probs < min_p * probs.amax(dim=-1, keepdim=True), 0.0)
    if top_p < 1.0:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        drop = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
        probs = probs.scatter(1, order, sorted_probs.masked_fill(drop, 0.0))
    return torch.multinomial(probs / probs.sum(dim=-1, keepdim=True), num_samples=1)
