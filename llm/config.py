"""Model hyperparameters. Every field is plain JSON so configs/*.json map onto it directly.

The defaults are a dense GPT-2-small-sized model. The DeepSeek components are
all off by default; each one is switched on by its own fields below.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 50304  # GPT-2's 50257, padded; see llm/tokenizer.py
    context_length: int = 1024
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    head_dim: int = 64

    # --- Attention ---
    attention: str = "gqa"  # "gqa" or "mla"
    n_kv_heads: int = 4  # GQA key/value heads; = n_heads for plain multi-head
    qk_norm: bool = True  # RMSNorm on queries and keys (Qwen3, OLMo 2, Gemma 3, DeepSeek-V4)
    attn_gate: bool = True  # sigmoid gate on the attention output (Gated Attention, Qwen 2025)
    rope_theta: float = 10000.0
    doc_masking: bool = False  # attend only within the current document (DeepSeek-V4 §4.1)

    # MLA: multi-head latent attention (DeepSeek-V2 §2.1, V3 §2.1.1)
    mla_kv_rank: int = 256  # d_c: width of the compressed key/value latent
    mla_rope_dim: int = 32  # d_h^R: decoupled RoPE dims, one rope key shared by all heads
    mla_q_rank: int | None = None  # d'_c: query compression; None projects queries directly

    # --- Feed-forward ---
    ffn_dim: int = 2048  # SwiGLU hidden size (~8/3 of d_model)
    swiglu_clamp: float | None = 10.0  # clamp gate and up branches (DeepSeek-V4 §4.2.3)

    # DeepSeekMoE: shared + fine-grained routed experts (V3 §2.1.2, V4 §2.1)
    moe: bool = False
    moe_experts: int = 12  # routed experts
    moe_top_k: int = 3
    moe_shared: int = 1  # shared experts, fused into one SwiGLU
    moe_expert_dim: int = 384
    moe_dense_layers: int = 1  # the first N blocks keep a dense FFN
    moe_score: str = "sqrt_softplus"  # V4's affinity; "sigmoid" is V3's
    moe_route_scale: float = 1.0
    moe_bias_rate: float = 1e-3  # gamma for auxiliary-loss-free balancing (V3 §2.1.2)
    moe_seq_aux: float = 0.0  # alpha for the sequence-wise balance loss (V3 uses 1e-4)

    # mHC: manifold-constrained hyper-connections (arXiv 2512.24880, DeepSeek-V4 §2.2)
    hc_streams: int = 1  # residual streams; 1 is a plain residual, V4 uses 4
    hc_sinkhorn_iters: int = 20

    # Engram: hashed n-gram memory (arXiv 2601.07372, DeepSeek-V4.1-Flash §2.4)
    engram_layers: list[int] = field(default_factory=list)  # blocks that get a module before them
    engram_orders: list[int] = field(default_factory=lambda: [2, 3])
    engram_heads: int = 4  # hash tables per n-gram order
    engram_head_dim: int = 32
    engram_rows: int = 65536  # rows per table, rounded up to distinct primes

    # MTP: multi-token prediction (DeepSeek-V3 §2.2); 0 disables, 1 is V3's depth
    mtp_depth: int = 0

    # --- Output ---
    tie_embeddings: bool = True
    logit_softcap: float | None = 30.0  # tanh cap on logits (Gemma 2); None to disable
    norm_eps: float = 1e-6

    def __post_init__(self):
        assert self.attention in ("gqa", "mla"), f"unknown attention {self.attention!r}"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        assert self.head_dim % 2 == 0 and self.mla_rope_dim % 2 == 0, "RoPE needs even dims"
        assert self.moe_score in ("sigmoid", "sqrt_softplus")
        assert self.mtp_depth in (0, 1), "only MTP depth 1 is implemented"
        assert all(0 <= i < self.n_layers for i in self.engram_layers)
        assert all(n >= 2 for n in self.engram_orders)

    @property
    def rope_dim(self) -> int:
        return self.mla_rope_dim if self.attention == "mla" else self.head_dim
