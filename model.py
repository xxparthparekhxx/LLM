"""
Modern GPT-style Language Model Implementation
Features:
- Flash Attention support
- Rotary Position Embeddings (RoPE)
- RMSNorm for stability
- SwiGLU activation
- Weight initialization best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import warnings


@dataclass
class ModelConfig:
    """Configuration for the language model"""
    vocab_size: int = 50257
    context_length: int = 2048
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    head_dim: Optional[int] = None  # If None, computed as n_embd // n_heads
    dropout: float = 0.1
    bias: bool = False  # Modern models don't use bias
    use_rope: bool = True  # Rotary Position Embeddings
    use_rmsnorm: bool = True  # RMSNorm instead of LayerNorm
    use_swiglu: bool = True  # SwiGLU activation
    flash_attention: bool = True  # Use flash attention if available
    tie_weights: bool = True  # Tie input and output embeddings


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * (x / (norm + self.eps))


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        
        # Precompute cos and sin for efficiency
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]
        if hasattr(self, 'cos_cached'):
            self.cos_cached = cos_cached
            self.sin_cached = sin_cached
        else:
            self.register_buffer('cos_cached', cos_cached)
            self.register_buffer('sin_cached', sin_cached)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[-2]
        
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        return (
            self.cos_cached[:, :, :seq_len, :].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, :].to(dtype=x.dtype)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.gate_proj(x)) * self.up_proj(x)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional flash attention"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim or (config.n_embd // config.n_heads)
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_flash = config.flash_attention and hasattr(F, 'scaled_dot_product_attention')
        self.use_rope = config.use_rope
        
        # QKV projection
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Rotary embeddings
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(self.head_dim, config.context_length)
        
        # Causal mask for non-flash attention
        if not self.use_flash:
            self.register_buffer(
                'causal_mask',
                torch.triu(torch.ones(config.context_length, config.context_length), diagonal=1)
                .bool()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Compute QKV
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        
        # Apply rotary embeddings
        if self.use_rope:
            cos, sin = self.rotary_emb(q, seq_len=T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention
        if self.use_flash:
            # Use PyTorch's optimized attention
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,  # Causal mask is handled internally
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual attention with causal mask
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.causal_mask[:T, :T], float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        
        return y


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU or GELU"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.use_swiglu = config.use_swiglu
        
        if config.use_swiglu:
            # SwiGLU: gate and up projections
            self.gate_proj = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.up_proj = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.down_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        else:
            # Standard GELU
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
        else:
            return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.use_rmsnorm = config.use_rmsnorm
        
        if config.use_rmsnorm:
            self.ln_1 = RMSNorm(config.n_embd)
            self.ln_2 = RMSNorm(config.n_embd)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd)
            self.ln_2 = nn.LayerNorm(config.n_embd)
        
        self.attn = CausalSelfAttention(config)
        self.mlp = FeedForward(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class LanguageModel(nn.Module):
    """Modern GPT-style Language Model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
        if not config.use_rope:
            # Only use position embeddings if not using RoPE
            self.position_embedding = nn.Embedding(config.context_length, config.n_embd)
        else:
            self.position_embedding = None
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        # Final layer norm
        if config.use_rmsnorm:
            self.ln_f = RMSNorm(config.n_embd)
        else:
            self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special initialization for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))
        
        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Initialized model with {n_params / 1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            idx: (B, T) token indices
            targets: (B, T) target token indices for training
        
        Returns:
            logits: (B, T, vocab_size) or (B, vocab_size) if targets is None
            loss: scalar loss if targets is provided, None otherwise
        """
        B, T = idx.shape
        assert T <= self.config.context_length, f"Sequence length {T} exceeds context length {self.config.context_length}"
        
        # Token embeddings
        tok_emb = self.token_embedding(idx)
        
        # Position embeddings (if not using RoPE)
        if self.position_embedding is not None:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.position_embedding(pos)
            x = self.dropout(tok_emb + pos_emb)
        else:
            x = self.dropout(tok_emb)
        
        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        
        # Compute logits
        if targets is not None:
            # Training: compute logits for all positions
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference: only compute logits for last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        stop_token: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate new tokens
        
        Args:
            idx: (B, T) starting token indices
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: if set, only sample from top k tokens
            top_p: nucleus sampling threshold
            stop_token: token id to stop generation at
        
        Returns:
            Generated token indices (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to context length
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Stop if stop token is generated
            if stop_token is not None and idx_next.item() == stop_token:
                break
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def get_num_params(self) -> int:
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test the model
    config = ModelConfig(
        vocab_size=50257,
        context_length=512,
        n_layers=6,
        n_heads=8,
        n_embd=512,
        dropout=0.1
    )
    
    model = LanguageModel(config)
    
    # Test forward pass
    batch_size = 2
    seq_length = 128
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    dummy_targets = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    logits, loss = model(dummy_input, dummy_targets)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    model.eval()
    start_tokens = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(start_tokens, max_new_tokens=20, temperature=0.8, top_k=40)
    print(f"\nGenerated sequence shape: {generated.shape}")

