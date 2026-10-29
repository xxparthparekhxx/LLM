"""
DeepSeekMoE: shared experts plus fine-grained routed experts (DeepSeek-V3 §2.1.2, V4 §2.1)

Each token goes through the shared expert and its top-k routed experts. Routing
uses auxiliary-loss-free balancing: a per-expert bias, nudged after every
optimizer step toward equal load, shifts which experts get picked but never
the gate values, so balancing doesn't fight the language-modeling loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def swiglu(gate, up, clamp: float | None):
    if clamp is not None:
        # DeepSeek-V4 §4.2.3: cap the gate from above, the linear branch on both sides
        gate, up = gate.clamp(max=clamp), up.clamp(-clamp, clamp)
    return F.silu(gate) * up


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int, clamp: float | None = None):
        super().__init__()
        self.clamp = clamp
        self.gate_proj = nn.Linear(d, hidden, bias=False)
        self.up_proj = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x), self.clamp))


class MoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        e, h, d = cfg.moe_experts, cfg.moe_expert_dim, cfg.d_model
        self.n_experts, self.top_k = e, cfg.moe_top_k
        self.score_fn, self.route_scale = cfg.moe_score, cfg.moe_route_scale
        self.seq_aux, self.clamp = cfg.moe_seq_aux, cfg.swiglu_clamp
        self.router = nn.Parameter(torch.empty(e, d))
        # Expert weights stacked so Muon orthogonalizes each expert's matrix separately
        self.w_gate = nn.Parameter(torch.empty(e, h, d))
        self.w_up = nn.Parameter(torch.empty(e, h, d))
        self.w_down = nn.Parameter(torch.empty(e, d, h))
        self.shared = SwiGLU(d, h * cfg.moe_shared, cfg.swiglu_clamp) if cfg.moe_shared else None
        self.register_buffer("route_bias", torch.zeros(e))  # saved in checkpoints
        self.register_buffer("load", torch.zeros(e), persistent=False)  # tokens per expert this step

    def reset_parameters(self):
        for p in (self.router, self.w_gate, self.w_up):
            nn.init.normal_(p, std=0.02)
        nn.init.zeros_(self.w_down)

    def scores(self, x):
        logits = x.float() @ self.router.float().t()
        if self.score_fn == "sigmoid":
            return torch.sigmoid(logits)
        return F.softplus(logits).sqrt()

    def forward(self, x, aux_losses: list):
        b, t, d = x.shape
        xf = x.reshape(-1, d)
        s = self.scores(xf)  # (N, E)
        top = torch.topk(s + self.route_bias, self.top_k, dim=-1).indices  # bias only picks experts
        gates = s.gather(1, top)
        gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-20) * self.route_scale

        flat = top.flatten()
        counts = torch.bincount(flat, minlength=self.n_experts)
        if self.training:
            self.load += counts.to(self.load.dtype)
        order = flat.argsort()
        token = order // self.top_k
        weight = gates.flatten()[order].to(x.dtype)
        out = torch.zeros_like(xf)
        start = 0
        for e, n in enumerate(counts.tolist()):
            if n == 0:
                continue
            idx = token[start : start + n]
            h = swiglu(xf[idx] @ self.w_gate[e].t(), xf[idx] @ self.w_up[e].t(), self.clamp)
            out.index_add_(0, idx, (h @ self.w_down[e].t()) * weight[start : start + n, None])
            start += n
        if self.shared is not None:
            out = out + self.shared(xf)

        if self.training and self.seq_aux > 0:
            # Sequence-wise balance loss (V3 Eq. 17-20)
            p = (s / s.sum(dim=-1, keepdim=True)).view(b, t, -1).mean(dim=1)  # (B, E)
            hits = F.one_hot(top, self.n_experts).sum(dim=1).view(b, t, -1).sum(dim=1).float()
            f = hits * self.n_experts / (self.top_k * t)
            aux_losses.append(self.seq_aux * (f * p).sum(dim=-1).mean())
        return out.view(b, t, d)

    @torch.no_grad()
    def update_bias(self, rate: float):
        """Push overloaded experts' bias down and underloaded ones up (V3 Eq. 16 discussion)"""
        if self.load.sum() > 0:
            self.route_bias += rate * torch.sign(self.load.mean() - self.load)
        self.load.zero_()

    def expert_params(self) -> int:
        return self.w_gate[0].numel() + self.w_up[0].numel() + self.w_down[0].numel()
