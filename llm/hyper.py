"""
mHC: manifold-constrained hyper-connections (DeepSeek, arXiv 2512.24880; used in DeepSeek-V4)

The residual stream is widened to n copies. Around every sublayer, per-token
maps read a mix of the copies into the sublayer (H_pre), write its output back
to each copy (H_post), and mix the copies with each other (H_res). Unlike the
original Hyper-Connections, H_res is projected onto doubly-stochastic matrices
with Sinkhorn-Knopp, so the gain across depth stays bounded.

Follows the paper's Eqs. 7-19 and DeepSeek-V4's inference code (hc_pre / hc_post /
hc_head). The bias initialization is inferred from V4's released weights, since
the paper doesn't give one: H_pre reads mostly from stream (block mod n) and
H_res starts near the identity.
"""

import torch
import torch.nn as nn

BIAS_INIT = 3.0  # logit magnitude: sigmoid(3) = 0.95, Sinkhorn diagonal ~0.993
ALPHA_INIT = 0.01  # scale on the input-dependent part (Table 5)


def sinkhorn(logits: torch.Tensor, iters: int, eps: float = 1e-6) -> torch.Tensor:
    """Project (..., n, n) logits to a doubly-stochastic matrix; ends on rows (exact forward gain)"""
    m = torch.softmax(logits, dim=-2) + eps
    m = m / (m.sum(dim=-1, keepdim=True) + eps)
    for _ in range(iters - 1):
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
    return m


class HyperConnection(nn.Module):
    """Wraps one sublayer: x (B, T, n, d) -> x' (B, T, n, d)"""

    def __init__(self, d: int, n: int, block: int, iters: int):
        super().__init__()
        self.n, self.iters = n, iters
        m = n * n + 2 * n
        self.phi = nn.Parameter(torch.zeros(n * d, m))  # input-dependent maps (Eq. 7)
        self.alpha = nn.Parameter(torch.full((3,), ALPHA_INIT))  # pre, post, res scales (Eq. 12)
        pre = torch.full((n,), -BIAS_INIT)
        pre[block % n] = BIAS_INIT
        post = torch.zeros(n)  # H_post = 2 * sigmoid(0) = 1 for every stream
        res = torch.full((n, n), -BIAS_INIT).fill_diagonal_(BIAS_INIT)
        self.bias = nn.Parameter(torch.cat((pre, post, res.flatten())))

    def coefficients(self, x):
        n = self.n
        with torch.autocast(x.device.type, enabled=False):
            xf = x.float().flatten(2)  # (B, T, n*d)
            inv_rms = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + 1e-6)
            mix = (xf @ self.phi.float()) * inv_rms  # RMSNorm folded into phi (Eq. 14-15)
            m_pre, m_post, m_res = mix.split([n, n, n * n], dim=-1)
            b_pre, b_post, b_res = self.bias.float().split([n, n, n * n])
            a_pre, a_post, a_res = self.alpha.float()
            h_pre = torch.sigmoid(a_pre * m_pre + b_pre) + 1e-6  # (B, T, n)
            h_post = 2 * torch.sigmoid(a_post * m_post + b_post)  # (B, T, n), each in (0, 2)
            h_res = sinkhorn((a_res * m_res + b_res).unflatten(-1, (n, n)), self.iters)  # (B, T, n, n)
        return h_pre.to(x.dtype), h_post.to(x.dtype), h_res.to(x.dtype)

    def forward(self, x, sublayer):
        h_pre, h_post, h_res = self.coefficients(x)
        u = torch.einsum("btn,btnd->btd", h_pre, x)
        y = sublayer(u)
        # Row i of H_res: how much of each input stream goes into output stream i
        return torch.einsum("btij,btjd->btid", h_res, x) + h_post[..., None] * y[:, :, None]


class HyperHead(nn.Module):
    """Collapses the n streams to one before the final norm (DeepSeek-V4 hc_head)"""

    def __init__(self, d: int, n: int, n_layers: int):
        super().__init__()
        self.phi = nn.Parameter(torch.zeros(n * d, n))
        self.alpha = nn.Parameter(torch.tensor(ALPHA_INIT))
        bias = torch.full((n,), -BIAS_INIT)
        bias[n_layers % n] = BIAS_INIT
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            xf = x.float().flatten(2)
            inv_rms = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + 1e-6)
            w = torch.sigmoid(self.alpha.float() * (xf @ self.phi.float()) * inv_rms + self.bias.float()) + 1e-6
        return torch.einsum("btn,btnd->btd", w.to(x.dtype), x)


def expand_streams(x: torch.Tensor, n: int) -> torch.Tensor:
    """(B, T, d) embedding -> (B, T, n, d) identical streams"""
    return x[:, :, None].expand(-1, -1, n, -1).contiguous()
