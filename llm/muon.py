"""
Muon for hidden weight matrices, AdamW for everything else, in one optimizer

Muon (Keller Jordan, 2024) replaces each 2D weight's momentum update with its
nearest orthogonal matrix, computed by a few Newton-Schulz iterations. It trains
transformers markedly faster than AdamW per token and holds one state buffer
per weight instead of two. Embeddings, the output head, norm gains and other
non-matrix parameters still go through AdamW.

Two modes per param group:
  default   Keller Jordan's: 5 quintic Newton-Schulz steps, update scaled by
            sqrt(max(1, rows/cols)), its own learning rate (~0.02)
  deepseek  DeepSeek-V4 §2.4: 8 quintic + 2 refining Newton-Schulz steps
            (Eq. 28), update rescaled to RMS 0.18 so it shares AdamW's
            learning rate; set ns_mode="hybrid" and rms_scale=0.18

Based on https://github.com/KellerJordan/Muon and the DeepSeek-V4 report.
"""

import torch

# Quintic Newton-Schulz coefficients from the reference implementation
NS_COEFFS = (3.4445, -4.7750, 2.0315)
# DeepSeek-V4's final refining steps: converge singular values to exactly 1
NS_REFINE = (2.0, -1.5, 0.5)


def newton_schulz(g: torch.Tensor, steps: int = 5, refine_steps: int = 0) -> torch.Tensor:
    """Approximately orthogonalize g (the U V^T of its SVD)"""
    # bf16 is fast on Ampere and newer; older GPUs only emulate it, so stay in fp32 there
    fast_bf16 = g.is_cuda and torch.cuda.get_device_capability(g.device)[0] >= 8
    x = g.bfloat16() if fast_bf16 else g.float()
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)  # spectral norm <= 1
    for i in range(steps + refine_steps):
        a, b, c = NS_COEFFS if i < steps else NS_REFINE
        s = x @ x.mT
        x = a * x + (b * s + c * s @ s) @ x
    if tall:
        x = x.mT
    return x


class MuonAdamW(torch.optim.Optimizer):
    """Each param group sets use_muon=True (2D hidden weights) or False (AdamW)"""

    def __init__(self, param_groups):
        for group in param_groups:
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("nesterov", True)
                group.setdefault("ns_mode", "quintic")  # or "hybrid" (DeepSeek-V4)
                group.setdefault("rms_scale", None)  # e.g. 0.18 (DeepSeek-V4)
                group.setdefault("weight_decay", 0.0)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0.0)
            # The LR schedule scales this per step
            group.setdefault("base_lr", group["lr"])
        super().__init__(param_groups, {})

    def set_lr_multiplier(self, mult: float):
        for group in self.param_groups:
            group["lr"] = group["base_lr"] * mult

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)

    def _muon_step(self, group):
        beta = group["momentum"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["momentum"] = torch.zeros_like(p)
            buf = state["momentum"]
            grad = p.grad
            buf.lerp_(grad, 1 - beta)
            update = grad.lerp(buf, beta) if group["nesterov"] else buf
            # 3D params (stacked MoE experts) are a batch of matrices; anything else is flattened to 2D
            if update.ndim != 3:
                update = update.reshape(update.size(0), -1)
            if group["ns_mode"] == "hybrid":
                update = newton_schulz(update, steps=8, refine_steps=2)
            else:
                update = newton_schulz(update, steps=5)
            rows, cols = update.size(-2), update.size(-1)
            if group["rms_scale"] is not None:
                # An orthogonal matrix has RMS 1/sqrt(max(rows, cols)); rescale to rms_scale
                update = update * (max(rows, cols) ** 0.5 * group["rms_scale"])
            else:
                # Same RMS for tall and wide matrices
                update = update * max(1.0, rows / cols) ** 0.5
            p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(update.view_as(p).to(p.dtype), alpha=-group["lr"])

    def _adamw_step(self, group):
        b1, b2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            t = state["step"]
            m, v = state["exp_avg"], state["exp_avg_sq"]
            m.lerp_(p.grad, 1 - b1)
            v.lerp_(p.grad.square(), 1 - b2)
            denom = (v / (1 - b2**t)).sqrt_().add_(group["eps"])
            p.mul_(1 - group["lr"] * group["weight_decay"])
            p.addcdiv_(m, denom, value=-group["lr"] / (1 - b1**t))
