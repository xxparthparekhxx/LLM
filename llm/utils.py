"""Precision, learning-rate schedule and checkpoint helpers shared by the scripts"""

import math
from contextlib import nullcontext
from pathlib import Path

import torch


def autocast_dtype(device: str):
    """bfloat16 on Ampere and newer GPUs; plain fp32 everywhere else.

    Pre-Ampere cards have no fast path for either half type: on a GTX 1650 Ti,
    fp16 matmuls measure 0.41 TFLOPS against 1.78 for fp32, and bf16 is emulated.
    """
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return None


def autocast(device: str):
    dtype = autocast_dtype(device)
    return torch.autocast("cuda", dtype=dtype) if dtype else nullcontext()


def lr_multiplier(step: int, max_steps: int, cfg: dict) -> float:
    """Warmup, then cosine or warmup-stable-decay (WSD) down to min_lr_ratio"""
    floor = cfg.get("min_lr_ratio", 0.1)
    warmup = cfg.get("warmup_steps", 0)
    if step < warmup:
        return (step + 1) / warmup
    if cfg.get("schedule", "wsd") == "wsd":
        # Flat at peak, then linear decay over the last decay_frac of training.
        # A WSD run can be extended mid-way: only the decay phase depends on max_steps.
        decay_start = int(max_steps * (1 - cfg.get("decay_frac", 0.2)))
        if step < decay_start:
            return 1.0
        frac = (step - decay_start) / max(1, max_steps - decay_start)
        return 1.0 + (floor - 1.0) * min(frac, 1.0)
    frac = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return floor + 0.5 * (1 - floor) * (1 + math.cos(math.pi * frac))


def load_model(path: str | Path, device: str = "cpu"):
    """Rebuild a model from a pretrain.py checkpoint (ckpt.pt or final.pt)"""
    from .config import ModelConfig
    from .model import Transformer

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Transformer(ModelConfig(**ckpt["config"]["model"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
