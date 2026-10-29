import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from llm.config import ModelConfig
from llm.data import TokenData
from llm.model import KVCache, Transformer
from llm.muon import MuonAdamW, newton_schulz
from llm.utils import lr_multiplier

ROOT = Path(__file__).resolve().parents[1]

TINY = dict(vocab_size=256, context_length=64, n_layers=2, d_model=64, n_heads=4, head_dim=16,
            n_kv_heads=2, ffn_dim=128)


def tiny_model(**overrides):
    torch.manual_seed(0)
    model = Transformer(ModelConfig(**{**TINY, **overrides}))
    # Zero-initialized output projections make blocks the identity; randomize so tests exercise them
    for name, p in model.named_parameters():
        if p.ndim >= 2:
            torch.nn.init.normal_(p, std=0.05)
    return model.eval()


@pytest.mark.parametrize("overrides", [{}, {"n_kv_heads": 4}, {"qk_norm": False, "attn_gate": False}])
def test_cached_forward_matches_full_forward(overrides):
    model = tiny_model(**overrides)
    idx = torch.randint(0, 256, (2, 20))
    full = model(idx)
    cache = KVCache(64)
    prefill = model(idx[:, :12], cache=cache)
    torch.testing.assert_close(prefill[:, -1], full[:, 11], rtol=1e-4, atol=1e-4)
    for t in range(12, 20):
        step = model(idx[:, t : t + 1], cache=cache)
        torch.testing.assert_close(step[:, -1], full[:, t], rtol=1e-4, atol=1e-4)


def test_greedy_generation_matches_uncached_argmax():
    model = tiny_model()
    idx = torch.randint(0, 256, (1, 5))
    out = model.generate(idx, max_new_tokens=15, temperature=0)
    seq = idx
    for _ in range(15):
        nxt = model(seq)[:, -1].argmax(-1, keepdim=True)
        seq = torch.cat((seq, nxt), dim=1)
    assert torch.equal(out, seq)


def test_generation_slides_past_context_length():
    model = tiny_model()
    out = model.generate(torch.randint(0, 256, (1, 10)), max_new_tokens=100, temperature=1.0)
    assert out.shape == (1, 110)


def test_chunked_loss_matches_full_loss():
    model = tiny_model().train()
    idx, tgt = torch.randint(0, 256, (2, 32)), torch.randint(0, 256, (2, 32))
    loss_full, _ = model(idx, tgt)
    grads_full = torch.autograd.grad(loss_full, list(model.parameters()))
    model.ce_chunk = 10
    loss_chunk, _ = model(idx, tgt)
    grads_chunk = torch.autograd.grad(loss_chunk, list(model.parameters()))
    torch.testing.assert_close(loss_chunk, loss_full)
    for a, b in zip(grads_chunk, grads_full):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_newton_schulz_orthogonalizes():
    g = torch.randn(32, 64)
    x = newton_schulz(g, steps=5)
    singular = torch.linalg.svdvals(x)
    # The quintic iteration lands singular values roughly in [0.7, 1.2] rather than exactly 1
    assert singular.min() > 0.5 and singular.max() < 1.3


def test_muon_overfits_a_batch():
    model = tiny_model().train()
    opt = MuonAdamW(model.optimizer_groups({"muon_lr": 0.02, "embed_lr": 3e-3, "adam_lr": 3e-3}))
    idx = torch.randint(0, 256, (4, 32))
    first = None
    for _ in range(60):
        loss, _ = model(idx[:, :-1], idx[:, 1:])
        loss.backward()
        opt.step()
        opt.zero_grad()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.3 * first


def test_token_data_covers_each_window_once_per_epoch(tmp_path):
    tokens = np.arange(1001, dtype=np.uint16)
    tokens.tofile(tmp_path / "train.bin")
    data = TokenData(tmp_path / "train.bin", block_size=10, seed=3)
    assert data.num_windows == 100
    starts = []
    for i in range(10):
        x, y = data.batch(i, 10)
        assert torch.equal(x[:, 1:], y[:, :-1])
        starts += x[:, 0].tolist()
    assert sorted(starts) == list(range(0, 1000, 10))
    # Same index, same batch: resume depends on this
    again = TokenData(tmp_path / "train.bin", block_size=10, seed=3)
    assert torch.equal(again.batch(7, 10)[0], data.batch(7, 10)[0])


def test_lr_schedules():
    cfg = {"warmup_steps": 10, "min_lr_ratio": 0.1}
    assert lr_multiplier(0, 100, {**cfg, "schedule": "wsd"}) == pytest.approx(0.1)
    assert lr_multiplier(50, 100, {**cfg, "schedule": "wsd"}) == 1.0
    assert lr_multiplier(99, 100, {**cfg, "schedule": "wsd"}) == pytest.approx(0.1 + 0.9 * 1 / 20)
    assert lr_multiplier(10, 100, {**cfg, "schedule": "cosine"}) == pytest.approx(1.0)
    assert lr_multiplier(100, 100, {**cfg, "schedule": "cosine"}) == pytest.approx(0.1)


def run_pretrain(config_path, data_dir, out_dir, *extra):
    cmd = [sys.executable, str(ROOT / "pretrain.py"), "--config", str(config_path), "--data", str(data_dir),
           "--out", str(out_dir), "--device", "cpu", *extra]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)


def test_resume_matches_uninterrupted_run(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 20_000), ("val", 2_000)):
        rng.integers(0, 256, n, dtype=np.uint16).tofile(tmp_path / f"{split}.bin")
    config = {
        "model": TINY,
        "train": {"block_size": 32, "micro_batch_size": 4, "tokens_per_step": 256, "total_tokens": 256 * 6,
                  "warmup_steps": 2, "eval_interval": 1000, "sample_interval": 0, "log_interval": 1},
    }
    cfg_path = tmp_path / "tiny.json"
    cfg_path.write_text(json.dumps(config))

    run_pretrain(cfg_path, tmp_path, tmp_path / "straight")
    # Same run, stopped after 3 steps and resumed
    run_pretrain(cfg_path, tmp_path, tmp_path / "resumed", "--stop_after_steps", "3")
    assert not (tmp_path / "resumed" / "final.pt").exists()
    run_pretrain(cfg_path, tmp_path, tmp_path / "resumed")

    a = torch.load(tmp_path / "straight" / "final.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed" / "final.pt", weights_only=False)
    assert a["step"] == b["step"] == 6
    for k in a["model"]:
        torch.testing.assert_close(a["model"][k], b["model"][k])


# --- DeepSeek components -------------------------------------------------------------

DEEPSEEK_VARIANTS = {
    "mla": dict(attention="mla", mla_kv_rank=32, mla_rope_dim=8),
    "mla_qrank": dict(attention="mla", mla_kv_rank=32, mla_rope_dim=8, mla_q_rank=48),
    "moe": dict(moe=True, moe_experts=4, moe_top_k=2, moe_expert_dim=32, moe_dense_layers=1),
    "mhc": dict(hc_streams=4, hc_sinkhorn_iters=10),
    "engram": dict(engram_layers=[1], engram_heads=2, engram_head_dim=8, engram_rows=97),
    "everything": dict(attention="mla", mla_kv_rank=32, mla_rope_dim=8, moe=True, moe_experts=4,
                       moe_top_k=2, moe_expert_dim=32, hc_streams=4, engram_layers=[0, 1],
                       engram_heads=2, engram_head_dim=8, engram_rows=97, mtp_depth=1, doc_masking=True),
}


@pytest.mark.parametrize("name", DEEPSEEK_VARIANTS)
def test_deepseek_variant_cache_matches_full_forward(name):
    model = tiny_model(**DEEPSEEK_VARIANTS[name])
    idx = torch.randint(0, 256, (2, 20))
    full = model(idx)
    cache = KVCache(64)
    model(idx[:, :12], cache=cache)
    for t in range(12, 20):
        step = model(idx[:, t : t + 1], cache=cache)
        torch.testing.assert_close(step[:, -1], full[:, t], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("name", DEEPSEEK_VARIANTS)
def test_deepseek_variant_trains(name):
    model = tiny_model(**DEEPSEEK_VARIANTS[name]).train()
    opt = MuonAdamW(model.optimizer_groups({"muon_mode": "deepseek", "adam_lr": 3e-3, "weight_decay": 0.1}))
    idx = torch.randint(0, 256, (4, 33))
    losses = []
    for _ in range(40):
        loss, metrics = model(idx[:, :-1], idx[:, 1:])
        loss.backward()
        opt.step()
        opt.zero_grad()
        model.after_optimizer_step()
        losses.append(metrics["lm_loss"].item())
    assert losses[-1] < 0.5 * losses[0]
    # Every parameter is in exactly one optimizer group
    grouped = [p for g in opt.param_groups for p in g["params"]]
    assert len(grouped) == len(set(map(id, grouped))) == len(list(model.parameters()))


def test_mhc_starts_close_to_a_plain_residual():
    from llm.hyper import HyperConnection, expand_streams

    hc = HyperConnection(d=16, n=4, block=0, iters=20)
    x = expand_streams(torch.randn(2, 5, 16), 4)
    h_pre, h_post, h_res = hc.coefficients(x)
    torch.testing.assert_close(h_res.sum(-1), torch.ones(2, 5, 4))  # rows sum to 1 exactly
    torch.testing.assert_close(h_res.sum(-2), torch.ones(2, 5, 4), atol=1e-3, rtol=0)
    assert (h_res.diagonal(dim1=-2, dim2=-1) > 0.99).all()
    torch.testing.assert_close(h_post, torch.ones(2, 5, 4))
    out = hc(x, lambda u: torch.zeros_like(u))
    torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)


def test_moe_bias_balances_load():
    from llm.moe import MoE

    moe = MoE(ModelConfig(**TINY, moe=True, moe_experts=4, moe_top_k=1, moe_expert_dim=16)).train()
    moe.reset_parameters()
    with torch.no_grad():
        moe.router[0] += 5.0  # expert 0 grabs every token
    x = torch.randn(4, 16, 64) + moe.router[0] / 5.0
    first = None
    for _ in range(300):
        moe(x, [])
        load = moe.load.clone()
        first = first if first is not None else load
        moe.update_bias(0.05)
    assert first[0] == first.sum()  # all tokens on one expert at the start
    assert load.max() < 0.6 * load.sum()  # spread out after bias updates


def test_engram_compression_merges_case_and_spacing():
    from llm.engram import compression_table
    from llm.tokenizer import get_tokenizer

    table, n_ids = compression_table(50304)
    enc = get_tokenizer()
    ids = [enc.encode_ordinary(w)[0] for w in (" Apple", "Apple", " apple", "apple")]
    assert len({table[i].item() for i in ids}) == 1
    assert 30_000 < n_ids < 36_000  # GPT-2's 50257 shrinks by about a third
    assert len(set(table[50257:].tolist())) == 50304 - 50257  # padding ids stay distinct


def test_hybrid_newton_schulz_is_tighter():
    g = torch.randn(64, 128)
    plain = torch.linalg.svdvals(newton_schulz(g, steps=5))
    hybrid = torch.linalg.svdvals(newton_schulz(g, steps=8, refine_steps=2))
    assert (hybrid - 1).abs().max() < (plain - 1).abs().max()
    assert (hybrid - 1).abs().max() < 0.05
