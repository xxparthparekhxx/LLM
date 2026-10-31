"""Measure ONNX against PyTorch and against the other exported variants

    python measure_parity.py --variants model/model.onnx model/model_fp16.onnx model/model_int8.onnx

For each variant it reports, over real TinyStories validation tokens:
  - max absolute logit difference against torch fp32 (same weights, same inputs)
  - next-token argmax agreement
  - validation loss on the same ~50k tokens torch fp32 is scored on

It also checks that cached step-by-step decoding produces the same logits as one
full forward pass, and that greedy decoding produces the same tokens as
Transformer.generate.

The PyTorch reference is computed window by window, so only one window of logits
is in memory at a time (1 x 512 x 50304 floats each).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from llm.tokenizer import EOT, get_tokenizer
from llm.utils import load_model

ROOT = Path(__file__).resolve().parent
BLOCK = 512


def feed(session, ids, position_ids, past_keys, past_values, mask):
    return session.run(None, {
        "input_ids": ids,
        "position_ids": position_ids,
        "past_keys": past_keys,
        "past_values": past_values,
        "attention_mask": mask,
    })


def cross_entropy_sum(logits, targets):
    """Summed NLL for (T, V) logits and (T,) int targets, in float64"""
    x = logits.astype(np.float64)
    x -= x.max(axis=-1, keepdims=True)
    logz = np.log(np.exp(x).sum(axis=-1))
    return float((logz - x[np.arange(len(targets)), targets]).sum())


def greedy_onnx(session, ids, steps, cfg):
    """Greedy decode through the export's cache; returns the generated ids and tokens/s"""
    n_layers, n_kv, hd = cfg.n_layers, cfg.n_kv_heads, cfg.head_dim
    past_k = np.zeros((n_layers, 1, n_kv, 0, hd), dtype=np.float32)
    past_v = np.zeros_like(past_k)
    t = len(ids)
    logits, _, past_k, past_v = feed(
        session,
        np.array([ids], dtype=np.int64),
        np.arange(t, dtype=np.int64)[None],
        past_k,
        past_v,
        np.tril(np.ones((t, t), dtype=bool)),
    )
    out = []
    start = time.perf_counter()
    for step in range(steps):
        nxt = int(np.argmax(logits[0, -1]))
        if nxt == EOT:
            break
        out.append(nxt)
        logits, _, past_k, past_v = feed(
            session,
            np.array([[nxt]], dtype=np.int64),
            np.array([[t + step]], dtype=np.int64),
            past_k,
            past_v,
            np.ones((1, t + step + 1), dtype=bool),
        )
    elapsed = time.perf_counter() - start
    return out, len(out) / elapsed


def check_cache(session, ids, cfg):
    """Prefill, then one token per step; compare with a full forward at every position"""
    n_layers, n_kv, hd = cfg.n_layers, cfg.n_kv_heads, cfg.head_dim
    t = len(ids)
    full, _, _, _ = feed(
        session,
        np.array([ids], dtype=np.int64),
        np.arange(t, dtype=np.int64)[None],
        np.zeros((n_layers, 1, n_kv, 0, hd), dtype=np.float32),
        np.zeros((n_layers, 1, n_kv, 0, hd), dtype=np.float32),
        np.tril(np.ones((t, t), dtype=bool)),
    )

    prefill = 40
    logits, _, past_k, past_v = feed(
        session,
        np.array([ids[:prefill]], dtype=np.int64),
        np.arange(prefill, dtype=np.int64)[None],
        np.zeros((n_layers, 1, n_kv, 0, hd), dtype=np.float32),
        np.zeros((n_layers, 1, n_kv, 0, hd), dtype=np.float32),
        np.tril(np.ones((prefill, prefill), dtype=bool)),
    )
    max_diff = float(np.abs(logits[0] - full[0, :prefill]).max())
    agree = int((logits[0].argmax(-1) == full[0, :prefill].argmax(-1)).sum())
    total = prefill
    for pos in range(prefill, t):
        logits, _, past_k, past_v = feed(
            session,
            np.array([[ids[pos]]], dtype=np.int64),
            np.array([[pos]], dtype=np.int64),
            past_k,
            past_v,
            np.ones((1, pos + 1), dtype=bool),
        )
        max_diff = max(max_diff, float(np.abs(logits[0, 0] - full[0, pos]).max()))
        agree += int(logits[0, 0].argmax() == full[0, pos].argmax())
        total += 1
    return max_diff, agree, total


def make_session(path, threads):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(ROOT / "runs/long-tinystories-33m/muon-deepseek/final.pt"))
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--val", default=str(ROOT / "data/tinystories/val.bin"))
    parser.add_argument("--tokens", type=int, default=50000, help="validation tokens to score")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default=str(ROOT / "web/model/parity.json"))
    parser.add_argument("--limit-windows", type=int, help="quick run: only this many 512-token windows")
    args = parser.parse_args()
    if args.variants is None:
        args.variants = [str(ROOT / "web/model" / name) for name in ("model.onnx", "model_fp16.onnx", "model_int8.onnx")]

    model, ckpt = load_model(args.checkpoint, "cpu")
    cfg = model.config
    enc = get_tokenizer()

    tokens = np.fromfile(args.val, dtype=np.uint16).astype(np.int64)
    n_windows = min((min(len(tokens) - 1, args.tokens) // BLOCK), args.limit_windows or 10**9)
    print(f"PyTorch fp32 reference on {n_windows} windows x {BLOCK} tokens ({n_windows * BLOCK} tokens)")

    variants = []
    for path in args.variants:
        p = Path(path)
        if not p.exists():
            print(f"  {path}: missing, skipped")
            continue
        try:
            session = make_session(p, args.threads)
            variants.append({"path": path, "name": p.stem.replace("model_", "").replace("model", "fp32"),
                             "session": session, "size": p.stat().st_size,
                             "max_diff": 0.0, "agree": 0, "n": 0, "loss": 0.0, "error": None})
            print(f"  {path}: loaded ({p.stat().st_size / 1e6:.1f} MB)")
        except Exception as err:  # fp16 kernels may be missing on this CPU
            print(f"  {path}: could not load: {err}")
            variants.append({"path": path, "name": p.stem.replace("model_", "").replace("model", "fp32"),
                             "session": None, "size": p.stat().st_size, "error": f"load: {err}"})

    torch_loss = 0.0
    torch_n = 0
    t0 = time.perf_counter()
    for w in range(n_windows):
        ids = tokens[w * BLOCK : (w + 1) * BLOCK]
        targets = tokens[w * BLOCK + 1 : (w + 1) * BLOCK + 1]

        with torch.no_grad():
            ref_logits = model(torch.from_numpy(ids)[None])[0].numpy()
        torch_loss += cross_entropy_sum(ref_logits, targets)
        torch_n += len(targets)

        for v in variants:
            if v["session"] is None or v["error"] is not None:
                continue
            try:
                logits = feed(
                    v["session"],
                    ids[None],
                    np.arange(BLOCK, dtype=np.int64)[None],
                    np.zeros((cfg.n_layers, 1, cfg.n_kv_heads, 0, cfg.head_dim), dtype=np.float32),
                    np.zeros((cfg.n_layers, 1, cfg.n_kv_heads, 0, cfg.head_dim), dtype=np.float32),
                    np.tril(np.ones((BLOCK, BLOCK), dtype=bool)),
                )[0][0]
                v["max_diff"] = max(v["max_diff"], float(np.abs(logits - ref_logits).max()))
                v["agree"] += int((logits.argmax(-1) == ref_logits.argmax(-1)).sum())
                v["n"] += BLOCK
                v["loss"] += cross_entropy_sum(logits, targets)
            except Exception as err:
                v["error"] = f"run: {err}"
                print(f"  {v['name']}: failed during inference: {err}")
        if (w + 1) % 10 == 0 or w + 1 == n_windows:
            print(f"  window {w + 1}/{n_windows}  ({time.perf_counter() - t0:.1f}s)", flush=True)

    torch_loss /= torch_n
    print(f"\nPyTorch fp32 validation loss: {torch_loss:.4f}\n")

    results = []
    for v in variants:
        row = {
            "variant": v["name"],
            "size_mb": round(v["size"] / 1e6, 1),
            "max_logit_diff": None,
            "argmax_agreement": None,
            "val_loss": None,
            "error": v["error"],
        }
        if v["session"] is not None and v["error"] is None:
            row["max_logit_diff"] = round(v["max_diff"], 6)
            row["argmax_agreement"] = round(v["agree"] / v["n"], 6)
            row["val_loss"] = round(v["loss"] / torch_n, 4)
        results.append(row)
        if row["error"]:
            print(f"{row['variant']:6s}  {row['size_mb']:7.1f} MB  FAILED: {row['error']}")
        else:
            print(f"{row['variant']:6s}  {row['size_mb']:7.1f} MB  "
                  f"max |dlogit| {row['max_logit_diff']:.5f}  argmax {row['argmax_agreement'] * 100:.2f}%  "
                  f"val loss {row['val_loss']:.4f} ({row['val_loss'] - torch_loss:+.4f})")

    # Cache, generation and tokenizer checks on the fp32 graph.
    fp32 = next((v for v in variants if v["session"] is not None and v["name"] == "fp32"), None)
    extra = {}
    if fp32 is not None:
        ids = tokens[:200].tolist()
        diff, agree, total = check_cache(fp32["session"], ids, cfg)
        extra["cache_vs_full"] = {"max_logit_diff": round(diff, 6), "argmax_agreement": round(agree / total, 6), "positions": total}
        print(f"\nCached steps vs full forward: max |dlogit| {diff:.5f}, argmax {agree / total * 100:.2f}% over {total} positions")

        prompt = "Once upon a time"
        prompt_ids = [EOT] + enc.encode_ordinary(prompt)
        onnx_ids, rate = greedy_onnx(fp32["session"], prompt_ids, 30, cfg)
        with torch.no_grad():
            torch_out = model.generate(torch.tensor([prompt_ids]), 30, temperature=0, stop_tokens=(EOT,))[0].tolist()[len(prompt_ids):]
        same = onnx_ids == torch_out
        extra["greedy_generation"] = {"tokens": len(onnx_ids), "match_torch": same, "tokens_per_second_ort_cpu": round(rate, 2)}
        print(f"Greedy 30 tokens through the cache: ONNX matches Transformer.generate: {same} ({rate:.1f} tok/s, ORT CPU)")
        if not same:
            print("  ONNX:", enc.decode(onnx_ids))
            print("  torch:", enc.decode(torch_out))

    report = {
        "checkpoint": str(args.checkpoint),
        "step": ckpt.get("step"),
        "tokens_seen": ckpt.get("tokens_seen"),
        "val_tokens_scored": int(torch_n),
        "torch_val_loss": round(torch_loss, 4),
        "variants": results,
        **extra,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
