"""
Token-budget pretraining on pre-tokenized data

You set how many tokens to train on; the learning rate warms up, holds and
decays over exactly that many. Reads train.bin / val.bin from prepare_data.py.

- Resumes automatically: rerun the same command and it continues from <out>/ckpt.pt
  at the exact step and data position (safe across crashes and power cuts)
- Checkpoints every few minutes and on SIGINT / SIGTERM
- Logs to <out>/log.jsonl and <out>/samples.txt

    python pretrain.py --config configs/tinystories-30m.json --data data/tinystories
    python pretrain.py --config configs/tinystories-30m.json --data data/tinystories --benchmark 10
"""

import argparse
import json
import os
import signal
import time
from pathlib import Path

# Less fragmentation on small GPUs; must be set before CUDA initializes
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from llm.config import ModelConfig
from llm.data import TokenData
from llm.model import Transformer
from llm.muon import MuonAdamW
from llm.tokenizer import EOT, get_tokenizer
from llm.utils import autocast, autocast_dtype, lr_multiplier


def flops_per_token(model: Transformer, block_size: int) -> float:
    """Training FLOPs per token: 6 x params touched per token, plus attention scores"""
    c = model.config
    n = model.active_params() if hasattr(model, "active_params") else model.get_num_params()
    return 6 * n + 12 * c.n_layers * c.n_heads * c.head_dim * block_size


def main():
    parser = argparse.ArgumentParser(description="Token-budget pretraining")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, help="Directory with train.bin and val.bin")
    parser.add_argument("--out", help="Run directory (default: runs/<config name>)")
    parser.add_argument("--total_tokens", type=float, help="Override train.total_tokens")
    parser.add_argument("--stop_after_hours", type=float,
                        help="Checkpoint and exit after this long (e.g. to fit a Colab/Kaggle session); "
                             "rerun to continue")
    parser.add_argument("--stop_after_steps", type=int, help="Checkpoint and exit after this many steps")
    parser.add_argument("--benchmark", type=int, metavar="STEPS",
                        help="Run this many optimizer steps, print speed and memory, write nothing")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    tcfg = config["train"]
    if args.total_tokens:
        tcfg["total_tokens"] = int(args.total_tokens)
    out = Path(args.out or Path("runs") / Path(args.config).stem)
    ckpt_path = out / "ckpt.pt"
    device = args.device
    data_dir = Path(args.data)

    # On resume the model shape comes from the checkpoint; everything else from the config file
    ckpt = None
    if not args.benchmark and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt["config"]["model"] != config["model"]:
            print("⚠️  Model section differs from the checkpoint; using the checkpoint's")
        config["model"] = ckpt["config"]["model"]

    torch.manual_seed(tcfg.get("seed", 1337))
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    block_size = tcfg["block_size"]
    micro_bs = tcfg["micro_batch_size"]
    tokens_per_micro = micro_bs * block_size
    grad_accum = max(1, tcfg["tokens_per_step"] // tokens_per_micro)
    tokens_per_step = grad_accum * tokens_per_micro

    train_data = TokenData(data_dir / "train.bin", block_size, tcfg.get("seed", 1337))
    val_data = TokenData(data_dir / "val.bin", block_size)
    total_tokens = int(tcfg.get("total_tokens") or len(train_data))  # default: one epoch
    max_steps = max(1, total_tokens // tokens_per_step)

    model_cfg = ModelConfig(**config["model"])
    assert block_size <= model_cfg.context_length, "block_size is longer than the model's context_length"
    model = Transformer(model_cfg).to(device)
    model.ce_chunk = tcfg.get("ce_chunk")
    optimizer = MuonAdamW(model.optimizer_groups(tcfg))

    step, tokens_seen, train_seconds = 0, 0, 0.0
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step, tokens_seen = ckpt["step"], ckpt["tokens_seen"]
        train_seconds = ckpt.get("train_seconds", 0.0)
        print(f"✓ Resumed from {ckpt_path} at step {step}/{max_steps} ({tokens_seen / 1e6:.1f}M tokens)")
        del ckpt

    fwd = torch.compile(model) if tcfg.get("compile", False) else model
    fpt = flops_per_token(model, block_size)

    print(f"Params: {model.get_num_params() / 1e6:.1f}M | device: {device} | "
          f"precision: {autocast_dtype(device) or torch.float32}")
    print(f"Batch: {micro_bs} x {block_size} tokens x {grad_accum} accum = {tokens_per_step:,} tokens/step")
    print(f"Budget: {total_tokens / 1e9:.3f}B tokens = {max_steps:,} steps "
          f"(train.bin holds {len(train_data) / 1e9:.3f}B tokens)")

    if not args.benchmark:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(config, indent=2))
    log_file = None if args.benchmark else open(out / "log.jsonl", "a")

    def log(record):
        if log_file:
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

    def save_checkpoint():
        tmp = out / "ckpt.pt.tmp"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "train_seconds": train_seconds,
            "config": config,
        }, tmp)
        os.replace(tmp, ckpt_path)  # atomic: a crash mid-save never corrupts ckpt.pt
        print(f"💾 Checkpoint saved at step {step}")

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for x, y in val_data.eval_batches(micro_bs, tcfg.get("eval_tokens", 2_000_000)):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device):
                _, metrics = fwd(x, y)
            losses.append(metrics["lm_loss"].item())
        model.train()
        return sum(losses) / len(losses)

    @torch.no_grad()
    def sample():
        enc = get_tokenizer()
        prompt = tcfg.get("sample_prompt", "Once upon a time")
        idx = torch.tensor([[EOT] + enc.encode_ordinary(prompt)], device=device)
        model.eval()
        with autocast(device):
            outp = model.generate(idx, max_new_tokens=tcfg.get("sample_tokens", 120), stop_tokens=(EOT,))
        model.train()
        return enc.decode(outp[0, 1:].tolist())

    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"\n⚠️  Signal {signum}: finishing this step, then saving and exiting")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    ckpt_every = tcfg.get("checkpoint_minutes", 15) * 60
    eval_interval = tcfg.get("eval_interval", 250)
    sample_interval = tcfg.get("sample_interval", eval_interval)
    log_interval = tcfg.get("log_interval", 10)
    last_ckpt = run_start = time.time()
    start_step = step
    bench_times = []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    window_start, window_tokens = time.time(), 0

    while step < max_steps:
        t0 = time.time()
        mult = lr_multiplier(step, max_steps, tcfg)
        optimizer.set_lr_multiplier(mult)
        if model.mtp is not None:
            # V3: MTP weight 0.3, then 0.1 once the learning rate starts decaying
            decaying = step >= tcfg.get("warmup_steps", 0) and mult < 1.0
            model.mtp_weight = tcfg.get("mtp_weight_late" if decaying else "mtp_weight", 0.1 if decaying else 0.3)

        totals = {}
        for micro in range(grad_accum):
            x, y = train_data.batch(step * grad_accum + micro, micro_bs)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device):
                loss, metrics = fwd(x, y)
            (loss / grad_accum).backward()
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v / grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("max_grad_norm", 1.0))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if hasattr(model, "after_optimizer_step"):
            model.after_optimizer_step()  # e.g. MoE router bias updates

        step += 1
        tokens_seen += tokens_per_step
        window_tokens += tokens_per_step
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        train_seconds += dt

        if args.benchmark:
            bench_times.append(dt)
            print(f"bench step {len(bench_times)}/{args.benchmark}: {tokens_per_step / dt:,.0f} tok/s "
                  f"| loss {totals['lm_loss'].item():.3f}")
            if len(bench_times) >= args.benchmark:
                break
            continue

        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - window_start
            tok_s = window_tokens / elapsed
            eta_h = (max_steps - step) * tokens_per_step / tok_s / 3600
            values = {k: v.item() for k, v in totals.items()}
            extras = " | ".join(f"{k} {v:.4f}" for k, v in values.items() if k != "lm_loss")
            print(f"step {step}/{max_steps} | loss {values['lm_loss']:.4f} | lr x{mult:.3f} | "
                  f"grad {grad_norm.item():.2f} | {tok_s:,.0f} tok/s | {tok_s * fpt / 1e12:.2f} TFLOPS | "
                  f"ETA {eta_h:.1f}h" + (f" | {extras}" if extras else ""))
            log({"step": step, "tokens": tokens_seen, **values, "lr_mult": mult,
                 "grad_norm": grad_norm.item(), "tok_per_s": tok_s, "train_hours": train_seconds / 3600})
            window_start, window_tokens = time.time(), 0

        if (args.stop_after_steps and step - start_step >= args.stop_after_steps) or (
            args.stop_after_hours and time.time() - run_start >= args.stop_after_hours * 3600
        ):
            stop_requested = True
            print("⏹  Stop limit reached; saving so the run can be continued")

        done = step == max_steps
        did_extra = False
        if step % eval_interval == 0 or done:
            val_loss = evaluate()
            print(f"📊 step {step} | val loss {val_loss:.4f}")
            log({"step": step, "tokens": tokens_seen, "val_loss": val_loss})
            did_extra = True
        if sample_interval and (step % sample_interval == 0 or done):
            text = sample()
            print(f"📝 {text!r}")
            with open(out / "samples.txt", "a") as f:
                f.write(f"--- step {step} ({tokens_seen / 1e6:.0f}M tokens)\n{text}\n")
            did_extra = True
        if done or stop_requested or time.time() - last_ckpt >= ckpt_every:
            save_checkpoint()
            last_ckpt = time.time()
            did_extra = True
        if stop_requested and not done:
            return
        if did_extra:
            # Keep eval / sampling / checkpoint time out of the tok/s figure
            window_start, window_tokens = time.time(), 0

    if args.benchmark:
        steady = bench_times[2:] if len(bench_times) > 3 else bench_times
        tok_s = tokens_per_step * len(steady) / sum(steady)
        print("\n=== Benchmark ===")
        print(f"Throughput: {tok_s:,.0f} tok/s ({tok_s * fpt / 1e12:.2f} TFLOPS achieved)")
        if device == "cuda":
            print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        print(f"Full budget ({total_tokens / 1e9:.2f}B tokens) would take {total_tokens / tok_s / 3600:.1f}h")
        for hours in (1, 3, 24, 24 * 7):
            print(f"  tokens in {hours:>4}h: {tok_s * hours * 3600 / 1e9:.2f}B")
        return

    # Small weights-only file for sharing / Hugging Face upload
    torch.save({"model": model.state_dict(), "config": config,
                "step": step, "tokens_seen": tokens_seen}, out / "final.pt")
    print(f"✓ Done: {tokens_seen / 1e9:.2f}B tokens in {train_seconds / 3600:.2f}h → {out / 'final.pt'}")


if __name__ == "__main__":
    main()
