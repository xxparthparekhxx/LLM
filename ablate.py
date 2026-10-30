"""
Train architecture variants on the same data and token budget, one after another

Each variant is the base config plus a few overrides, run through pretrain.py
into runs/<tag>/<variant>/. Finished variants are skipped and interrupted ones
resume, so rerunning the same command just continues the sweep.

    python ablate.py --config configs/tinystories-33m.json --data data/tinystories --tokens 5e7
    python ablate.py --config ... --data ... --only baseline mhc --benchmark 5
    python report.py runs/ablation-tinystories
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DEEPSEEK_MUON = {"train": {"muon_mode": "deepseek", "weight_decay": 0.1}}
DOC_MASK_CLAMP = {"model": {"doc_masking": True, "swiglu_clamp": 10.0}}
MHC = {"model": {"hc_streams": 4}}
ENGRAM = {"model": {"engram_layers": [1]}}
MLA = {"model": {"attention": "mla", "mla_kv_rank": 256, "mla_rope_dim": 32}}
MOE = {"model": {"moe": True, "moe_experts": 8, "moe_top_k": 2, "moe_shared": 1, "moe_expert_dim": 352}}
MTP = {"model": {"mtp_depth": 1}}


def merge(*parts):
    out = {}
    for part in parts:
        for section, values in part.items():
            out.setdefault(section, {}).update(values)
    return out


VARIANTS = {
    "baseline": {},
    "muon-deepseek": DEEPSEEK_MUON,
    "docmask-clamp": DOC_MASK_CLAMP,
    "mhc": MHC,
    "engram": ENGRAM,
    "mla": MLA,
    "moe": MOE,
    "mtp": MTP,
    # Half the micro-batch (same tokens per step) to fit in 4 GB with everything on
    "deepseek-all": merge(DEEPSEEK_MUON, DOC_MASK_CLAMP, MHC, ENGRAM, MLA, MOE, MTP,
                          {"train": {"micro_batch_size": 4}}),
    # The parts that beat the baseline at little speed cost; leaves out doc
    # masking and MTP (each ~40% slower) and Engram (no gain at this size)
    "deepseek-lite": merge(DEEPSEEK_MUON, MHC, MLA, MOE),
}


def variant_config(base: dict, overrides: dict) -> dict:
    config = copy.deepcopy(base)
    for section, values in overrides.items():
        config[section].update(values)
    return config


def main():
    parser = argparse.ArgumentParser(description="Run architecture ablations")
    parser.add_argument("--config", required=True, help="Base config; variants override parts of it")
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokens", type=float, required=True, help="Token budget per variant")
    parser.add_argument("--tag", default=None, help="Run group name (default: ablation-<config name>)")
    parser.add_argument("--only", nargs="+", choices=list(VARIANTS), help="Run just these variants")
    parser.add_argument("--evals", type=int, default=8, help="Validation points per run")
    parser.add_argument("--warmup_frac", type=float, default=0.05, help="Warmup as a fraction of each run")
    parser.add_argument("--benchmark", type=int, metavar="STEPS", help="Speed/memory check per variant")
    args = parser.parse_args()

    base = json.loads(Path(args.config).read_text())
    group = ROOT / "runs" / (args.tag or f"ablation-{Path(args.config).stem}")
    names = args.only or list(VARIANTS)

    for name in names:
        out = group / name
        if not args.benchmark and (out / "final.pt").exists():
            print(f"✓ {name}: done, skipping")
            continue
        out.mkdir(parents=True, exist_ok=True)
        config = variant_config(base, VARIANTS[name])
        steps = int(args.tokens) // config["train"]["tokens_per_step"]
        config["train"]["eval_interval"] = max(1, steps // args.evals)
        config["train"]["warmup_steps"] = max(1, round(steps * args.warmup_frac))
        cfg_path = out / "variant.json"
        cfg_path.write_text(json.dumps(config, indent=2))
        cmd = [sys.executable, "-u", str(ROOT / "pretrain.py"), "--config", str(cfg_path),
               "--data", args.data, "--out", str(out), "--total_tokens", str(args.tokens)]
        if args.benchmark:
            cmd += ["--benchmark", str(args.benchmark)]
        print(f"\n{'=' * 70}\n▶ {name}\n{'=' * 70}", flush=True)
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"✗ {name} exited with {result.returncode}; continuing with the next variant")


if __name__ == "__main__":
    main()
