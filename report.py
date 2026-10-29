"""
Summarize a group of pretrain.py runs: a markdown table and a validation-loss chart

    python report.py runs/ablation-tinystories-33m
"""

import argparse
import json
from pathlib import Path


def read_log(path: Path):
    train, val = [], []
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        (val if "val_loss" in rec else train).append(rec)
    return train, val


def main():
    parser = argparse.ArgumentParser(description="Summarize runs")
    parser.add_argument("group", help="Directory containing one subdirectory per run")
    args = parser.parse_args()
    group = Path(args.group)

    rows, curves = [], {}
    for run in sorted(p for p in group.iterdir() if (p / "log.jsonl").exists()):
        train, val = read_log(run / "log.jsonl")
        if not val:
            continue
        curves[run.name] = val
        speeds = [r["tok_per_s"] for r in train[len(train) // 10 :]] or [float("nan")]
        rows.append({
            "run": run.name,
            "tokens": val[-1]["tokens"],
            "val_loss": val[-1]["val_loss"],
            "best_val": min(r["val_loss"] for r in val),
            "tok_s": sum(speeds) / len(speeds),
            "hours": train[-1]["train_hours"] if train else float("nan"),
            "done": (run / "final.pt").exists(),
        })
    if not rows:
        print("No runs with validation results yet")
        return

    base = next((r["val_loss"] for r in rows if r["run"] == "baseline"), None)
    rows.sort(key=lambda r: r["val_loss"])
    lines = ["| Run | Tokens | Val loss | vs baseline | Tokens/s | Hours | Status |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        delta = f"{r['val_loss'] - base:+.4f}" if base is not None and r["run"] != "baseline" else ""
        lines.append(f"| {r['run']} | {r['tokens'] / 1e6:.0f}M | {r['val_loss']:.4f} | {delta} | "
                     f"{r['tok_s']:,.0f} | {r['hours']:.2f} | {'done' if r['done'] else 'running'} |")
    table = "\n".join(lines)
    print(table)
    (group / "results.md").write_text(table + "\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: skipping the chart)")
        return
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for name, val in sorted(curves.items()):
        ax.plot([r["tokens"] / 1e6 for r in val], [r["val_loss"] for r in val], label=name,
                linewidth=2.2 if name == "baseline" else 1.4)
    ax.set_xlabel("Training tokens (M)")
    ax.set_ylabel("Validation loss")
    ax.set_title(group.name)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    lo = min(r["best_val"] for r in rows)
    ax.set_ylim(lo - 0.05, lo + 1.0)
    fig.tight_layout()
    fig.savefig(group / "val_loss.png")
    print(f"\nWrote {group / 'results.md'} and {group / 'val_loss.png'}")


if __name__ == "__main__":
    main()
