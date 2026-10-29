"""
HellaSwag accuracy for a pretrain.py checkpoint (same method as nanoGPT / llm.c)

Each example has a context and 4 endings; the model picks the ending with the
lowest average per-token loss (acc_norm). Uses the 10,042-example validation set.

Reference on this eval: GPT-2 124M scores ~29.6% acc_norm, random is 25%.

    python eval_hellaswag.py runs/gpt2-124m/final.pt
    python eval_hellaswag.py runs/gpt2-124m/ckpt.pt --limit 1000   # quick check
"""

import argparse
import json
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from llm.tokenizer import EOT, get_tokenizer
from llm.utils import autocast, load_model

VAL_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"


def load_examples(cache_dir: Path):
    path = cache_dir / "hellaswag_val.jsonl"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {VAL_URL}")
        urllib.request.urlretrieve(VAL_URL, path)
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def build_batch(example, enc, max_len):
    """(4, T) tokens, (4, T) mask over ending tokens, and the label"""
    ctx = enc.encode_ordinary(example["ctx"])
    rows, masks = [], []
    for ending in example["endings"]:
        end = enc.encode_ordinary(" " + ending)
        tokens = ([EOT] + ctx + end)[-max_len:]
        rows.append(tokens)
        masks.append([0] * (len(tokens) - len(end)) + [1] * len(end))
    width = max(len(r) for r in rows)
    tokens = torch.zeros((4, width), dtype=torch.long)
    mask = torch.zeros((4, width), dtype=torch.long)
    for i, (r, m) in enumerate(zip(rows, masks)):
        tokens[i, : len(r)] = torch.tensor(r)
        mask[i, : len(m)] = torch.tensor(m)
    return tokens, mask, example["label"]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="HellaSwag eval for a pretrain.py checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("--limit", type=int, help="Only score the first N examples")
    parser.add_argument("--cache_dir", default="data/hellaswag")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    enc = get_tokenizer()
    model, _ = load_model(args.checkpoint, args.device)

    correct = correct_norm = total = 0
    examples = list(load_examples(Path(args.cache_dir)))[: args.limit]
    pbar = tqdm(examples, desc="HellaSwag")
    for example in pbar:
        tokens, mask, label = build_batch(example, enc, model.config.context_length)
        tokens, mask = tokens.to(args.device), mask.to(args.device)
        with autocast(args.device):
            logits = model(tokens)
        # Loss of predicting token t+1 from position t, counted only over the ending
        losses = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1),
            reduction="none",
        ).view(4, -1)
        shift_mask = mask[:, 1:].float()
        sum_loss = (losses * shift_mask).sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        total += 1
        correct += int(sum_loss.argmin().item() == label)
        correct_norm += int(avg_loss.argmin().item() == label)
        pbar.set_postfix(acc_norm=f"{correct_norm / total:.4f}")

    print(f"\nHellaSwag ({total} examples): acc {correct / total:.4f} | acc_norm {correct_norm / total:.4f}")
    print("Reference: GPT-2 124M ~0.2955 acc_norm, random 0.25")


if __name__ == "__main__":
    main()
