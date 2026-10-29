"""
Generate text from a pretrain.py checkpoint

    python sample.py runs/tinystories-30m/final.pt --prompt "Once upon a time" -n 3
    python sample.py runs/gpt2-124m/final.pt --interactive
"""

import argparse

import torch

from llm.tokenizer import EOT, get_tokenizer
from llm.utils import autocast, load_model


def main():
    parser = argparse.ArgumentParser(description="Sample from a checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("-n", "--num_samples", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--interactive", action="store_true", help="Read prompts from stdin")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    enc = get_tokenizer()
    model, ckpt = load_model(args.checkpoint, args.device)
    print(f"Loaded {model.get_num_params() / 1e6:.1f}M params, "
          f"{ckpt.get('tokens_seen', 0) / 1e9:.2f}B tokens trained")

    def complete(prompt: str):
        idx = torch.tensor([[EOT] + enc.encode_ordinary(prompt)] * args.num_samples, device=args.device)
        with autocast(args.device):
            out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                                 top_p=args.top_p, min_p=args.min_p,
                                 repetition_penalty=args.repetition_penalty, stop_tokens=(EOT,))
        for row in out:
            tokens = row[1:].tolist()
            if EOT in tokens:
                tokens = tokens[: tokens.index(EOT)]
            print("-" * 60)
            print(enc.decode(tokens))

    if not args.interactive:
        complete(args.prompt)
        return
    while True:
        try:
            prompt = input("\nprompt> ")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip():
            complete(prompt)


if __name__ == "__main__":
    main()
