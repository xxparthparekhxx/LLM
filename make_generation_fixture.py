"""Write web/test/fixtures/generation.json: torch's greedy continuation of a prompt.

    .venv/bin/python make_generation_fixture.py

The browser test runs the same prompt through the ONNX graph and the JS tokenizer
and compares the token ids, which checks the tokenizer, the cache and the sampler
end to end against the PyTorch model.
"""

import json
from pathlib import Path

import torch

from llm.tokenizer import EOT, get_tokenizer
from llm.utils import load_model

ROOT = Path(__file__).resolve().parent
PROMPT = "Once upon a time"
STEPS = 30


def main():
    enc = get_tokenizer()
    model, ckpt = load_model(ROOT / "runs/long-tinystories-33m/muon-deepseek/final.pt", "cpu")
    prompt_ids = [EOT] + enc.encode_ordinary(PROMPT)
    with torch.no_grad():
        out = model.generate(torch.tensor([prompt_ids]), STEPS, temperature=0, stop_tokens=(EOT,))
    generated = out[0].tolist()[len(prompt_ids):]
    fixture = {
        "prompt": PROMPT,
        "prompt_ids": prompt_ids,
        "max_new_tokens": STEPS,
        "token_ids": generated,
        "text": enc.decode(generated),
        "checkpoint_step": ckpt.get("step"),
        "torch": torch.__version__,
    }
    path = ROOT / "web" / "test" / "fixtures" / "generation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Wrote {path}: {len(generated)} tokens")
    print(fixture["text"])


if __name__ == "__main__":
    main()
