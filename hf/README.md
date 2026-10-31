---
language: en
library_name: onnx
pipeline_tag: text-generation
tags:
  - tinystories
  - onnx
  - from-scratch
  - small-language-model
  - text-generation
datasets:
  - roneneldan/TinyStories
---

# TinyStories 33M

A 33M-parameter decoder-only transformer trained from scratch on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). It writes
simple English children's stories from a short prompt. It is a small research and
demo model, not a chatbot or an assistant.

The model is implemented and trained in
[xxparthparekhxx/LLM](https://github.com/xxparthparekhxx/LLM), a from-scratch
PyTorch codebase with switchable DeepSeek-style components. This checkpoint uses
the dense base architecture; the optional MLA, MoE, mHC, Engram and MTP paths are
off. A browser demo built on this ONNX export lives in `web/` in that repository.

## Model details

| | |
|---|---|
| Parameters | 33,086,848 |
| Architecture | decoder-only transformer, pre-norm RMSNorm, RoPE, SwiGLU |
| Attention | grouped-query attention, 6 query heads, 2 KV heads, head dim 64 |
| Width / depth | `d_model` 384, 8 layers, FFN 1024 |
| Extra components | QK-norm, sigmoid attention gate, tanh logit soft-capping |
| Vocabulary | 50,304 (GPT-2 BPE, padded from 50,257; `<|endoftext|>` = 50,256) |
| Context length | 512 tokens |
| Precision | fp16 weights, fp32 graph inputs/outputs and normalization |
| Training data | TinyStories (English short stories) |
| Tokens seen | 449,970,176 (about 0.95 epoch) |
| Optimizer | Muon for hidden matrices, AdamW for embeddings/norms, WSD schedule |
| Batch | 65,536 tokens per step, 6,866 steps |
| Hardware | GTX 1650 Ti laptop GPU (4 GB), fp32 |
| Training time | 20.9 hours |
| Final validation loss | 1.3703 (1M-token eval) |

The final validation loss in the training log is 1.3703 over 1M validation
tokens. On the first 49,664 tokens of the validation file (the slice used for the
ONNX parity check) the loss is 1.0501; it is an easier, smaller slice and the two
numbers are not directly comparable.

## Files

| File | Size | Notes |
|---|---|---|
| `model_fp16.onnx` | 66.5 MB | Recommended. fp16 weights, fp32 inputs/outputs; measured parity below |
| `model.onnx` | 132.7 MB | fp32 reference export |

An int8 dynamic-quantized variant was measured but **not** included: only 96.46%
argmax agreement with fp32 (below the 98% bar) and 91.7 MB, larger than fp16. See
the parity table.

The tokenizer is not bundled as a `tokenizer.json`. Use the standard GPT-2 BPE
encoding from `tiktoken` (`tiktoken.get_encoding("gpt2")`) and prepend
`<|endoftext|>` (id 50256) to the prompt, as the training code does. IDs 50,257
through 50,303 are reserved for chat-format special tokens used in later
fine-tuning; they are not meaningful for this pretrained checkpoint.

## ONNX interface

The graph exposes an explicit key/value cache. One prefill call processes the
prompt with an empty cache; each following token is one call with sequence
length 1 and the previous cache.

| Input | Shape | Type |
|---|---|---|
| `input_ids` | `(1, T)` | int64 |
| `position_ids` | `(1, T)` | int64 |
| `past_keys` | `(8, 1, 2, P, 64)` | float32 |
| `past_values` | `(8, 1, 2, P, 64)` | float32 |
| `attention_mask` | `(T, P + T)` | bool |

| Output | Shape | Type |
|---|---|---|
| `logits` | `(1, T, 50304)` | float32 |
| `last_logits` | `(1, 1, 50304)` | float32 |
| `present_keys` | `(8, 1, 2, P + T, 64)` | float32 |
| `present_values` | `(8, 1, 2, P + T, 64)` | float32 |

`attention_mask` is `True` where a query may attend to a key. For a normal causal
prefill it is the lower triangle of an all-ones `(T, T)` matrix; for a single
cached token it is an all-ones `(1, P + 1)` row.

Minimal prefill with ONNX Runtime:

```python
import numpy as np
import onnxruntime as ort
import tiktoken

session = ort.InferenceSession("model_fp16.onnx", providers=["CPUExecutionProvider"])
enc = tiktoken.get_encoding("gpt2")
ids = [50256, *enc.encode_ordinary("Once upon a time")]
t = len(ids)

feeds = {
    "input_ids": np.array([ids], dtype=np.int64),
    "position_ids": np.array([list(range(t))], dtype=np.int64),
    "past_keys": np.zeros((8, 1, 2, 0, 64), dtype=np.float32),
    "past_values": np.zeros((8, 1, 2, 0, 64), dtype=np.float32),
    "attention_mask": np.tril(np.ones((t, t), dtype=bool)),
}
logits, last_logits, keys, values = session.run(None, feeds)
next_id = int(last_logits[0, 0].argmax())  # greedy
```

For cached decoding, reuse `keys`/`values` as `past_keys`/`past_values`, pass the
new token with its position, and use an all-ones mask of shape `(1, P + 1)`. The
browser implementation (including sampling, streaming and the 512-token sliding
window) is in `web/src/generate.ts`.

## Parity and quantization

Measured against the PyTorch fp32 model on the first 49,664 TinyStories
validation tokens:

| Variant | Size | Max abs logit diff | Argmax agreement | Val loss |
|---|---|---|---|---|
| fp32 | 132.7 MB | 4.1e-5 | 100.00% | 1.0501 |
| **fp16 (included)** | **66.5 MB** | **0.035724** | **99.91%** | **1.0501** |
| int8 dynamic (not included) | 91.7 MB | 2.923301 | 96.46% | 1.0556 |

Cached step decoding was also checked against a full forward pass over 200
positions: max abs logit difference 3.4e-5, 100% argmax agreement. Greedy
generation matches PyTorch's `Transformer.generate` token for token.

## Intended use

- Generating short, simple English stories from a prompt such as
  "Once upon a time".
- Running a small language model locally or in a browser via ONNX Runtime, as a
  demo or teaching example.
- Research on small models, quantization and on-device inference.

## Out of scope

- Conversation, instruction following, question answering, summarization,
  translation, code, or factual/knowledge tasks.
- Production use, safety-critical use, or any use where incorrect or harmful
  text could matter.
- Languages other than English. The training data is English children's
  stories; other scripts are tokenizable but not modeled well.
- Contexts longer than 512 tokens. The browser demo slides the window, but the
  model was trained at 512.
- Personas or claims about real people. Any names and events in its output are
  invented.

## Limitations and biases

TinyStories is synthetic and deliberately simple: vocabulary, plots and
characters are repetitive (children, parents, animals, parks, school). The model
inherits the dataset's stereotypes and simplistic views of family and gender
roles, and it has no safety tuning or content filtering. It can produce
repetitive, ungrammatical or nonsensical text, especially with high temperature
or unusual prompts. Sampling defaults used in the demo are temperature 0.8,
top-k 50, top-p 1.0; temperature 0 is greedy.

No benchmark results are claimed beyond validation loss and the ONNX parity
measurements above.

## License

The source repository [xxparthparekhxx/LLM](https://github.com/xxparthparekhxx/LLM)
has **no LICENSE file**, so no license is granted for this model or its weights.
Treat them as all rights reserved unless the repository owner states otherwise.
The training dataset, TinyStories, has its own terms; see its dataset card.

## Citation

The dataset:

```bibtex
@article{eldan2023tinystories,
  title   = {TinyStories: How Small Can Language Models Be and Still Speak Coherent English?},
  author  = {Ronen Eldan and Yuanzhi Li},
  journal = {arXiv preprint arXiv:2305.07759},
  year    = {2023}
}
```

The model and training code have no associated paper. If you use the code, link
to [xxparthparekhxx/LLM](https://github.com/xxparthparekhxx/LLM).
