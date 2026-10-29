# LLM from scratch

A small decoder-only language model written from scratch in PyTorch, with the
architecture ideas from DeepSeek's papers (V2 through V4.1-Flash) built in as
switchable components, and a training setup designed for a tiny compute budget:
a 4 GB laptop GPU and a few hours of a rented one.

## What's in the model

The base is a modern dense transformer:

| Component | Source |
|---|---|
| Pre-norm RMSNorm, rotary positions, SwiGLU, no biases | LLaMA |
| Grouped-query attention | GQA (Ainslie et al., 2023) |
| QK-norm | Qwen3, OLMo 2, Gemma 3, DeepSeek-V4 |
| Sigmoid gate on the attention output | Gated Attention (Qwen, 2025) |
| Tanh logit soft-capping | Gemma 2 |
| Muon optimizer for hidden matrices, AdamW for the rest | Keller Jordan (2024) |
| Warmup–stable–decay learning rate | DeepSeek-V3/V4, MiniCPM |

DeepSeek components, each off by default and switched on in the config:

| Config | Component | Paper |
|---|---|---|
| `attention: "mla"` | Multi-head latent attention: keys and values from a small shared latent, decoupled RoPE key | DeepSeek-V2 §2.1, V3 §2.1.1 |
| `moe: true` | DeepSeekMoE: shared + fine-grained routed experts, √softplus routing, auxiliary-loss-free load balancing | V3 §2.1.2, V4 §2.1 |
| `hc_streams: 4` | mHC: manifold-constrained hyper-connections (n residual streams mixed by doubly-stochastic matrices) | [arXiv 2512.24880](https://arxiv.org/abs/2512.24880), V4 §2.2 |
| `engram_layers: [1]` | Engram: hashed n-gram memory gated into the residual stream | [arXiv 2601.07372](https://arxiv.org/abs/2601.07372), V4.1-Flash §2.4 |
| `mtp_depth: 1` | Multi-token prediction | V3 §2.2 |
| `doc_masking: true` | No attention across document boundaries | V4 §4.1 |
| `swiglu_clamp: 10` | Clamped SwiGLU branches | V4 §4.2.3 |
| `muon_mode: "deepseek"` | Muon with hybrid Newton–Schulz and update RMS matched to AdamW | V4 §2.4 |

Not implemented, on purpose: DeepSeek Sparse Attention and V4's compressed
attention (CSA/HCA). They exist to make 64K–1M-token contexts affordable; at
this model's 512–1024 context the top-k selection would pick every token
anyway. FP8/FP4 training is also out of scope.

## Quick start

```bash
uv venv --python 3.12 && uv pip install -e . pytest
python -m pytest                       # 27 tests, CPU, ~20 s

# Tokenize (GPT-2 BPE, streamed to flat uint16 files)
python prepare_data.py --dataset roneneldan/TinyStories --out data/tinystories
python prepare_data.py --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \
    --out data/fineweb-edu --max_tokens 2.6e9 --val_tokens 1e7

# Measure speed and memory first, then train
python pretrain.py --config configs/tinystories-33m.json --data data/tinystories --benchmark 10
python pretrain.py --config configs/tinystories-33m.json --data data/tinystories

python sample.py runs/tinystories-33m/final.pt --prompt "Once upon a time" -n 3
python eval_hellaswag.py runs/gpt2-124m/final.pt
```

`pretrain.py` trains to a fixed token budget, checkpoints every 15 minutes, and
resumes exactly (same step, same data order) when rerun with the same command.
`--stop_after_hours 2.75` saves and exits early, for time-limited sessions such
as Colab or Kaggle; running it again continues the same schedule.

## Ablations

`ablate.py` trains each variant (the base plus one DeepSeek component, and all
of them together) on the same data and token budget; `report.py` turns the logs
into a table and a validation-loss chart.

```bash
python ablate.py --config configs/tinystories-33m.json --data data/tinystories --tokens 5e7
python report.py runs/ablation-tinystories-33m
```

Results: in progress.

## Watching a run

`dashboard.py` serves a live page on the local network with GPU load, the
current run's progress and time left, loss and throughput curves for every
run in a group, and each run's latest sample. It only reads the files
`pretrain.py` writes and needs no extra packages.

```bash
python dashboard.py            # http://<machine>:8000
```

## Hardware notes

Measured on a GTX 1650 Ti (4 GB, Turing):

- fp16 matmuls run at 0.41 TFLOPS, against 1.78 for fp32, and bfloat16 is only
  emulated. So pre-Ampere GPUs train in plain fp32 here, and bfloat16 is used
  on Ampere and newer (`llm/utils.py`). Note that `torch.cuda.is_bf16_supported()`
  returns `True` on this card even though bf16 is emulated.
- The 33M TinyStories model trains at about 6,000 tokens/s (74% of fp32 peak)
  in under 2 GB, so one epoch of TinyStories (474M tokens) takes about 22 hours.
- The loss is computed in chunks with recomputation (`ce_chunk`), because the
  full (tokens × 50k vocab) fp32 logits don't fit in 4 GB.

## Layout

```
llm/
  config.py      ModelConfig: every architecture switch
  model.py       Transformer, blocks, MTP, generation and sampling
  attention.py   GQA, MLA, RoPE, KV cache, document masking
  moe.py         DeepSeekMoE and SwiGLU
  hyper.py       mHC
  engram.py      Engram
  muon.py        Muon + AdamW optimizer
  data.py        token-file windows with exact-resume shuffling
  tokenizer.py   GPT-2 BPE plus reserved chat tokens
prepare_data.py  tokenize a Hugging Face dataset to train.bin / val.bin
pretrain.py      token-budget training
ablate.py        architecture sweeps
report.py        tables and charts from run logs
dashboard.py     live training dashboard
sample.py        text generation
eval_hellaswag.py
```
