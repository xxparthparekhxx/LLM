"""Export a pretrain.py checkpoint to ONNX with a KV cache

    python export_onnx.py runs/.../final.pt --out model/model.onnx

The exported graph is the trained Transformer (master's model code, unchanged)
with explicit past/present key and value tensors, so each new token is one small
step instead of a full-context forward pass. Generation must match the PyTorch
model in sample.py; measure_parity.py checks that.

Inputs
  input_ids      (1, T)              int64    tokens to process
  position_ids   (1, T)              int64    their positions in the sequence
  past_keys      (layers, 1, kv, P, head_dim) float32
  past_values    (layers, 1, kv, P, head_dim) float32
  attention_mask (T, P + T)          bool     True where a query may attend

Outputs
  logits         (1, T, vocab)       float32  every position, for scoring
  last_logits    (1, 1, vocab)       float32  only the last position, for generation
  present_keys   (layers, 1, kv, P + T, head_dim)
  present_values (layers, 1, kv, P + T, head_dim)

With P = 0 the prompt is prefilled in one call; every later call passes T = 1 and
the cache it got back. The graph only contains the inference path: no MTP, no
MoE/Engram/mHC, because this checkpoint does not use them.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from llm.utils import load_model

ROOT = Path(__file__).resolve().parent


def rms_norm(x, weight, eps):
    # Written as sum/size instead of mean, so torch's graph passes do not fuse it
    # back into aten::rms_norm, which the ONNX exporter cannot lower. The
    # normalization itself runs in float32, which keeps an fp16 export stable.
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) / xf.shape[-1] + eps) * weight.float()).to(x.dtype)


def rotate(x, cos, sin):
    """RoPE for (B, H, T, D); cos/sin are (1, 1, T, D/2)"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class CachedTransformer(nn.Module):
    """The trained model's weights under an explicit-cache forward signature"""

    def __init__(self, model):
        super().__init__()
        self.model = model
        cfg = model.config
        assert cfg.attention == "gqa" and not cfg.moe and not cfg.engram_layers
        assert cfg.hc_streams == 1 and cfg.mtp_depth == 0
        self.rope_cos = model.rope_cos
        self.rope_sin = model.rope_sin
        self.eps = cfg.norm_eps
        self.n_heads, self.n_kv, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.repeat = self.n_heads // self.n_kv
        self.softcap = cfg.logit_softcap

    def attention(self, block, x, cos, sin, past_k, past_v, mask):
        a = block.attn
        b, t, _ = x.shape
        q = rms_norm(a.q_proj(x).view(b, t, self.n_heads, self.head_dim), a.q_norm.weight, self.eps)
        k = rms_norm(a.k_proj(x).view(b, t, self.n_kv, self.head_dim), a.k_norm.weight, self.eps)
        v = a.v_proj(x).view(b, t, self.n_kv, self.head_dim)
        q = rotate(q.transpose(1, 2), cos, sin)
        k = rotate(k.transpose(1, 2), cos, sin)
        v = v.transpose(1, 2)

        k = torch.cat((past_k, k), dim=2)
        v = torch.cat((past_v, v), dim=2)
        new_k, new_v = k, v  # (B, n_kv, total, D) for the cache
        if self.repeat > 1:
            # Same expansion as repeat_interleave, written with reshape so ONNX keeps it exact.
            k = k.unsqueeze(2).expand(b, self.n_kv, self.repeat, k.size(2), self.head_dim).reshape(b, self.n_heads, k.size(2), self.head_dim)
            v = v.unsqueeze(2).expand(b, self.n_kv, self.repeat, v.size(2), self.head_dim).reshape(b, self.n_heads, v.size(2), self.head_dim)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        scores = torch.where(mask, scores, torch.full_like(scores, float("-inf")))
        y = torch.matmul(torch.softmax(scores, dim=-1), v)
        y = y.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
        y = y * torch.sigmoid(a.gate(x))
        return a.o_proj(y), new_k, new_v

    def forward(self, input_ids, position_ids, past_keys, past_values, attention_mask):
        cfg = self.model.config
        dtype = self.model.embed.weight.dtype
        # The graph's I/O stays float32 in both variants: the cache tensors and
        # logits cross the JS boundary, and an fp16 model only casts them inside.
        past_keys = past_keys.to(dtype)
        past_values = past_values.to(dtype)
        cos = self.rope_cos[position_ids].unsqueeze(1).to(dtype)
        sin = self.rope_sin[position_ids].unsqueeze(1).to(dtype)
        mask = attention_mask.unsqueeze(0).unsqueeze(0)

        x = self.model.embed(input_ids)
        present_keys = []
        present_values = []
        for i, block in enumerate(self.model.blocks):
            xn = rms_norm(x, block.attn_norm.weight, self.eps)
            attn_out, new_k, new_v = self.attention(block, xn, cos, sin, past_keys[i], past_values[i], mask)
            present_keys.append(new_k)
            present_values.append(new_v)
            x = x + attn_out
            mlp = block.mlp
            xn = rms_norm(x, block.mlp_norm.weight, self.eps)
            gate = torch.nn.functional.silu(mlp.gate_proj(xn))
            x = x + mlp.down_proj(gate * mlp.up_proj(xn))

        h = rms_norm(x, self.model.norm.weight, self.eps)
        # Weight on the left, so this MatMul uses the embedding matrix without a
        # transposed copy in the graph (the exporter would otherwise write it twice).
        logits = torch.matmul(self.model.embed.weight, h.transpose(1, 2)).transpose(1, 2).float()
        if self.softcap:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return (logits, logits[:, -1:, :],
                torch.stack(present_keys, dim=0).float(), torch.stack(present_values, dim=0).float())


def dedupe_initializers(path: Path):
    """Drop initializers that are byte-identical to an earlier one.

    The tied embedding is used twice (embedding lookup and the output projection),
    and the legacy exporter writes it as two separate 77 MB initializers.
    """
    import hashlib

    import onnx

    model = onnx.load(str(path))
    seen = {}
    rename = {}
    keep = []
    saved = 0
    for init in model.graph.initializer:
        key = (init.data_type, tuple(init.dims), hashlib.sha256(init.raw_data).hexdigest())
        if key in seen:
            rename[init.name] = seen[key]
            saved += init.ByteSize()
        else:
            seen[key] = init.name
            keep.append(init)
    if rename:
        for node in model.graph.node:
            for i, name in enumerate(node.input):
                if name in rename:
                    node.input[i] = rename[name]
        del model.graph.initializer[:]
        model.graph.initializer.extend(keep)
        onnx.save(model, str(path))
    print(f"Deduplicated initializers: saved {saved / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?", default=str(ROOT / "runs/long-tinystories-33m/muon-deepseek/final.pt"))
    parser.add_argument("--out", default=str(ROOT / "web/model/model.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--variant", choices=["fp32", "fp16", "int8"], default="fp32",
                        help="fp32 (default), fp16 weights, or dynamic int8 quantization")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")

    model, ckpt = load_model(args.checkpoint, "cpu")
    cfg = model.config
    print(f"Loaded {model.get_num_params() / 1e6:.2f}M params from {args.checkpoint} (step {ckpt.get('step')})")
    print(f"  {cfg.n_layers} layers, d_model {cfg.d_model}, {cfg.n_heads} heads, {cfg.n_kv_heads} KV heads, "
          f"head_dim {cfg.head_dim}, ffn {cfg.ffn_dim}, vocab {cfg.vocab_size}, context {cfg.context_length}")

    wrapper = CachedTransformer(model).eval()
    b, t = 1, 16
    probe = torch.randint(0, cfg.vocab_size, (b, t), dtype=torch.long)
    with torch.no_grad():
        moved = wrapper(probe, torch.arange(t).unsqueeze(0),
                        torch.zeros(cfg.n_layers, b, cfg.n_kv_heads, 0, cfg.head_dim),
                        torch.zeros(cfg.n_layers, b, cfg.n_kv_heads, 0, cfg.head_dim),
                        torch.tril(torch.ones(t, t, dtype=torch.bool)))[0]
        reference = model(probe)
    diff = (moved - reference).abs().max().item()
    print(f"Wrapper vs model, full forward: max |dlogit| {diff:.3g}")
    assert diff < 1e-3, "the export wrapper does not reproduce the model"
    if args.variant == "fp16":
        wrapper.half()
        print("Converted the weights and activations to float16 (I/O stays float32)")

    t = 7
    input_ids = torch.randint(0, cfg.vocab_size, (b, t), dtype=torch.long)
    position_ids = torch.arange(t, dtype=torch.long).unsqueeze(0)
    past = torch.zeros(cfg.n_layers, b, cfg.n_kv_heads, 0, cfg.head_dim)
    mask = torch.ones(t, t, dtype=torch.bool).tril()

    print(f"Exporting to {out} (opset {args.opset}, {args.variant}) ...")
    tmp = out.parent / "_export.onnx"
    torch.onnx.export(
        wrapper,
        (input_ids, position_ids, past, past, mask),
        str(tmp),
        input_names=["input_ids", "position_ids", "past_keys", "past_values", "attention_mask"],
        output_names=["logits", "last_logits", "present_keys", "present_values"],
        dynamic_axes={
            "input_ids": {1: "seq"},
            "position_ids": {1: "seq"},
            "past_keys": {3: "past"},
            "past_values": {3: "past"},
            "attention_mask": {0: "seq", 1: "total"},
            "logits": {1: "seq"},
            "present_keys": {3: "total"},
            "present_values": {3: "total"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    dedupe_initializers(tmp)

    if args.variant == "fp32" or args.variant == "fp16":
        tmp.replace(out)
    else:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            str(tmp),
            str(out),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
            per_channel=True,
            reduce_range=False,
        )
        tmp.unlink()
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
