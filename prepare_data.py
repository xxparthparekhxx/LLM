"""
Pre-tokenize a Hugging Face text dataset into flat token files for pretrain.py

Streams the dataset, so RAM stays flat however big the dataset is. Writes
<out>/train.bin and <out>/val.bin as uint16 GPT-2 token ids (llm/tokenizer.py),
with <|endoftext|> before every document, plus <out>/meta.json.

Examples:
    # TinyStories (has its own validation split)
    python prepare_data.py --dataset roneneldan/TinyStories --out data/tinystories

    # FineWeb-Edu: first 10M tokens become val, then 2.6B tokens of train
    python prepare_data.py --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \\
        --out data/fineweb-edu --max_tokens 2.6e9 --val_tokens 1e7

    # Local parquet files (e.g. fetched with `hf download`) go through the same path
    python prepare_data.py --dataset parquet --data_files "raw/*.parquet" --out data/fineweb-edu
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from llm.tokenizer import EOT, get_tokenizer


def open_stream(args, split):
    from datasets import load_dataset

    if args.data_files:
        return load_dataset(args.dataset, data_files=args.data_files, split=split, streaming=True)
    return load_dataset(args.dataset, args.subset, split=split, streaming=True)


def iter_token_batches(stream, enc, text_column, batch_docs):
    """Yield one flat uint16 array per batch of documents"""
    eot = EOT
    threads = os.cpu_count() or 4
    batch = []

    def flush():
        encoded = enc.encode_ordinary_batch(batch, num_threads=threads)
        flat = []
        for tokens in encoded:
            flat.append(eot)
            flat.extend(tokens)
        batch.clear()
        return np.array(flat, dtype=np.uint16)

    for row in stream:
        text = row.get(text_column)
        if text:
            batch.append(text)
        if len(batch) >= batch_docs:
            yield flush()
    if batch:
        yield flush()


def write_tokens(batches, path, limit, desc):
    """Write batches to path until limit tokens; returns (tokens written, leftover array)"""
    written = 0
    leftover = None
    with open(path, "wb") as f, tqdm(total=limit, unit="tok", unit_scale=True, desc=desc) as pbar:
        for arr in batches:
            if limit is not None and written + len(arr) >= limit:
                take = limit - written
                arr[:take].tofile(f)
                written += take
                pbar.update(take)
                leftover = arr[take:]
                break
            arr.tofile(f)
            written += len(arr)
            pbar.update(len(arr))
    return written, leftover


def main():
    parser = argparse.ArgumentParser(description="Tokenize a dataset into train.bin / val.bin")
    parser.add_argument("--dataset", required=True, help="HF dataset name, or 'parquet'/'json' with --data_files")
    parser.add_argument("--subset", default=None, help="Dataset config name, e.g. sample-10BT")
    parser.add_argument("--data_files", default=None, help="Glob of local files when --dataset is a file format")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--max_tokens", type=float, default=None, help="Stop train.bin after this many tokens")
    parser.add_argument("--val_tokens", type=float, default=1e7,
                        help="Val size taken from the start of train, used only when there is no validation split")
    parser.add_argument("--val_split", default="validation", help="Name of the validation split, if any")
    parser.add_argument("--batch_docs", type=int, default=1024)
    args = parser.parse_args()

    enc = get_tokenizer()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    max_tokens = int(args.max_tokens) if args.max_tokens else None

    # Validation: use the dataset's own split when it has one
    try:
        val_stream = open_stream(args, args.val_split) if not args.data_files else None
    except (ValueError, KeyError):
        val_stream = None

    train_batches = iter_token_batches(open_stream(args, "train"), enc, args.text_column, args.batch_docs)

    if val_stream is not None:
        val_batches = iter_token_batches(val_stream, enc, args.text_column, args.batch_docs)
        n_val, _ = write_tokens(val_batches, out / "val.bin", None, "val")
        leftover = None
    else:
        # No validation split: the first val_tokens of the train stream become val
        n_val, leftover = write_tokens(train_batches, out / "val.bin", int(args.val_tokens), "val (from train)")

    def train_source():
        if leftover is not None and len(leftover):
            yield leftover
        yield from train_batches

    n_train, _ = write_tokens(train_source(), out / "train.bin", max_tokens, "train")

    meta = {
        "dataset": args.dataset,
        "subset": args.subset,
        "data_files": args.data_files,
        "tokenizer": enc.name,
        "eot_token": EOT,
        "dtype": "uint16",
        "train_tokens": n_train,
        "val_tokens": n_val,
        "val_from_train": val_stream is None,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
