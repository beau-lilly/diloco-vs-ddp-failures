"""One-time TinyStories tokenization into a uint16 .bin file (nanoGPT convention).

Run this once on the target machine before Group A launches:

    python prepare_tinystories.py --out data_cache/tinystories_train.bin --split train
    python prepare_tinystories.py --out data_cache/tinystories_val.bin   --split validation

Requires `tiktoken` and `datasets`. Not imported by the training loop.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["train", "validation"], required=True)
    ap.add_argument("--max-docs", type=int, default=None, help="optional cap for quick dev")
    args = ap.parse_args()

    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    ds = load_dataset("roneneldan/TinyStories", split=args.split)
    if args.max_docs is not None:
        ds = ds.select(range(min(args.max_docs, len(ds))))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    buf: list[int] = []
    flush_every = 4096
    total = 0
    with open(out_path, "wb") as f:
        for i, row in enumerate(ds):
            ids = enc.encode_ordinary(row["text"])
            ids.append(eot)
            buf.extend(ids)
            if len(buf) >= flush_every * 1024:
                arr = np.asarray(buf, dtype=np.uint16)
                f.write(arr.tobytes())
                total += arr.size
                buf = []
        if buf:
            arr = np.asarray(buf, dtype=np.uint16)
            f.write(arr.tobytes())
            total += arr.size
    print(f"wrote {total:,} tokens to {out_path}")


if __name__ == "__main__":
    main()
