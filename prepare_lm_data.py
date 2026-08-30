"""
One-time pre-tokenization: raw docs (local parquet dir or HF hub id) -> flat uint16
token-id memmap binaries (train.bin, val.bin) + meta.json.

This is what actually gets rsynced to HPC, not the parquet. Loading a memmap at train
time needs zero `datasets`/tokenizer/hub access (train_stoch_layernorm_gpt.py's
build_lm_loaders_pretokenized reads meta.json for vocab_size and never touches the
tokenizer), which matters because HPC compute nodes commonly have no outbound network.
It's also reusable across every --block_size you try later, since chunking happens at
load time via memmap slicing, not baked in at prep time -- unlike the parquet +
datasets.map() pipeline in train_stoch_layernorm_gpt.py, which re-tokenizes and re-caches an
Arrow blob from scratch for every distinct block_size.

Reuses train_stoch_layernorm_gpt.py's load_raw_docs (held-out-validation-split logic) and
eos_tokenize_fn (EOS-between-docs) so the token stream this produces is IDENTICAL to
what the parquet-based pipeline would produce for the same dataset -- this is a pure
re-encoding of the same pipeline, not a second implementation to keep in sync by hand.
"""

import argparse
import json
import os

import numpy as np
from transformers import GPT2TokenizerFast

from train_stoch_layernorm_gpt import eos_tokenize_fn, load_raw_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_name", required=True, help="local parquet dir or HF hub dataset id")
    ap.add_argument("--data_config", default="")
    ap.add_argument("--model_name", default="gpt2")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_docs", type=int, default=5000)
    ap.add_argument("--num_proc", type=int, default=8)
    ap.add_argument("--shards", type=int, default=1024,
                    help="write in this many contiguous shards to bound peak memory")
    ap.add_argument("--max_train_docs", type=int, default=0,
                    help="slice raw train docs before tokenizing (0=all); for calibration runs "
                         "or producing a small pre-tokenized slice for fast smoke tests")
    ap.add_argument("--max_val_docs", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = GPT2TokenizerFast.from_pretrained(args.model_name)
    tok.model_max_length = int(1e12)  # silence per-doc "longer than model_max_length" spam;
                                       # we deliberately don't truncate, group_texts-equivalent
                                       # flattening handles arbitrary doc length downstream
    assert tok.vocab_size < 2 ** 16, f"vocab_size={tok.vocab_size} doesn't fit uint16"
    dtype = np.uint16

    ds = load_raw_docs(args.data_name, args.data_config, args.val_docs)
    if args.max_train_docs:
        ds["train"] = ds["train"].select(range(min(args.max_train_docs, len(ds["train"]))))
    if args.max_val_docs:
        ds["validation"] = ds["validation"].select(range(min(args.max_val_docs, len(ds["validation"]))))
    tokenize_fn = eos_tokenize_fn(tok)

    def add_len(examples):
        out = tokenize_fn(examples)
        out["len"] = [len(ids) for ids in out["input_ids"]]
        return out

    meta = {"dataset": args.data_name, "model_name": args.model_name,
            "vocab_size": tok.vocab_size, "dtype": "uint16", "eos_token_id": tok.eos_token_id}
    for split, out_name, meta_key in [("train", "train.bin", "train_tokens"),
                                       ("validation", "val.bin", "val_tokens")]:
        tokd = ds[split].map(add_len, batched=True, remove_columns=["text"],
                             num_proc=max(1, args.num_proc), desc=f"tokenize[{split}]")
        total = int(np.sum(tokd["len"], dtype=np.uint64))
        out_path = os.path.join(args.out_dir, out_name)
        arr = np.memmap(out_path, dtype=dtype, mode="w+", shape=(total,))

        idx = 0
        n_shards = min(args.shards, max(1, len(tokd)))
        report_every = max(1, n_shards // 20)
        for i in range(n_shards):
            shard = tokd.shard(num_shards=n_shards, index=i, contiguous=True).with_format("numpy")
            batch = np.concatenate(shard["input_ids"]).astype(dtype) if len(shard) else np.array([], dtype=dtype)
            arr[idx: idx + len(batch)] = batch
            idx += len(batch)
            if i % report_every == 0:
                print(f"  [{split}] shard {i}/{n_shards} ({idx:,}/{total:,} tokens)", flush=True)
        arr.flush()
        meta[meta_key] = total
        print(f"[done] {split}: {total:,} tokens -> {out_path} "
              f"({os.path.getsize(out_path) / 1e9:.2f} GB)", flush=True)

    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print(f"[done] wrote {args.out_dir}/meta.json: {meta}", flush=True)


if __name__ == "__main__":
    main()
