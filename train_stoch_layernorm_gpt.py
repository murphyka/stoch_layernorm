"""
Train GPT-2 (small) with a vMF LayerNorm channel at a FIXED total rate budget.

Direct adaptation of train_stoch_layernorm_vit.py (ViT/ImageNet) to a causal-LM setting. The
noisy-LN / rate-budget machinery (NoisyLayerNorm, RateBudgetPolicy, the vMF
rate<->sigma maps) is architecture-agnostic -- it just wraps every nn.LayerNorm
in the model -- and is duplicated here character for character rather than imported,
so the two trainers stay independently runnable; keep them in sync when editing
either. HF's GPT2 exposes
ln_1/ln_2 per block plus a final ln_f, all plain nn.LayerNorm, so the same
named_modules() sweep finds them (12-layer gpt2-small -> 25 taps, the same
count as the 12-block ViT, purely coincidental).

What's actually different from the vision script:
  - data: causal LM on token blocks (wikitext-103 by default) instead of
    image classification: build_lm_loaders tokenizes+groups text into
    fixed-length non-overlapping blocks (no padding / no attention_mask needed).
  - loss: manual shifted cross-entropy (and shifted KL for --distill) instead
    of mixup/cutmix + soft-target CE.
  - eval: perplexity + next-token top-1 accuracy instead of image accuracy.
  - no separate "clean teacher" training step is needed for --distill: the
    pretrained checkpoint IS the clean reference (no aug/mixup shift to worry
    about the way ViT's mixup training does), so the default teacher is just
    a frozen copy of the pretrained model unless --teacher points elsewhere.
  - n_taps is measured by building the model FIRST, then the rate budget B is
    set from the measured tap count, so the same code path works across GPT-2
    sizes without a hardcoded tap count.

Two arms (same data): --arm scratch (random init, HF config only) | finetune
(pretrained gpt2 weights).
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

import vmf_utils

def interp1d(x, xp, fp):
    """Linear interp, xp strictly increasing; differentiable in x."""
    x = x.clamp(xp[0], xp[-1])
    idx = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


class NoiseController:
    def __init__(self):
        self.sigma = None      # [n_taps]
        self.enabled = False


class NoisyLayerNorm(nn.Module):
    def __init__(self, ln, controller, idx):
        super().__init__()
        self.eps = ln.eps
        self.weight = ln.weight
        self.bias = ln.bias
        self.controller = controller
        self.idx = idx

    def _normalize(self, x):
        m = x.mean(-1, keepdim=True)
        v = x.var(-1, unbiased=False, keepdim=True)
        return (x - m) / torch.sqrt(v + self.eps)

    def forward(self, x):
        xhat = self._normalize(x)
        c = self.controller
        if c.enabled and c.sigma is not None:
            s = c.sigma[self.idx]
            xhat = self._normalize(xhat + s * torch.randn_like(xhat))
        return xhat * self.weight + self.bias


class RateBudgetPolicy(nn.Module):
    """Learned per-tap allocation of a fixed total rate budget B (conserved in rate)."""

    def __init__(self, n_taps, B, maps):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(n_taps))       # logits, init uniform
        self.register_buffer("B", torch.tensor(float(B)))
        self.register_buffer("rate_grid", torch.tensor(maps["rate"], dtype=torch.float32))
        self.register_buffer("sigma_grid", torch.tensor(maps["sigma"], dtype=torch.float32))
        self.register_buffer("kappa_grid", torch.tensor(maps["kappa"], dtype=torch.float32))

    def rates(self):
        return self.B * torch.softmax(self.a, 0)          # [n], sum = B

    def sigmas(self):
        return interp1d(self.rates(), self.rate_grid, self.sigma_grid).clamp(min=1e-4)

    def kappas(self):
        return interp1d(self.rates(), self.rate_grid, self.kappa_grid)

def lr_at(step, total_steps, warmup_steps, base_lr, min_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))

def safe_kappa_lo(p, floor=1e-250):
    """vmf_utils.build_rate_sigma_maps defaults kappa_lo=5, which is safe at the ViT
    widths used here (p=383). GPT-2's LN width is p=767: the Bessel order nu=p/2-1 is
    twice as big, so ive(nu, 5) underflows to a hard 0.0 (not just small) and rho
    becomes 0/0. Search for the smallest kappa where ive(nu, kappa) clears a safety
    floor well above underflow, scaled to this p rather than hardcoded."""
    from scipy.special import ive
    nu = p / 2.0 - 1.0
    lo, hi = 1.0, max(50.0, p)
    while ive(nu, hi) < floor:
        hi *= 2
    for _ in range(50):
        mid = (lo + hi) / 2
        if ive(nu, mid) < floor:
            lo = mid
        else:
            hi = mid
    return hi


def _local_parquet_files(local_dir):
    """wikitext-style layout has train-*/validation-*/test-* prefixes; openwebtext's
    parquet-converted layout is flat (0000.parquet..0079.parquet, all train) since the
    files came straight out of the hub's plain_text/train/ dir -- treat an unprefixed
    flat directory as all-train."""
    import glob
    train_files = sorted(glob.glob(os.path.join(local_dir, "train-*.parquet")))
    val_files = sorted(glob.glob(os.path.join(local_dir, "validation-*.parquet")))
    if not train_files:
        train_files = sorted(glob.glob(os.path.join(local_dir, "*.parquet")))
    assert train_files, f"no *.parquet found in {local_dir}"
    data_files = {"train": train_files}
    if val_files:
        data_files["validation"] = val_files
    return data_files


def load_raw_docs(data_name, data_config, val_docs=5000):
    """Returns a DatasetDict of raw {'text': ...} docs with 'train'/'validation' splits,
    resolving local-parquet-dir vs HF-hub-id and carving a held-out tail slice for
    validation if the source has none natively (e.g. openwebtext ships train-only).
    Shared by build_lm_loaders (parquet path) and prepare_lm_data.py (pre-tokenize path)
    so both produce an IDENTICAL train/validation doc split for a given dataset."""
    import datasets as hfds

    if os.path.isdir(data_name):
        data_files = _local_parquet_files(data_name)
        ds = hfds.load_dataset("parquet", data_files=data_files)
        print(f"[data] local parquet from {data_name}: {len(data_files['train'])} train file(s)"
              + (f", {len(data_files['validation'])} validation file(s)" if "validation" in data_files else ""),
              flush=True)
    else:
        ds = hfds.load_dataset(data_name, data_config)
    if "validation" not in ds:
        # e.g. openwebtext ships train-only (8M docs); carve a held-out tail slice
        # off (cheap contiguous .select, no shuffle needed -- scrape order is
        # already arbitrary) rather than shuffling the full 8M-row index.
        n = len(ds["train"])
        n_val = min(val_docs, n // 20)
        ds["validation"] = ds["train"].select(range(n - n_val, n))
        ds["train"] = ds["train"].select(range(n - n_val))
        print(f"[data] {data_name} has no validation split; held out "
              f"last {n_val} of {n} train docs", flush=True)
    return ds


def eos_tokenize_fn(tok):
    """append EOS after every doc so packed blocks carry GPT-2's own
    '<|endoftext|> = context reset' convention at doc boundaries instead of
    silently splicing unrelated documents together. Shared by build_lm_loaders
    (applied before group_texts) and prepare_lm_data.py (applied before flattening
    to the token-id memmap), so both produce an IDENTICAL token stream."""
    def _fn(examples):
        out = tok(examples["text"], return_attention_mask=False)
        out["input_ids"] = [ids + [tok.eos_token_id] for ids in out["input_ids"]]
        return out
    return _fn


def pretokenized_paths(data_name):
    """If data_name is a dir produced by prepare_lm_data.py (train.bin/val.bin/meta.json),
    return their paths; else None (caller falls back to the parquet+datasets.map() path)."""
    if not os.path.isdir(data_name):
        return None
    train_bin = os.path.join(data_name, "train.bin")
    val_bin = os.path.join(data_name, "val.bin")
    meta = os.path.join(data_name, "meta.json")
    if os.path.exists(train_bin) and os.path.exists(val_bin) and os.path.exists(meta):
        return {"train": train_bin, "val": val_bin, "meta": meta}
    return None


class MemmapBlockDataset(torch.utils.data.Dataset):
    """Fixed block_size windows over a flat token-id memmap. Reopened fresh per
    __getitem__ (cheap -- mmap is page-cache backed) rather than held open across
    the DataLoader worker fork, matching nanoGPT's convention to avoid a known
    memmap + multiprocessing memory-growth issue."""

    def __init__(self, path, block_size, dtype):
        self.path, self.dtype, self.block_size = path, dtype, block_size
        self.n_blocks = len(np.memmap(path, dtype=dtype, mode="r")) // block_size

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, i):
        data = np.memmap(self.path, dtype=self.dtype, mode="r")
        chunk = data[i * self.block_size:(i + 1) * self.block_size]
        return {"input_ids": torch.from_numpy(chunk.astype(np.int64))}


def build_lm_loaders_pretokenized(paths, block_size, batch_size, workers):
    meta = json.load(open(paths["meta"]))
    dtype = np.dtype(meta["dtype"])
    print(f"[data] pre-tokenized bin from {os.path.dirname(paths['train'])}: "
          f"{meta['train_tokens']:,} train / {meta['val_tokens']:,} val tokens, "
          f"vocab={meta['vocab_size']} (no tokenizer/hub access needed)", flush=True)
    train_ds = MemmapBlockDataset(paths["train"], block_size, dtype)
    val_ds = MemmapBlockDataset(paths["val"], block_size, dtype)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                           num_workers=workers, pin_memory=True, drop_last=True,
                                           persistent_workers=workers > 0)
    val_ld = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                         num_workers=workers, pin_memory=True,
                                         persistent_workers=workers > 0)
    return train_ld, val_ld, meta["vocab_size"], len(val_ds)


def build_lm_loaders(data_name, data_config, model_name, block_size, batch_size, workers,
                      max_train_docs=0, max_val_docs=0, val_docs=5000):
    pretok = pretokenized_paths(data_name)
    if pretok is not None:
        if max_train_docs or max_val_docs:
            print("[data] --max_train_docs/--max_val_docs are ignored for pre-tokenized data "
                  "(doc-level slicing doesn't apply post-flattening); use --limit_train_batches "
                  "/--eval_max_batches instead for smoke tests", flush=True)
        return build_lm_loaders_pretokenized(pretok, block_size, batch_size, workers)

    tok = GPT2TokenizerFast.from_pretrained(model_name)
    ds = load_raw_docs(data_name, data_config, val_docs)
    if max_train_docs:
        ds["train"] = ds["train"].select(range(min(max_train_docs, len(ds["train"]))))
    if max_val_docs:
        ds["validation"] = ds["validation"].select(range(min(max_val_docs, len(ds["validation"]))))

    tokenize_fn = eos_tokenize_fn(tok)

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {k: [t[i:i + block_size] for i in range(0, total_len, block_size)]
                  for k, t in concatenated.items()}
        return result

    proc = {}
    for split in ("train", "validation"):
        toks = ds[split].map(tokenize_fn, batched=True, remove_columns=["text"],
                              num_proc=max(1, workers), desc=f"tokenize[{split}]")
        blocks = toks.map(group_texts, batched=True, num_proc=max(1, workers),
                          desc=f"group[{split}]")
        blocks.set_format(type="torch", columns=["input_ids"])
        proc[split] = blocks

    train_ld = torch.utils.data.DataLoader(proc["train"], batch_size=batch_size, shuffle=True,
                                           num_workers=workers, pin_memory=True, drop_last=True,
                                           persistent_workers=workers > 0)
    val_ld = torch.utils.data.DataLoader(proc["validation"], batch_size=batch_size, shuffle=False,
                                         num_workers=workers, pin_memory=True,
                                         persistent_workers=workers > 0)
    return train_ld, val_ld, tok.vocab_size, len(proc["validation"])


def build_model(controller, pretrained, model_name, dropout_overrides=None):
    dropout_overrides = dropout_overrides or {}
    if pretrained:
        model = GPT2LMHeadModel.from_pretrained(model_name, **dropout_overrides)
    else:
        model = GPT2LMHeadModel(GPT2Config.from_pretrained(model_name, **dropout_overrides))
    idx = 0
    for name, module in model.named_modules():
        for cn, child in list(module.named_children()):
            if isinstance(child, nn.LayerNorm):
                setattr(module, cn, NoisyLayerNorm(child, controller, idx))
                idx += 1
    return model, idx


def shifted_ce(logits, input_ids, label_smoothing=0.0):
    logits = logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                           label_smoothing=label_smoothing)


def shifted_kl(student_logits, teacher_logits, T, chunk_size=128):
    """KL(teacher||student) over next-token distributions. Chunked along the sequence
    dim: p_t/logp_s are full [B, L, V] fp32 tensors (vocab=50257), which at e.g.
    batch=32/block=512 is ~3.3GB EACH (~6.6GB combined) -- large enough to be the
    difference between fitting a MIG partition and OOMing. Chunking bounds that to
    [B, chunk_size, V] at a time with identical fp32 precision and an identical
    result (same sum/mean, just accumulated in pieces), not a bf16-precision tradeoff."""
    s = student_logits[:, :-1, :] / T
    t = teacher_logits[:, :-1, :] / T
    L = s.shape[1]
    kl_sum = s.new_zeros(())
    for i in range(0, L, chunk_size):
        sc = s[:, i:i + chunk_size, :].contiguous().float()
        tc = t[:, i:i + chunk_size, :].contiguous().float()
        p_t = F.softmax(tc, dim=-1)
        logp_s = F.log_softmax(sc, dim=-1)
        kl_sum = kl_sum + (p_t * (torch.log(p_t.clamp_min(1e-9)) - logp_s)).sum()
    n_tok = student_logits.shape[0] * L
    return (kl_sum / n_tok) * (T * T)


@torch.no_grad()
def evaluate(model, policy, controller, val_ld, device, amp_dtype, max_batches=None,
             teacher=None, clean=False):
    """Returns dict(loss, ppl, token_acc, kl_teacher)."""
    model.eval()
    if clean:
        controller.enabled = False
    else:
        controller.sigma = policy.sigmas().detach()
        controller.enabled = True
    loss_sum = kl_sum = correct = tot_tok = 0.0
    for bi, batch in enumerate(val_ld):
        if max_batches and bi >= max_batches:
            break
        x = batch["input_ids"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(input_ids=x).logits
            if teacher is not None:
                t_logits = teacher(input_ids=x).logits
        n_tok = x.shape[0] * (x.shape[1] - 1)
        loss_sum += shifted_ce(logits, x).item() * n_tok
        pred = logits[:, :-1, :].argmax(-1)
        correct += (pred == x[:, 1:]).sum().item()
        tot_tok += n_tok
        if teacher is not None:
            kl_sum += shifted_kl(logits, t_logits, T=1.0).item() * n_tok
    controller.enabled = False
    loss = loss_sum / max(1, tot_tok)
    return {"loss": loss, "ppl": math.exp(min(loss, 20.0)),
            "token_acc": correct / max(1, tot_tok),
            "kl_teacher": (kl_sum / tot_tok) if teacher is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_name", default="PATH/TO/DATA",
                    help="local dir of *.parquet, or a dir of pre-tokenized .bin files from "
                         "prepare_lm_data.py "
                         "or an HF hub dataset id (falls back to hub download if not a local dir)")
    ap.add_argument("--data_config", default="wikitext-103-raw-v1",
                    help="HF hub config name; unused when --data_name is a local dir")
    ap.add_argument("--model_name", default="gpt2", help="HF model id (gpt2 = small, 124M)")
    ap.add_argument("--arm", choices=["scratch", "finetune"], required=True)
    ap.add_argument("--sigma_g", type=float, required=True,
                    help="uniform-equivalent noise level; sets the total rate budget B")
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--policy_lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup_epochs", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--resid_pdrop", type=float, default=None,
                    help="override GPT2Config resid_pdrop (default: leave at pretrained 0.1). "
                         "Dropout is an uncontrolled second noise source layered under the "
                         "vMF LN-channel noise during training (eval is unaffected, model.eval() "
                         "disables it); set to 0.0 for a dropout-free run if that confound matters.")
    ap.add_argument("--embd_pdrop", type=float, default=None, help="override GPT2Config embd_pdrop")
    ap.add_argument("--attn_pdrop", type=float, default=None, help="override GPT2Config attn_pdrop")
    ap.add_argument("--clean", action="store_true", help="train a clean teacher (channel OFF, CE)")
    ap.add_argument("--distill", action="store_true", help="loss = KL from a frozen teacher")
    ap.add_argument("--teacher", default="",
                    help="teacher ckpt saved by this script; default (empty) = frozen pretrained "
                         "--model_name, which is a valid clean reference since there's no aug/mixup shift")
    ap.add_argument("--distill_T", type=float, default=1.0)
    ap.add_argument("--noise_ramp_epochs", type=int, default=0)
    ap.add_argument("--ramp_start_sigma_g", type=float, default=0.05)
    ap.add_argument("--ramp_shape", choices=["geom", "linsigma", "loglinsigma"], default="geom",
                    help="geom: linear in log-rate; linsigma: linear in sigma_g (front-loaded); "
                         "loglinsigma: linear in log(sigma_g) (keeps sigma low longest, steep finish)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_train_docs", type=int, default=0, help="slice raw train docs before tokenizing (0=all); use for fast smoke tests")
    ap.add_argument("--max_val_docs", type=int, default=0)
    ap.add_argument("--limit_train_batches", type=int, default=0)
    ap.add_argument("--eval_max_batches", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--probe_n", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_resume", action="store_true",
                    help="ignore an existing ckpt_last.pt in --out and start fresh (default: "
                         "auto-resume if one is found -- safer default for SLURM preemption/"
                         "requeue, where forgetting an opt-in --resume flag would silently "
                         "waste the whole allocation restarting from scratch)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    controller = NoiseController()
    train_ld, val_ld, vocab_size, val_len = build_lm_loaders(
        args.data_name, args.data_config, args.model_name, args.block_size, args.batch_size,
        args.workers, args.max_train_docs, args.max_val_docs)

    dropout_overrides = {k: v for k, v in {
        "resid_pdrop": args.resid_pdrop, "embd_pdrop": args.embd_pdrop, "attn_pdrop": args.attn_pdrop,
    }.items() if v is not None}
    model, n_taps = build_model(controller, pretrained=(args.arm == "finetune"),
                                model_name=args.model_name, dropout_overrides=dropout_overrides)
    assert args.block_size <= model.config.n_positions, \
        f"--block_size {args.block_size} exceeds {args.model_name}'s n_positions={model.config.n_positions}"
    model.to(device)
    D = model.config.n_embd
    p = D - 1
    maps = vmf_utils.build_rate_sigma_maps(p, kappa_lo=safe_kappa_lo(p))
    kappa_g = float(np.interp(args.sigma_g, maps["sigma"][::-1], maps["kappa"][::-1]))
    rate_g = float(vmf_utils.rate_kl(kappa_g, p))
    B = n_taps * rate_g

    def B_of(sg):
        kg = float(np.interp(sg, maps["sigma"][::-1], maps["kappa"][::-1]))
        return n_taps * float(vmf_utils.rate_kl(kg, p))
    B_start = B_of(args.ramp_start_sigma_g) if args.noise_ramp_epochs > 0 else B

    policy = RateBudgetPolicy(n_taps, B, maps).to(device)
    print(f"[setup] model={args.model_name} arm={args.arm} D={D} p={p} sigma_g={args.sigma_g} "
          f"kappa_g={kappa_g:.1f} rate/tap={rate_g:.2f} nats ({rate_g/math.log(2):.0f} bits) "
          f"B={B:.1f} nats ({B/math.log(2):.0f} bits total) vocab={vocab_size} taps={n_taps} "
          f"amp={amp_dtype} pdrop(resid/embd/attn)="
          f"{model.config.resid_pdrop}/{model.config.embd_pdrop}/{model.config.attn_pdrop}", flush=True)

    g = torch.Generator().manual_seed(12345)
    probe_idx = torch.randperm(val_len, generator=g)[:min(args.probe_n, val_len)].tolist()
    json.dump(probe_idx, open(os.path.join(args.out, "probe_idx.json"), "w"))

    teacher = None
    if args.distill:
        teacher = GPT2LMHeadModel.from_pretrained(args.model_name).to(device).eval()
        if args.teacher:
            tsd = torch.load(args.teacher, map_location="cpu")["model"]
            teacher.load_state_dict(tsd, strict=True)
            model.load_state_dict(tsd, strict=False)  # student starts AT the teacher
            print(f"[distill] teacher={args.teacher} T={args.distill_T}", flush=True)
        else:
            print(f"[distill] teacher=pretrained:{args.model_name} (frozen) T={args.distill_T}",
                  flush=True)
        for pp in teacher.parameters():
            pp.requires_grad_(False)

    # standard transformer LM convention (GPT-3/nanoGPT/HF Trainer): don't decay
    # biases or 1-D params. That includes every LayerNorm gain/bias -- exactly the
    # parameters this study cares most about -- so applying uniform wd there (as a
    # naive single `model.parameters()` group would) fights the channel we're
    # trying to measure. 2-D+ params (attn/mlp weight matrices, embeddings) decay.
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    opt = torch.optim.AdamW([
        {"params": decay, "lr": args.lr, "weight_decay": args.wd},
        {"params": no_decay, "lr": args.lr, "weight_decay": 0.0},
        {"params": policy.parameters(), "lr": args.policy_lr, "weight_decay": 0.0},
    ], betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    steps_per_epoch = args.limit_train_batches or len(train_ld)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    noise_ramp_steps = args.noise_ramp_epochs * steps_per_epoch

    def B_at_step(step):
        """Continuous per-step interpolation (not per-epoch -- a coarser epoch-level
        staircase is itself a milder version of the sudden-change problem ramping is
        meant to avoid)."""
        frac = min(1.0, step / noise_ramp_steps)
        if args.ramp_shape == "linsigma":
            sg_t = args.ramp_start_sigma_g + frac * (args.sigma_g - args.ramp_start_sigma_g)
            return B_of(sg_t)                          # even (front-loaded) noise arrival
        elif args.ramp_shape == "loglinsigma":
            sg_t = math.exp(math.log(args.ramp_start_sigma_g)
                            + frac * (math.log(args.sigma_g) - math.log(args.ramp_start_sigma_g)))
            return B_of(sg_t)                          # keep sigma low longest, steep finish
        return math.exp(math.log(B_start) + frac * (math.log(B) - math.log(B_start)))

    history, step, start_epoch = [], 0, 0
    ckpt_path = os.path.join(args.out, "ckpt_last.pt")
    if os.path.exists(ckpt_path) and not args.no_resume:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        policy.load_state_dict(ckpt["policy"])
        opt.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        step = ckpt["step"]
        start_epoch = ckpt["epoch"] + 1
        hist_path = os.path.join(args.out, "history.json")
        if os.path.exists(hist_path):
            history = json.load(open(hist_path))["history"]
        # steps_per_epoch/total_steps/warmup_steps must match the original run for the LR
        # schedule to stay continuous across the resume boundary -- warn (not hard-fail,
        # you may genuinely want to extend --epochs) if the args that determine them changed.
        sched_keys = ["batch_size", "block_size", "epochs", "limit_train_batches",
                      "warmup_epochs", "lr", "min_lr"]
        changed = {k: (ckpt["args"].get(k), getattr(args, k)) for k in sched_keys
                  if ckpt["args"].get(k) != getattr(args, k)}
        if changed:
            print(f"[resume] WARNING: these args differ from the checkpointed run "
                  f"(old -> new): {changed} -- LR schedule will NOT be a continuation "
                  f"of the original one unless that's intentional", flush=True)
        print(f"[resume] loaded {ckpt_path}: resuming at epoch {start_epoch}, step {step} "
              f"(checkpoint was epoch {ckpt['epoch']})", flush=True)

    torch.cuda.reset_peak_memory_stats(device)  # measure training/eval steps only, not setup/load
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, seen, run_loss = time.time(), 0, 0.0
        for bi, batch in enumerate(train_ld):
            if args.limit_train_batches and bi >= args.limit_train_batches:
                break
            if args.noise_ramp_epochs > 0:
                policy.B.fill_(B_at_step(step))
            x = batch["input_ids"].to(device, non_blocking=True)
            if args.clean:
                controller.enabled = False
            else:
                controller.sigma = policy.sigmas()
                controller.enabled = True

            lr = lr_at(step, total_steps, warmup_steps, args.lr, args.min_lr)
            opt.param_groups[0]["lr"] = lr  # decay group
            opt.param_groups[1]["lr"] = lr  # no_decay group (policy group keeps its own const lr)
            opt.zero_grad(set_to_none=True)
            if args.distill:
                with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
                    t_logits = teacher(input_ids=x).logits
                with torch.autocast("cuda", dtype=amp_dtype):
                    s_logits = model(input_ids=x).logits
                loss = shifted_kl(s_logits, t_logits, args.distill_T)
            else:
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = model(input_ids=x).logits
                loss = shifted_ce(logits, x, args.label_smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(policy.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * x.shape[0]
            seen += x.shape[0]
            step += 1
            if bi % 100 == 0:
                print(f"  e{epoch} b{bi}/{steps_per_epoch} loss={loss.item():.3f} lr={lr:.2e}",
                      flush=True)

        ips = seen / (time.time() - t0)
        with torch.no_grad():
            rates = policy.rates().cpu().tolist()
            kappas = policy.kappas().cpu().tolist()
            sigmas = policy.sigmas().cpu().tolist()
        rec = {"epoch": epoch, "loss": run_loss / seen, "seq_per_s": ips,
               "rates_nats": rates, "kappas": kappas, "sigmas": sigmas,
               "total_bits": float(sum(rates) / math.log(2))}
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            ev = evaluate(model, policy, controller, val_ld, device, amp_dtype,
                         args.eval_max_batches or None, teacher=teacher, clean=args.clean)
            mem_alloc = torch.cuda.max_memory_allocated(device) / 1e9
            mem_reserved = torch.cuda.max_memory_reserved(device) / 1e9
            rec.update({"val_loss": ev["loss"], "val_ppl": ev["ppl"],
                       "val_token_acc": ev["token_acc"], "val_kl_teacher": ev["kl_teacher"],
                       "gpu_mem_alloc_gb": mem_alloc, "gpu_mem_reserved_gb": mem_reserved})
            klstr = f" klT={ev['kl_teacher']:.4f}" if ev["kl_teacher"] is not None else ""
            print(f"[epoch {epoch}] loss={rec['loss']:.3f} {ips:.0f}seq/s "
                  f"val_ppl={ev['ppl']:.2f} tok_acc={ev['token_acc']:.3f}{klstr} "
                  f"| rate spread [{min(rates):.2f},{max(rates):.2f}] nats "
                  f"| gpu_mem peak alloc/reserved={mem_alloc:.2f}/{mem_reserved:.2f}GB", flush=True)
            torch.save({"model": model.state_dict(), "policy": policy.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scaler": scaler.state_dict() if amp_dtype == torch.float16 else None,
                        "step": step, "epoch": epoch, "args": vars(args), "B": B,
                        "rate_g": rate_g, "kappa_g": kappa_g}, ckpt_path)
        else:
            print(f"[epoch {epoch}] loss={rec['loss']:.3f} {ips:.0f}seq/s", flush=True)
        history.append(rec)
        json.dump({"args": vars(args), "B": B, "history": history},
                  open(os.path.join(args.out, "history.json"), "w"), indent=2)

    torch.save({"model": model.state_dict(), "policy": policy.state_dict(),
                "args": vars(args), "B": B, "rate_g": rate_g, "kappa_g": kappa_g},
               os.path.join(args.out, "ckpt.pt"))
    print(f"[done] saved {args.out}/ckpt.pt", flush=True)


if __name__ == "__main__":
    main()
