"""
Train a ViT with a vMF LayerNorm channel at a FIXED total rate budget.

Each LN read is modeled as vMF(mu, kappa) on the (D-2)-sphere; kappa is the rate
knob. A run fixes a total rate budget B (set by a uniform-equivalent sigma_g) and
learns the per-tap ALLOCATION of that budget via softmax rate-shares
(r_i = B * softmax(a_i), sum r_i = B). Because the budget is conserved in RATE,
there is no "compress everything / one heavy tap" collapse; CE alone drives a
water-filling allocation (low noise on reads that matter). No penalty term.

Sampling stays add-Gaussian-then-reLN with sigma matched to kappa (validated in
vmf_utils) -> reparam-clean gradients to both weights and the allocation, no
rejection sampler. Rate and (post-hoc) Bhattacharyya similarity are exact vMF.

Two arms (same data, IN100): --arm scratch (random init) | finetune (pretrained).
Everything needed for after-the-fact analysis is saved.
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
import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

import vmf_utils


def build_loaders(data_dir, batch_size, workers, val_subdir="val",
                  auto_augment="rand-m7-mstd0.5-inc1", rrc_scale=(0.4, 1.0),
                  re_prob=0.0, hflip=0.5):
    from torchvision import datasets
    cfg = dict(input_size=(3, 224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
               interpolation="bicubic", crop_pct=0.9)
    # Finetuning-appropriate augmentation: moderate RandAugment, RRC scale floored at 0.4 (no
    # extreme tiny crops), no random-erasing. The pretrained-from-scratch recipe's heavy aug just
    # slows a short adaptation and confounds the rate measurement. Keep FIXED across the whole
    # sweep so cross-sigma_g / cross-model bite-points stay comparable.
    train_tf = timm.data.create_transform(**cfg, is_training=True, auto_augment=auto_augment,
                                           scale=rrc_scale, hflip=hflip, re_prob=re_prob)
    eval_tf = timm.data.create_transform(**cfg, is_training=False)
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, val_subdir), transform=eval_tf)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                           num_workers=workers, pin_memory=True, drop_last=True,
                                           persistent_workers=workers > 0)
    val_ld = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                         num_workers=workers, pin_memory=True,
                                         persistent_workers=workers > 0)
    return train_ld, val_ld, len(train_ds.classes)


def lr_at(step, total_steps, warmup_steps, base_lr, min_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))



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


def build_model(model_name, num_classes, drop_path, controller, pretrained, weights_dir=""):
    # weights_dir set => load timm weights from a local file (offline/HPC) instead of the HF hub
    hub = pretrained and not weights_dir
    model = timm.create_model(model_name, pretrained=hub,
                              num_classes=num_classes, drop_path_rate=drop_path)
    if pretrained and weights_dir:
        sd = torch.load(os.path.join(weights_dir, f"{model_name}.pth"), map_location="cpu")
        model.load_state_dict(sd.get("model", sd), strict=False)  # head may differ (num_classes)
    idx = 0
    for name, module in model.named_modules():
        for cn, child in list(module.named_children()):
            if isinstance(child, nn.LayerNorm):
                setattr(module, cn, NoisyLayerNorm(child, controller, idx))
                idx += 1
    return model, idx


@torch.no_grad()
def evaluate(model, policy, controller, val_ld, device, amp_dtype, max_batches=None,
             teacher=None, clean=False):
    """Returns (accuracy, mean KL-from-teacher or None). KL is the distillation distortion."""
    model.eval()
    if clean:
        controller.enabled = False
    else:
        controller.sigma = policy.sigmas().detach()
        controller.enabled = True
    correct = tot = 0
    kl_sum = 0.0
    for bi, (x, y) in enumerate(val_ld):
        if max_batches and bi >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(x)
            if teacher is not None:
                t_logits = teacher(x)
        correct += (logits.argmax(1) == y).sum().item()
        tot += y.numel()
        if teacher is not None:
            p_t = F.softmax(t_logits.float(), dim=1)
            logp_s = F.log_softmax(logits.float(), dim=1)
            kl_sum += (p_t * (torch.log(p_t.clamp_min(1e-9)) - logp_s)).sum(1).sum().item()
    controller.enabled = False
    return correct / max(1, tot), (kl_sum / max(1, tot) if teacher is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="PATH/TO/DATA",
                    help="ImageFolder root containing train/ and the --val_subdir split")
    ap.add_argument("--val_subdir", default="val",
                    help="val subfolder under --data (e.g. val_split for full ImageNet)")
    ap.add_argument("--model", default="vit_small_patch16_224",
                    help="timm model, e.g. vit_small_patch16_224")
    ap.add_argument("--weights_dir", default="",
                    help="load pretrained timm weights from {weights_dir}/{model}.pth (offline/HPC) "
                         "instead of the HF hub; used for finetune init and --teacher_pretrained")
    ap.add_argument("--arm", choices=["scratch", "finetune"], required=True)
    ap.add_argument("--sigma_g", type=float, required=True,
                    help="uniform-equivalent noise level; sets the total rate budget B")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=384)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--policy_lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup_epochs", type=float, default=5)
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.8)
    # augmentation (finetune-appropriate defaults; logged per run). Keep fixed across the sweep.
    ap.add_argument("--auto_augment", default="rand-m7-mstd0.5-inc1",
                    help="timm RandAugment/AutoAugment policy (scratch recipe uses rand-m9)")
    ap.add_argument("--rrc_scale_min", type=float, default=0.4,
                    help="RandomResizedCrop lower scale bound (finetune 0.4; scratch recipe 0.08)")
    ap.add_argument("--re_prob", type=float, default=0.0, help="random-erasing prob (finetune default 0)")
    ap.add_argument("--clean", action="store_true", help="train a clean teacher (channel OFF, CE)")
    ap.add_argument("--distill", action="store_true", help="loss = KL from a frozen teacher")
    ap.add_argument("--teacher", default="", help="teacher ckpt (required with --distill)")
    ap.add_argument("--teacher_pretrained", action="store_true",
                    help="use the timm --model pretrained weights as the frozen distill teacher "
                         "(instead of a local --teacher ckpt)")
    ap.add_argument("--distill_T", type=float, default=1.0, help="distillation temperature")
    ap.add_argument("--noise_ramp_epochs", type=int, default=0,
                    help="ramp total rate budget from ramp_start_sigma_g down to target over N epochs")
    ap.add_argument("--ramp_start_sigma_g", type=float, default=0.05, help="near-clean start of the ramp")
    ap.add_argument("--ramp_shape", choices=["geom", "linsigma", "loglinsigma"], default="geom",
                    help="geom: linear in log-rate; linsigma: linear in sigma_g (front-loaded); "
                         "loglinsigma: linear in log(sigma_g) (keeps sigma low longest, steep finish)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit_train_batches", type=int, default=0)
    ap.add_argument("--eval_max_batches", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--save_every", type=int, default=0,
                    help="save a distinct ckpt_ep{N}.pt every N epochs (0=off; for sweep runs)")
    ap.add_argument("--probe_n", type=int, default=2048, help="fixed probe-set size to persist")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_resume", action="store_true",
                    help="ignore an existing ckpt_last.pt and start fresh (default: auto-resume)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    controller = NoiseController()
    train_ld, val_ld, num_classes = build_loaders(
        args.data, args.batch_size, args.workers, args.val_subdir,
        auto_augment=args.auto_augment, rrc_scale=(args.rrc_scale_min, 1.0), re_prob=args.re_prob)
    model, n_taps = build_model(args.model, num_classes, args.drop_path, controller,
                                pretrained=(args.arm == "finetune"), weights_dir=args.weights_dir)
    model.to(device)

    # vMF rate machinery is dimension-specific: derive p from the model (ViT-Small D=384).
    # sigma_g is the dimension-independent noise knob; the resulting rate/budget scales with D.
    D = model.embed_dim
    p = D - 1
    maps = vmf_utils.build_rate_sigma_maps(p)
    kappa_g = float(np.interp(args.sigma_g, maps["sigma"][::-1], maps["kappa"][::-1]))
    rate_g = float(vmf_utils.rate_kl(kappa_g, p))
    B = n_taps * rate_g

    def B_of(sg):
        kg = float(np.interp(sg, maps["sigma"][::-1], maps["kappa"][::-1]))
        return n_taps * float(vmf_utils.rate_kl(kg, p))
    B_start = B_of(args.ramp_start_sigma_g) if args.noise_ramp_epochs > 0 else B
    policy = RateBudgetPolicy(n_taps, B, maps).to(device)
    print(f"[setup] model={args.model} D={D} arm={args.arm} sigma_g={args.sigma_g} kappa_g={kappa_g:.1f} "
          f"rate/tap={rate_g:.2f} nats ({rate_g/math.log(2):.0f} bits) "
          f"B={B:.1f} nats ({B/math.log(2):.0f} bits total) "
          f"classes={num_classes} taps={n_taps} amp={amp_dtype}", flush=True)

    # persist fixed probe-set indices for reproducible post-hoc activations
    g = torch.Generator().manual_seed(12345)
    from torchvision import datasets
    val_len = len(datasets.ImageFolder(os.path.join(args.data, args.val_subdir)).samples)
    probe_idx = torch.randperm(val_len, generator=g)[:args.probe_n].tolist()
    json.dump(probe_idx, open(os.path.join(args.out, "probe_idx.json"), "w"))

    if args.mixup > 0 and not args.distill:
        mixup = Mixup(mixup_alpha=args.mixup, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
                      label_smoothing=args.label_smoothing, num_classes=num_classes)
        criterion = SoftTargetCrossEntropy()
    else:
        mixup, criterion = None, nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    teacher = None
    if args.distill:
        assert args.teacher or args.teacher_pretrained, \
            "--distill needs --teacher (local ckpt) or --teacher_pretrained (timm weights)"
        teacher = timm.create_model(args.model,
                                    pretrained=(args.teacher_pretrained and not args.weights_dir),
                                    num_classes=num_classes)
        if args.teacher_pretrained and args.weights_dir:
            wsd = torch.load(os.path.join(args.weights_dir, f"{args.model}.pth"), map_location="cpu")
            teacher.load_state_dict(wsd.get("model", wsd), strict=True)
        teacher = teacher.to(device).eval()
        if args.teacher_pretrained:
            tsd = teacher.state_dict()                    # timm pretrained IS the teacher
            tsrc = f"{args.model}:pretrained"
        else:
            tsd = torch.load(args.teacher, map_location="cpu")["model"]
            teacher.load_state_dict(tsd, strict=True)
            tsrc = args.teacher
        for pp in teacher.parameters():
            pp.requires_grad_(False)
        model.load_state_dict(tsd, strict=False)  # student starts AT the teacher
        print(f"[distill] teacher={tsrc} T={args.distill_T}; student init from teacher; "
              f"loss=KL(teacher||noisy-student)", flush=True)

    # Split by rank so that one-dimensional parameters -- LayerNorm gains/biases and every other
    # bias -- are exempt from weight decay, matching train_stoch_layernorm_gpt.py (and the usual
    # transformer convention). This matters more here than in a plain finetune: the LN gain is
    # applied immediately after the noisy channel, so decaying it pulls the channel's output
    # scale toward zero independently of any gradient signal. Runs before this change decayed
    # all of model.parameters() and are NOT resumable into this optimizer (different number of
    # param groups); start those fresh with --no_resume or a new --out.
    # The rank rule alone would still decay the few rank>=2 tensors timm's own recipe exempts
    # (pos_embed, cls_token) -- position/class embeddings have no GPT-2 analogue, so the LM
    # convention has nothing to say about them; defer to timm via model.no_weight_decay().
    nwd = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    decay, no_decay, named_exempt = [], [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or pname in nwd:
            no_decay.append(p)
            if p.ndim >= 2:
                named_exempt.append(pname)
        else:
            decay.append(p)
    opt = torch.optim.AdamW([
        {"params": decay, "lr": args.lr, "weight_decay": args.wd},
        {"params": no_decay, "lr": args.lr, "weight_decay": 0.0},
        {"params": policy.parameters(), "lr": args.policy_lr, "weight_decay": 0.0},
    ], betas=(0.9, 0.999))
    print(f"[optim] weight decay {args.wd} on {len(decay)} tensors "
          f"({sum(p.numel() for p in decay):,} params); exempt: {len(no_decay)} tensors "
          f"({sum(p.numel() for p in no_decay):,} params), of which rank>=2 by "
          f"no_weight_decay(): {sorted(named_exempt) or 'none'}", flush=True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    steps_per_epoch = args.limit_train_batches or len(train_ld)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    noise_ramp_steps = args.noise_ramp_epochs * steps_per_epoch

    def B_at_step(step):
        """Continuous per-step interpolation of the rate budget (not per-epoch -- a coarse
        epoch-level staircase is itself a milder version of the sudden-change problem the
        ramp is meant to avoid). Mirrors train_stoch_layernorm_gpt.py."""
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
        if ckpt.get("optimizer") is not None:
            opt.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        step = ckpt.get("step", 0)
        start_epoch = ckpt["epoch"] + 1
        hist_path = os.path.join(args.out, "history.json")
        if os.path.exists(hist_path):
            history = json.load(open(hist_path))["history"]
        # steps_per_epoch/total_steps/warmup_steps must match the original run for the LR
        # schedule (and the ramp) to stay continuous across the resume boundary.
        sched_keys = ["batch_size", "epochs", "limit_train_batches", "warmup_epochs",
                      "lr", "min_lr", "noise_ramp_epochs", "sigma_g"]
        changed = {k: (ckpt["args"].get(k), getattr(args, k)) for k in sched_keys
                   if ckpt.get("args", {}).get(k) != getattr(args, k)}
        if changed:
            print(f"[resume] WARNING: args differ from checkpointed run (old->new): {changed} "
                  f"-- LR/ramp schedule will not be a clean continuation unless intended", flush=True)
        print(f"[resume] loaded {ckpt_path}: resuming at epoch {start_epoch}, step {step}", flush=True)

    torch.cuda.reset_peak_memory_stats(device)  # measure training/eval steps only, not setup/load
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, seen, run_ce = time.time(), 0, 0.0
        for bi, (x, y) in enumerate(train_ld):
            if args.limit_train_batches and bi >= args.limit_train_batches:
                break
            if args.noise_ramp_epochs > 0:
                policy.B.fill_(B_at_step(step))        # continuous ramp, updated every step
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if mixup is not None:
                x, y = mixup(x, y)
            if args.clean:
                controller.enabled = False                 # teacher: channel off
            else:
                controller.sigma = policy.sigmas()          # differentiable, updates as `a` trains
                controller.enabled = True

            lr = lr_at(step, total_steps, warmup_steps, args.lr, args.min_lr)
            opt.param_groups[0]["lr"] = lr  # decay group
            opt.param_groups[1]["lr"] = lr  # no_decay group (policy group keeps its own const lr)
            opt.zero_grad(set_to_none=True)
            if args.distill:
                with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
                    t_logits = teacher(x)
                with torch.autocast("cuda", dtype=amp_dtype):
                    s_logits = model(x)
                T = args.distill_T
                p_t = F.softmax(t_logits.float() / T, dim=1)
                logp_s = F.log_softmax(s_logits.float() / T, dim=1)
                loss = (p_t * (torch.log(p_t.clamp_min(1e-9)) - logp_s)).sum(1).mean() * (T * T)
            else:
                with torch.autocast("cuda", dtype=amp_dtype):
                    loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(policy.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()
            run_ce += loss.item() * x.shape[0]
            seen += x.shape[0]
            step += 1
            if bi % 100 == 0:
                print(f"  e{epoch} b{bi}/{steps_per_epoch} ce={loss.item():.3f} lr={lr:.2e}", flush=True)

        ips = seen / (time.time() - t0)
        # instantaneous uniform-equivalent sigma_g at the current (end-of-epoch) budget
        cur_sg = float(np.interp(float(policy.B) / n_taps, maps["rate"], maps["sigma"]))
        with torch.no_grad():
            rates = policy.rates().cpu().tolist()
            kappas = policy.kappas().cpu().tolist()
            sigmas = policy.sigmas().cpu().tolist()
        mem_alloc = torch.cuda.max_memory_allocated(device) / 1e9
        mem_reserved = torch.cuda.max_memory_reserved(device) / 1e9
        # rate_kl is the TOTAL rate of a read (over all DOF), so total bits = sum(rates)/ln2
        rec = {"epoch": epoch, "ce": run_ce / seen, "img_per_s": ips,
               "sigma_g_t": cur_sg, "B_t": float(policy.B),
               "rates_nats": rates, "kappas": kappas, "sigmas": sigmas,
               "total_bits": float(sum(rates) / math.log(2)),
               "gpu_mem_alloc_gb": mem_alloc, "gpu_mem_reserved_gb": mem_reserved}
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            acc, kl_t = evaluate(model, policy, controller, val_ld, device, amp_dtype,
                                 args.eval_max_batches or None, teacher=teacher, clean=args.clean)
            rec["val_acc"] = acc
            rec["val_kl_teacher"] = kl_t
            klstr = f" klT={kl_t:.4f}" if kl_t is not None else ""
            print(f"[epoch {epoch}] loss={rec['ce']:.3f} {ips:.0f}img/s acc={acc:.3f}{klstr} "
                  f"| rate spread [{min(rates):.2f},{max(rates):.2f}] nats "
                  f"| gpu_mem peak alloc/reserved={mem_alloc:.2f}/{mem_reserved:.2f}GB", flush=True)
            torch.save({"model": model.state_dict(), "policy": policy.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                        "step": step, "epoch": epoch, "args": vars(args), "B": B,
                        "rate_g": rate_g, "kappa_g": kappa_g},
                       os.path.join(args.out, "ckpt_last.pt"))
        else:
            print(f"[epoch {epoch}] ce={rec['ce']:.3f} {ips:.0f}img/s "
                  f"| gpu_mem peak alloc/reserved={mem_alloc:.2f}/{mem_reserved:.2f}GB", flush=True)
        if args.save_every and (epoch % args.save_every == 0 or epoch == args.epochs - 1):
            torch.save({"model": model.state_dict(), "policy": policy.state_dict(),
                        "epoch": epoch, "sigma_g_t": cur_sg, "B_t": float(policy.B)},
                       os.path.join(args.out, f"ckpt_ep{epoch}.pt"))
        history.append(rec)
        json.dump({"args": vars(args), "B": B, "history": history},
                  open(os.path.join(args.out, "history.json"), "w"), indent=2)

    torch.save({"model": model.state_dict(), "policy": policy.state_dict(),
                "args": vars(args), "B": B, "rate_g": rate_g, "kappa_g": kappa_g},
               os.path.join(args.out, "ckpt.pt"))
    print(f"[done] saved {args.out}/ckpt.pt", flush=True)


if __name__ == "__main__":
    main()
