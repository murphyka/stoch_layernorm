"""What fraction of a tap's log-BC does an UNINFORMATIVE 64-dim read preserve?

Every retention number in this project is reported without a reference: r = 0.10 has been
treated as "low" and r = 0.46 as "high" on intuition. The dimension ratio 64/767 = 0.083 was
floated as a guess at the baseline but never computed. This computes it.

Method: take the SAME CRN-conditioned mu pair the real scan used at a given tap/position, and
push it through RANDOM 64-dimensional projections drawn uniformly (Haar) instead of through
the learned W_K/W_Q/W_V. Everything else -- kappa, D, the exact pushforward density, the
bridge importance sampler, n_mc -- is identical, so the only thing that changes is which
subspace is read.

A random projection is the right null here because it holds the ARITHMETIC of the measurement
fixed (same dimension, same concentration, same estimator) while removing any alignment
between the subspace and the direction in which the two posteriors actually differ. It does
NOT model "a head that does nothing useful" -- a real head's subspace is not Haar-random, so
this is a floor for subspace alignment, not a null model of head behaviour.
"""
import argparse
import itertools

import numpy as np
import torch
from scipy.special import logsumexp
from transformers import GPT2TokenizerFast

import bhat_qkv as bq
import vmf_utils
from qkv_bc_gpt import (MODEL_DIR, crn_local_mu_pair, load_run, shared_positions,
                        target_pos)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--battery", default="pair_battery_distance")
    ap.add_argument("--offset", type=int, default=1, help="probe this many tokens after the swap")
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--n_rand", type=int, default=12, help="random subspaces per (frame, layer)")
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--n_mc", type=int, default=1500)
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, controller, policy, depth_order, idx_of, kappas = load_run(args.ckpt, device)
    tok = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
    d_amb = model.config.n_embd          # C lives in AMBIENT d; D is the intrinsic dim
    D = model.config.n_embd - 1
    head_dim = model.config.n_embd // model.config.n_head

    import importlib
    _mod = importlib.import_module(args.battery)
    # id-based batteries (corpus windows, screened selections) expose validate_ids();
    # string batteries expose validate(). Support both rather than assuming the older one.
    if hasattr(_mod, "validate_ids"):
        groups = [(t, None, None) for t, a, b in _mod.validate_ids(tok, verbose=False)]
        _ids = [(a, b) for t, a, b in _mod.validate_ids(tok, verbose=False)]
    else:
        groups = _mod.validate(tok, verbose=False)
        _ids = None
    groups = groups[:args.frames]
    if _ids is not None:
        _ids = _ids[:args.frames]
    rng = np.random.default_rng(0)

    print(f"[baseline] {len(groups)} frames, layers {args.layers}, "
          f"{args.n_rand} random {head_dim}-dim subspaces each, R={args.R}, n_mc={args.n_mc}")
    print(f"           dimension ratio m/D = {head_dim}/{D} = {head_dim/D:.4f}\n")

    per_layer = {L: [] for L in args.layers}
    for gi, (title, tgt, sents) in enumerate(groups):
        if _ids is not None:
            ids_a, ids_b = _ids[gi]
        else:
            ids_a, _pt = target_pos(tok, sents[0], tgt)
            ids_b, _ = target_pos(tok, sents[1], tgt)
        diff = [k for k in range(len(ids_a)) if ids_a[k] != ids_b[k]]
        pos = diff[0] + args.offset
        if pos not in shared_positions(tok, ids_a, ids_b):
            continue
        ta = torch.tensor([ids_a], device=device)
        tb = torch.tensor([ids_b], device=device)
        T = len(ids_a)

        for L in args.layers:
            tap = f"transformer.h.{L}.ln_1"
            kappa = float(kappas[idx_of[tap]])
            mu_pairs = []
            for r in range(args.R):
                gen = torch.Generator(device=device).manual_seed((gi * 100 + L) * 100 + r)
                ma, mb = crn_local_mu_pair(model, controller, policy, depth_order, tap,
                                           ta, tb, T, device, gen)
                mu_pairs.append((unit(ma[pos]), unit(mb[pos])))
            cos = [float(np.clip(a @ b, -1, 1)) for a, b in mu_pairs]
            log_up = float(logsumexp([vmf_utils.log_bhattacharyya(kappa, kappa, c, D)
                                      for c in cos]) - np.log(args.R))
            if log_up >= -1.0:
                continue
            for _ in range(args.n_rand):
                # same pipeline the real heads go through: ambient [m, d] -> row-centered
                # onto the mean-zero hyperplane -> whitened so C C^T = I_m
                A = rng.standard_normal((head_dim, d_amb))
                C, rk = bq.whiten(bq.restrict_to_hyperplane(A))
                assert rk == head_dim
                lg = [bq.bc_projected(a, kappa, b, kappa, C, D, args.n_mc, rng)["log_bc"]
                      for a, b in mu_pairs]
                log_pr = float(logsumexp(lg) - np.log(args.R))
                per_layer[L].append(log_pr / log_up)
        print(f"  frame {gi+1}/{len(groups)} done", flush=True)

    print("\n  layer |   random-projection r        | for reference, real heads at +1")
    for L in args.layers:
        v = np.array(per_layer[L])
        if not len(v):
            continue
        print(f"    {L:3d} | mean {v.mean():.4f}  sd {v.std():.4f}  "
              f"p5 {np.percentile(v,5):.4f}  p95 {np.percentile(v,95):.4f}  (n={len(v)})")
    allv = np.concatenate([np.array(per_layer[L]) for L in args.layers if len(per_layer[L])])
    print(f"\n  pooled: mean {allv.mean():.4f}  sd {allv.std():.4f}   "
          f"vs dimension ratio {head_dim/D:.4f}")


if __name__ == "__main__":
    main()
