"""Per-head Q/K/V Bhattacharyya maps for the vMF-LayerNorm ViTs -- the image analog of
`qkv_bc_gpt.py` (GPT-2). For a base image and a designed patch perturbation, this asks,
at every attention tap: how much of the perturbation's distinguishability survives each
individual head's Q, K and V read?

Method is the GPT-2 one, unchanged in substance:
  * CRN prefix freeze -- every tap strictly BEFORE the target tap consumes one shared,
    REAL noise realization for both the base and the perturbed forward pass, so upstream
    stochasticity is common and the only difference is the perturbation. The target tap is
    captured PRE its own noise: that pre-noise unit direction is the vMF mean mu, which is
    a legitimate conditional mean given real frozen upstream noise, NOT a clean/mu pass.
  * Exact pushforward BC (`bhat_qkv.bc_projected`) of vMF_D(mu_base, kappa) vs
    vMF_D(mu_pert, kappa) through each head's whitened projection C. Carries its own
    correctness test: the data-processing inequality BC_projected >= BC_upstream must hold,
    and an AssertionError fires if it does not.

timm packs Q,K,V into one fused `attn.qkv` Linear ([3D, D], y = x W^T), so the head slices
are rows [kind*D + h*head_dim : ... ]; the LN gain gamma is folded into the columns and the
LN radius sqrt(D) into the scale (both cosmetic -- whitening removes any constant factor).

Tokens the perturbation has not reached are bit-identical under CRN (cos = 1 exactly) and
are short-circuited to BC = 1 rather than integrated.

MANDATORY NULL (--n_rand, on by default): "head" here means nothing more than "this 64-d
subspace, orthonormalized" -- bc_projected needs CC^T = I, so each head's anisotropy is
discarded (harmless in practice: measured effective rank of A is 47-60 of 64). Since the
n_head x head_dim subspaces tile the whole embedding space, a GENERIC perturbation projects
into every one of them by roughly the same amount, and the per-head maps come out nearly
identical for reasons that have nothing to do with what the head does. So every run also
pushes the same mu-pairs through `n_rand` RANDOM head_dim-subspaces on a token subset. Any
claim of head differentiation must clear that null: real-head map-to-map correlation must be
LOWER (and across-head spread HIGHER) than the random-subspace band. On a plain single-patch
fade it does not clear it: the apparent head separation there is at the null.
"""
import argparse, os, types
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

import bhat_qkv as bq
import vmf_utils
from train_stoch_layernorm_vit import NoisyLayerNorm
from crn_perturb_vit import load_run, load_image, perturb, pick_foreground_patches, _denorm

KINDS = ("q", "k", "v")


def head_A_matrices(model, li):
    """dict[(kind, head)] -> A_raw [head_dim, D] for block li's attention input tap."""
    blk = model.blocks[li]
    gamma = blk.norm1.weight.detach().cpu().double().numpy()          # [D]
    W = blk.attn.qkv.weight.detach().cpu().double().numpy()           # [3D, D]
    D = W.shape[1]
    nh = blk.attn.num_heads
    hd = D // nh
    out = {}
    for ki, kind in enumerate(KINDS):
        Wk = W[ki * D:(ki + 1) * D, :]                                # [D, D]
        for h in range(nh):
            Wh = Wk[h * hd:(h + 1) * hd, :]                           # [hd, D]
            out[(kind, h)] = (Wh * gamma[None, :]) * np.sqrt(D)
    return out, D, hd, nh


def _patch_crn_prefix(taps, frozen, target_name):
    """Taps before target consume `frozen` noise; target captures its PRE-noise read."""
    names = [nm for nm, _ in taps]
    mods = dict(taps)
    it = names.index(target_name)
    cap, restores = {}, []

    def factory(name, is_target):
        def forward(self, x):
            xhat = self._normalize(x)
            if is_target:
                cap["mu"] = xhat.detach()
            c = self.controller
            if c.enabled and c.sigma is not None:
                fn = frozen.get(name)
                noise = fn.to(x.device, x.dtype) if fn is not None else torch.randn_like(xhat)
                xhat = self._normalize(xhat + c.sigma[self.idx] * noise)
            return xhat * self.weight + self.bias
        return forward

    for name in names[:it + 1]:
        mod = mods[name]
        restores.append((mod, mod.forward))
        mod.forward = types.MethodType(factory(name, name == target_name), mod)

    def restore():
        for mod, orig in restores:
            mod.forward = orig
    return cap, restore


@torch.no_grad()
def crn_local_mu_pair(run, target_name, x_base, x_pert, device, gen):
    """One frozen-prefix realization -> (mu_base [T,D], mu_pert [T,D]) float64 on CPU."""
    taps = run["taps"]
    names = [nm for nm, _ in taps]
    it = names.index(target_name)
    D = run["model"].embed_dim
    frozen = {nm: torch.randn(1, run["T"], D, device=device, generator=gen)
              for nm in names[:it]}
    cap, restore = _patch_crn_prefix(taps, frozen, target_name)
    run["controller"].enabled = True
    try:
        run["model"](x_base.unsqueeze(0))
        mu_b = cap["mu"][0].double().cpu().numpy()
        run["model"](x_pert.unsqueeze(0))
        mu_p = cap["mu"][0].double().cpu().numpy()
    finally:
        restore()
        run["controller"].enabled = False
    return mu_b, mu_p

def separation_stats(bc_layer, bc_rand_layer, rtoks):
    """Do the real heads separate more than random subspaces of the same dimension?

    Both are evaluated on the SAME token subset and the same mu-pairs, so the only
    difference is which subspace the read is projected onto. Returns
    (real, null) where real = {kind: (corr, spread)} and null = (corr, spread):
      corr   mean pairwise Pearson r between subspaces' maps  (LOWER  = more distinct)
      spread mean over tokens of std/mean across subspaces    (HIGHER = more distinct)
    A head-differentiation claim needs corr below and spread above the null."""
    def stats(M):
        M = np.clip(np.asarray(M, dtype=float), 1e-300, None)
        nats = -np.log(M)
        good = np.isfinite(nats).all(0) & (nats > 0).all(0)
        if good.sum() < 3 or len(M) < 2:
            return np.nan, np.nan
        Z = np.corrcoef(np.log(nats[:, good]))
        return (float(Z[~np.eye(len(M), dtype=bool)].mean()),
                float((nats.std(0) / nats.mean(0))[good].mean()))
    real = {kind: stats(bc_layer[ki][:, rtoks]) for ki, kind in enumerate(KINDS)}
    # The null must be computed on GROUPS OF THE SAME SIZE as the real head set: mean
    # pairwise correlation and across-subspace spread both depend on how many subspaces
    # are in the group, so comparing 6 real heads against a pool of n_rand randoms
    # directly is not a like-for-like test. Bootstrap size-nh subsets of the pool instead
    # and report the null as mean +- sd, which also gives the verdict an error bar.
    nh = bc_layer.shape[1]
    pool = bc_rand_layer.shape[0]
    rs = np.random.default_rng(12345)
    cs, sps = [], []
    for _ in range(200 if pool > nh else 1):
        idx = rs.choice(pool, size=min(nh, pool), replace=False)
        c, sp = stats(bc_rand_layer[idx])
        cs.append(c); sps.append(sp)
    null = (float(np.nanmean(cs)), float(np.nanstd(cs)),
            float(np.nanmean(sps)), float(np.nanstd(sps)))
    return real, null


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--image", required=True)
    ap.add_argument("--patch", default=None, help="r,c on the 14x14 grid; default = auto foreground")
    ap.add_argument("--mode", default="fade")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n_mc", type=int, default=1500)
    ap.add_argument("--n_crn", type=int, default=1, help="frozen-prefix realizations to average")
    ap.add_argument("--layers", default=None, help="comma list; default all")
    ap.add_argument("--n_rand", type=int, default=12,
                    help="random head_dim-subspaces for the separation null (0 disables)")
    ap.add_argument("--rand_tokens", type=int, default=48,
                    help="token subset the random-subspace null is evaluated on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="vit_outputs/qkv_bc")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = load_run(args.ckpt, device)
    model, gh_gw = run["model"], run["grid"]
    gh, gw = gh_gw
    x = load_image(args.image, model, device)
    if args.patch:
        pr, pc = (int(v) for v in args.patch.split(","))
    else:
        pr, pc = pick_foreground_patches(x, gh_gw, k=1)[0]
    x_pert = perturb(x, (pr, pc), args.mode, args.alpha)
    layers = ([int(v) for v in args.layers.split(",")] if args.layers
              else list(range(len(model.blocks))))
    nh = model.blocks[0].attn.num_heads
    D = model.embed_dim
    kap_of = {nm: run["kappas"][mod.idx] for nm, mod in run["taps"]}
    gen = torch.Generator(device=device).manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    bc = np.ones((len(layers), len(KINDS), nh, run["T"]))
    exact = np.zeros((len(layers), len(KINDS), nh, run["T"]), dtype=bool)
    rtoks = np.linspace(1, run["T"] - 1, min(args.rand_tokens, run["T"] - 1)).astype(int)
    bc_rand = np.ones((len(layers), max(args.n_rand, 1), len(rtoks)))   # no kind axis: a
    # random subspace has no q/k/v identity, so one null band serves all three comparisons
    dpi_fail = 0
    print(f"[qkv-vit] {os.path.basename(args.ckpt)}  {os.path.basename(args.image)}  "
          f"patch=({pr},{pc}) {args.mode} a={args.alpha}  layers={layers}  heads={nh}  "
          f"n_mc={args.n_mc}  n_crn={args.n_crn}", flush=True)

    for li_i, li in enumerate(layers):
        tap = f"blocks.{li}.norm1"
        kappa = float(kap_of[tap])
        A_dict, _, hd, _ = head_A_matrices(model, li)
        Cs = {key: bq.whiten(bq.restrict_to_hyperplane(A))[0] for key, A in A_dict.items()}
        acc = np.zeros((len(KINDS), nh, run["T"]))
        for rep in range(args.n_crn):
            mu_b, mu_p = crn_local_mu_pair(run, tap, x, x_pert, device, gen)
            nb = mu_b / np.linalg.norm(mu_b, axis=-1, keepdims=True)
            npp = mu_p / np.linalg.norm(mu_p, axis=-1, keepdims=True)
            cos_up = np.clip((nb * npp).sum(-1), -1, 1)
            moved = cos_up < 1.0 - 1e-12
            for ki, kind in enumerate(KINDS):
                for h in range(nh):
                    C = Cs[(kind, h)]
                    for t in np.flatnonzero(moved):
                        try:
                            r = bq.bc_projected(mu_b[t], kappa, mu_p[t], kappa, C, D,
                                                args.n_mc, rng)
                            acc[ki, h, t] += r["bc"]
                        except AssertionError:
                            acc[ki, h, t] += np.nan
                            dpi_fail += 1
                    acc[ki, h, ~moved] += 1.0
            exact[li_i] = ~moved
        bc[li_i] = acc / args.n_crn

        if args.n_rand:                       # separation null: random subspaces, same mu-pairs
            Rs = [bq.whiten(bq.restrict_to_hyperplane(rng.normal(size=(hd, D))))[0]
                  for _ in range(args.n_rand)]
            for ri, C in enumerate(Rs):
                for j, t in enumerate(rtoks):
                    if not moved[t]:
                        continue
                    try:
                        bc_rand[li_i, ri, j] = bq.bc_projected(
                            mu_b[t], kappa, mu_p[t], kappa, C, D, args.n_mc, rng)["bc"]
                    except AssertionError:
                        bc_rand[li_i, ri, j] = np.nan
                        dpi_fail += 1

        n_moved = int(moved.sum())
        real, null = separation_stats(bc[li_i], bc_rand[li_i], rtoks)
        rc, rsp = real["k"]
        nc, nc_sd, ns, ns_sd = null
        zc = (nc - rc) / nc_sd if nc_sd > 0 else np.nan       # +z = heads MORE distinct
        zs = (rsp - ns) / ns_sd if ns_sd > 0 else np.nan
        verdict = "CLEARS NULL" if (zc > 2 and zs > 2) else "at null"
        print(f"  L{li:<2d} kappa={kappa:7.1f}  moved={n_moved}/{run['T']}  "
              f"min BC(K)={np.nanmin(bc[li_i, 1]):.4g}  ||  SEPARATION(K): "
              f"corr {rc:.4f} vs null {nc:.4f}+-{nc_sd:.4f} (z={zc:+.1f})  "
              f"spread {rsp:.4f} vs null {ns:.4f}+-{ns_sd:.4f} (z={zs:+.1f})  -> {verdict}",
              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out + ".npz", bc=bc, exact=exact, layers=np.array(layers),
             kinds=np.array(KINDS), patch=np.array([pr, pc]), grid=np.array([gh, gw]),
             kappas=np.array([kap_of[f"blocks.{li}.norm1"] for li in layers]),
             image=args.image, ckpt=args.ckpt, mode=args.mode, alpha=args.alpha,
             n_mc=args.n_mc, n_crn=args.n_crn, dpi_fail=dpi_fail,
             bc_rand=bc_rand, rand_tokens=rtoks, n_rand=args.n_rand)
    print(f"[qkv-vit] DPI violations: {dpi_fail}   saved {args.out}.npz", flush=True)


if __name__ == "__main__":
    main()
