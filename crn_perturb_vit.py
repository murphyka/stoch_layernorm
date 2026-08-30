"""
CRN perturbation-similarity for the vMF-LayerNorm ViTs -- the image analog of the
paired-draw CRN probe used on GPT-2. Two conditions: a BASE image and a designed PERTURBATION
(localized patch fade/noise, or broad recolor). Using common random numbers (the same
pre-generated per-tap noise in both forward passes), we measure, per token and per tap,
the vMF Bhattacharyya coefficient between the base and perturbed reads.

Because the noise realization is held fixed, an unaffected token gives BC=1 exactly
(same noise, same content) -- so 1-BC is a clean map of WHERE the perturbation actually
propagated, with no noise-floor subtraction. Patch tokens -> a 14x14 heatmap per tap;
the CLS token -> a separate curve vs depth. Watching the low-BC ("loud") region spread
across depth shows whether a foreground-patch perturbation stays within its object.

Loads any vMF ViT checkpoint: p and num_classes are read from
the ckpt, and each tap's kappa is taken from the module's own build-idx (no realignment
needed -- the NoisyLayerNorm knows its index).
"""
import argparse, os, types
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.special import logsumexp
from PIL import Image
import timm

import vmf_utils
from train_stoch_layernorm_vit import NoiseController, NoisyLayerNorm, RateBudgetPolicy, build_model


def depth_key(nm):
    return (99, 0) if nm == "norm" else (int(nm.split(".")[1]), 0 if nm.endswith("norm1") else 1)


def load_run(ckpt, device):
    sd = torch.load(ckpt, map_location="cpu")
    a = sd.get("args", {})
    model_name = a.get("model", "vit_small_patch16_224")
    num_classes = sd["model"]["head.weight"].shape[0]
    controller = NoiseController()
    model, n_taps = build_model(model_name, num_classes, 0.0, controller, pretrained=False)
    model.load_state_dict(sd["model"], strict=True)
    model.to(device).eval()
    P = model.embed_dim - 1
    maps = vmf_utils.build_rate_sigma_maps(P)
    pol = RateBudgetPolicy(n_taps, float(sd.get("B", 1.0)), maps)
    pol.load_state_dict(sd["policy"])
    controller.sigma = pol.sigmas().detach().to(device)
    kappas = pol.kappas().detach().cpu().numpy()          # build-idx order == module.idx
    taps = [(nm, mod) for nm, mod in model.named_modules() if isinstance(mod, NoisyLayerNorm)]
    taps.sort(key=lambda t: depth_key(t[0]))
    grid = model.patch_embed.grid_size                     # (14,14)
    T = model.patch_embed.num_patches + 1
    return dict(model=model, controller=controller, kappas=kappas, taps=taps, P=P,
                model_name=model_name, num_classes=num_classes, grid=grid, T=T)


def load_image(path, model, device):
    cfg = timm.data.resolve_model_data_config(model)
    tf = timm.data.create_transform(**{**cfg, "crop_pct": cfg.get("crop_pct", 0.9)}, is_training=False)
    return tf(Image.open(path).convert("RGB")).to(device)    # [3,224,224], normalized


def perturb(x, patch, mode, alpha, patch_px=16, seed=0):
    """x: [3,H,W] normalized. patch=(r,c) in grid units. Returns perturbed copy."""
    xp = x.clone()
    if mode == "recolor":                                    # broad: tint the whole image
        g = torch.tensor([1.0, -1.0, 1.0], device=x.device).view(3, 1, 1)
        return xp + alpha * g
    r, c = patch
    ys, xs = slice(r * patch_px, (r + 1) * patch_px), slice(c * patch_px, (c + 1) * patch_px)
    if mode == "fade":                                       # blend patch toward gray (norm 0)
        xp[:, ys, xs] = (1 - alpha) * xp[:, ys, xs]
    elif mode == "paste":                                    # overwrite with FIXED content
        # every site in every image ends up with the identical patch content, so the
        # perturbed state is matched across runs (the delta still depends on what was
        # there before -- callers record ||dx|| so it can be regressed out)
        g = torch.Generator(device="cpu").manual_seed(1234)
        block = torch.randn(3, patch_px, patch_px, generator=g).clamp(-2, 2).to(x.device)
        xp[:, ys, xs] = (1 - alpha) * xp[:, ys, xs] + alpha * block
    elif mode == "noise":                                    # add fixed gaussian to the patch
        g = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(3, patch_px, patch_px, generator=g).to(x.device)
        xp[:, ys, xs] = xp[:, ys, xs] + alpha * noise
    else:
        raise ValueError(mode)
    return xp


def crn_forward_factory(noise, capture):
    def forward(self, x):
        xhat = self._normalize(x)
        c = self.controller
        if c.enabled and c.sigma is not None:
            xhat = self._normalize(xhat + c.sigma[self.idx] * noise[:, : x.shape[1], :].to(x.dtype))
        capture["xhat"] = xhat.detach()
        return xhat * self.weight + self.bias
    return forward


@torch.no_grad()
def collect_crn(run, x_base, x_pert, M, device):
    model, taps = run["model"], run["taps"]
    run["controller"].enabled = True
    caps, restore = {}, []
    for nm, mod in taps:                                     # one shared noise tensor per tap
        cap = {}; caps[nm] = cap
        noise = torch.randn(M, run["T"], mod.weight.shape[0], device=device)
        orig = mod.forward
        mod.forward = types.MethodType(crn_forward_factory(noise, cap), mod)
        restore.append((mod, orig))
    base = {}
    model(x_base.unsqueeze(0).repeat(M, 1, 1, 1))
    for nm in caps:
        base[nm] = caps[nm]["xhat"].clone()
    pert = {}
    model(x_pert.unsqueeze(0).repeat(M, 1, 1, 1))
    for nm in caps:
        pert[nm] = caps[nm]["xhat"].clone()
    for mod, orig in restore:
        mod.forward = orig
    run["controller"].enabled = False
    return base, pert


def per_token_sums(a, b, kappa, p):
    """a,b: [m,T,D] one chunk of CRN-paired draws. Returns per-token SUMS over the chunk
    (bc_sum[T], angle_sum[T], m) so results accumulate across chunks -> unbounded total M
    at bounded memory. bc = paired vMF Bhattacharyya at this tap's kappa (how LOUD through
    the channel); angle = paired rotation in degrees (KAPPA-FREE geometric effect, visible
    even at collapsed taps; companion to GPT-2 paired_angles_deg). Linear float64 sums are
    safe -- individual BCs that underflow at high kappa correctly contribute ~0 to the mean."""
    an = F.normalize(a, dim=-1); bn = F.normalize(b, dim=-1)
    cos = (an * bn).sum(-1).clamp(-1, 1).double().cpu().numpy()     # [m,T]
    bc = np.exp(vmf_utils.log_bhattacharyya(kappa, kappa, cos, p))  # [m,T]
    ang = np.degrees(np.arccos(cos))                              # [m,T]
    return bc.sum(0), ang.sum(0), cos.shape[0]


def pick_foreground_patches(x, grid, k=3, min_sep=3, patch_px=16):
    """Pick k patches with highest input-gradient energy (edges/texture -> object, not flat
    background), spatially separated by >= min_sep grid cells. Returns [(r,c), ...]."""
    gh, gw = grid
    e = torch.zeros(x.shape[1], x.shape[2], device=x.device)
    e[:, :-1] += (x[:, :, 1:] - x[:, :, :-1]).abs().sum(0)
    e[:-1, :] += (x[:, 1:, :] - x[:, :-1, :]).abs().sum(0)
    pe = e[:gh * patch_px, :gw * patch_px].reshape(gh, patch_px, gw, patch_px).mean((1, 3)).cpu().numpy()
    picked = []
    for flat in np.argsort(pe.ravel())[::-1]:
        r, c = np.unravel_index(flat, pe.shape)
        if all(max(abs(r - pr), abs(c - pc)) >= min_sep for pr, pc in picked):
            picked.append((int(r), int(c)))
            if len(picked) == k:
                break
    return picked


def analyze_and_save(run, x, x_pert, patch, mode, alpha, M, m_chunk, ckpt, base_out, device):
    """Chunk-accumulate CRN paired stats, write _bc.png / _angle.png / .npz. Returns arrays."""
    pr, pc = patch; gh, gw = run["grid"]
    names = [nm for nm, _ in run["taps"]]
    kaps = [float(run["kappas"][mod.idx]) for _, mod in run["taps"]]
    sum_bc = {nm: 0.0 for nm in names}; sum_ang = {nm: 0.0 for nm in names}; total = 0
    while total < M:
        m = min(m_chunk, M - total)
        base, pert = collect_crn(run, x, x_pert, m, device)
        for nm, mod in run["taps"]:
            k = float(run["kappas"][mod.idx])
            sb, sa, _ = per_token_sums(base[nm], pert[nm], k, run["P"])
            sum_bc[nm] = sum_bc[nm] + sb; sum_ang[nm] = sum_ang[nm] + sa
        total += m
    heat_bc, heat_ang, cls_bc, cls_ang = [], [], [], []
    for nm in names:
        bc = sum_bc[nm] / total; ang = sum_ang[nm] / total
        cls_bc.append(bc[0]); cls_ang.append(ang[0])
        heat_bc.append(bc[1:].reshape(gh, gw)); heat_ang.append(ang[1:].reshape(gh, gw))
    heat_bc = np.stack(heat_bc); heat_ang = np.stack(heat_ang)
    cls_bc = np.array(cls_bc); cls_ang = np.array(cls_ang)
    meta = dict(ckpt=ckpt, mode=mode, alpha=alpha, M=M)
    _plot(run, x, 1 - heat_bc, cls_bc, names, kaps, patch, f"{base_out}_bc.png", 1.0, True,
          "CLS BC (base vs pert)", "1−BC (loud through the channel; κ-weighted)", meta)
    vmax = float(np.percentile(heat_ang, 98)) or 1.0
    _plot(run, x, heat_ang, cls_ang, names, kaps, patch, f"{base_out}_angle.png", vmax, False,
          "CLS rotation (deg)", f"paired rotation degrees (κ-FREE; vmax={vmax:.1f}°)", meta)
    np.savez(f"{base_out}.npz", heat_bc=heat_bc, heat_angle=heat_ang, cls_bc=cls_bc,
             cls_angle=cls_ang, names=np.array(names), kappas=np.array(kaps), patch=patch,
             mode=mode, alpha=alpha, M=M)
    return dict(heat_bc=heat_bc, heat_angle=heat_ang, cls_bc=cls_bc, cls_angle=cls_ang)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="PATH/TO/vit_checkpoint.pt",
                    help="checkpoint written by train_stoch_layernorm_vit.py")
    ap.add_argument("--img", required=True)
    ap.add_argument("--patch", default="7,7", help="perturbed patch 'row,col' in the 14x14 grid")
    ap.add_argument("--mode", choices=["fade", "noise", "recolor"], default="fade")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--M", type=int, default=256, help="total CRN-paired draws to average")
    ap.add_argument("--m_chunk", type=int, default=256, help="draws per batch (memory cap); accumulated")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run = load_run(args.ckpt, device)
    x = load_image(args.img, run["model"], device)
    pr, pc = (int(v) for v in args.patch.split(","))
    x_pert = perturb(x, (pr, pc), args.mode, args.alpha)
    print(f"[setup] {run['model_name']} P={run['P']} classes={run['num_classes']} "
          f"grid={run['grid']} M={args.M} mode={args.mode} alpha={args.alpha} patch={(pr,pc)}", flush=True)

    tag = os.path.splitext(os.path.basename(args.ckpt))[0] + "_" + os.path.splitext(os.path.basename(args.img))[0]
    base_out = args.out or f"outputs/crn_perturb_{tag}_p{pr}-{pc}_{args.mode}_M{args.M}"
    base_out = base_out[:-4] if base_out.endswith(".png") else base_out
    analyze_and_save(run, x, x_pert, (pr, pc), args.mode, args.alpha, args.M, args.m_chunk,
                     args.ckpt, base_out, device)
    print(f"[done] {base_out}_bc.png / _angle.png", flush=True)


def _denorm(x):
    return (x.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)


def _plot(run, x, heat, cls_vals, names, kaps, patch, out, vmax, cls_low_is_reached, ylabel, title, meta):
    pr, pc = patch; n = len(names)
    fig = plt.figure(figsize=(17, 20))
    gs = GridSpec(6, 5, figure=fig, hspace=0.32, wspace=0.08, height_ratios=[1.1, 1, 1, 1, 1, 1])
    # input image with perturbed patch marked
    axi = fig.add_subplot(gs[0, 0]); axi.imshow(_denorm(x)); axi.set_title("input (perturbed patch)", fontsize=9)
    axi.add_patch(Rectangle((pc * 16, pr * 16), 16, 16, ec="cyan", fc="none", lw=2)); axi.axis("off")
    # CLS metric vs depth
    axc = fig.add_subplot(gs[0, 1:])
    axc.plot(range(n), cls_vals, "-o", ms=4, color="#c1440e")
    if cls_low_is_reached:
        axc.set_ylim(0, 1.02); axc.axhline(1, color="grey", lw=0.6, ls=":")
        reach = "low = perturbation reached the CLS summary"
    else:
        axc.axhline(0, color="grey", lw=0.6, ls=":"); reach = "high = perturbation reached the CLS summary"
    axc.set_ylabel(ylabel); axc.set_xlabel("tap (depth order)")
    axc.set_xticks(range(n)); axc.set_xticklabels(
        [nm.replace("blocks.", "b").replace(".norm", ".n") for nm in names], rotation=90, fontsize=6)
    axc.set_title(f"CLS vs depth  ({reach})", fontsize=9); axc.grid(alpha=0.3)
    # per-tap patch heatmaps
    for i in range(n):
        ax = fig.add_subplot(gs[1 + i // 5, i % 5])
        ax.imshow(heat[i], vmin=0, vmax=vmax, cmap="magma")
        ax.add_patch(Rectangle((pc - 0.5, pr - 0.5), 1, 1, ec="cyan", fc="none", lw=1.5))
        lab = names[i].replace("blocks.", "b").replace(".norm", ".n")
        ax.set_title(f"d{i}: {lab}  κ={kaps[i]:.0f}", fontsize=7.5); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"CRN patch-perturbation — {title}  —  {os.path.basename(meta['ckpt'])}\n"
                 f"patch {patch} {meta['mode']} α={meta['alpha']}, M={meta['M']}", fontsize=12)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
