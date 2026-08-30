"""
Driver for bhat_qkv.py: per-head Q/K/V Bhattacharyya-coefficient
curves on a real GPT-2 checkpoint, using the EXACT pushforward density (no Gaussian moment
matching, no mixture/KDE).

MU EXTRACTION (the part that is easy to get wrong): a tap's vMF location parameter must
reflect the tap's actual input under REAL upstream noise, not a clean/deterministic upstream
pass. We use a CRN-conditioned LOCAL posterior: for a given frozen realization of every
tap's noise STRICTLY BEFORE the target tap (shared between the two sentences being compared,
so any difference downstream is attributable to content, not to independently-varying
upstream noise), run the real noisy forward pass and capture the target tap's PRE-noise
normalized input. Conditional on that frozen prefix, the target tap's OWN noise
process is exactly the standard local vMF(mu_local, kappa_own) the rest of this codebase
already models (kappa_own = the tap's ordinary learned per-tap kappa, no correction needed
-- the propagation-induced correction lives entirely in mu, not kappa). We repeat this over
R independent frozen-prefix realizations and average BC in log-space (realization-to-
realization spread is itself reported, since it measures how much the read's distinguish-
ability depends on the specific upstream noise draw).

This requires equal token length between the two sentences being compared (a CRN
constraint: the shared frozen noise tensor is [1,T,D]) --
so pairs are formed WITHIN length-matched example groups only, not across them.

Each pairwise entry is DPI-checked against the analytic pre-projection (upstream) BC
automatically inside bhat_qkv.bc_projected (raises on violation beyond MC tolerance), using
the SAME CRN-conditioned mu (not the old global-clean-pass mu) for the upstream reference too.
"""
import argparse
import itertools
import os
import time
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import logsumexp
from transformers import GPT2TokenizerFast

import bhat_qkv as bq
import vmf_utils
from train_stoch_layernorm_gpt import (NoiseController, NoisyLayerNorm, RateBudgetPolicy,
                                       build_model, safe_kappa_lo)

# Pretrained GPT-2 weights, as a local directory (the checkpoints were trained from a
# locally staged copy of the HuggingFace `gpt2` release, so that compute nodes need no
# hub access). Point this at your own copy, or set it to "gpt2" to fetch from the hub.
MODEL_DIR = os.environ.get("GPT2_WEIGHTS", "PATH/TO/gpt2-weights")


def load_run(ckpt_path, device="cuda"):
    """Rebuild a trained noisy-LN GPT-2 from a checkpoint.

    Returns (model, controller, policy, depth_order, idx_of, kappas). Note the two
    orderings, which are NOT the same and are kept deliberately separate: `kappas` is
    indexed by each tap's own `module.idx` (build order, which puts transformer.ln_f at
    index 0 despite it executing last), while `depth_order` is a plain named_modules()
    walk in forward-pass order. Look kappa up as kappas[idx_of[name]], never positionally.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    controller = NoiseController()
    model, n_taps = build_model(controller, pretrained=True, model_name=MODEL_DIR)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    controller.enabled = False          # callers enable the channel for their own draws

    maps = vmf_utils.build_rate_sigma_maps(model.config.n_embd - 1,
                                           kappa_lo=safe_kappa_lo(model.config.n_embd - 1))
    policy = RateBudgetPolicy(n_taps, float(ckpt["B"]), maps).to(device)
    policy.load_state_dict(ckpt["policy"])
    kappas = policy.kappas().detach().cpu().numpy()

    depth_order, idx_of = [], {}
    for name, mod in model.named_modules():
        if isinstance(mod, NoisyLayerNorm):
            depth_order.append(name)
            idx_of[name] = mod.idx
    return model, controller, policy, depth_order, idx_of, kappas

# Display-only floor: the underlying computation is exact and unclipped (npz stores the
# full-precision log_bc), this only affects what gets PLOTTED. Values this project has
# actually seen for these taps range down to ~1e-90+, which is real signal for the DPI
# sanity checks but not useful resolution for reading a depth curve by eye (this analysis
# doesn't need to distinguish anything below 1e-3).
BC_FLOOR = 1e-3

# Length-matched example groups (equal token count WITHIN each group -- required for CRN's
# shared [1,T,D] frozen-noise tensor; verified in main() at startup, not assumed here).
PAIR_GROUPS = [
    ("subject-swap (near-null control)", "store",
     ["I went to the store", "He went to the store", "We went to the store"]),
    ("bank: financial vs river (true polysemy split)", "bank",
     ["He deposited money at the bank", "He wandered along the river bank"]),
    ("suit: convergent resolution", "suit",
     ["The very slanderous article led to a suit", "The journey to the store led to a suit"]),
    # Targets previous-token heads specifically, verified empirically (not assumed) on the
    # base pretrained checkpoint via a standard prefix-matching score (attn to position
    # i-1 on random-token sequences): top
    # overall L4H11 (0.99), top early-layer L3H7 (0.54), L2H2 (0.48). A real noun-swap
    # (not the near-null pronoun swap above) 1-back from the first shared word -- if these
    # heads' K/Q reads are specialized for exactly-previous-token identity, THIS is the
    # position where that should show up as disproportionately better preservation.
    ("noun-swap, near (teacher/student, divergence 1-back from 'closed')", "door",
     ["The teacher closed the old wooden door", "The student closed the old wooden door"]),
    # Same subject swap, same downstream content, but filler pushes 'closed' to 5-back --
    # a controlled contrast for whether preservation at 'closed' specifically depends on
    # being the immediately-previous token, not just on being an early/salient position.
    ("noun-swap, far (teacher/student, divergence 5-back from 'closed')", "door",
     ["The teacher we all really like closed the old wooden door",
      "The student we all really like closed the old wooden door"]),
    # Induction-head triple: targets L5H1/L5H5/L6H9/L7H10 (empirically verified classic
    # GPT2-small induction set, matches Olsson et al./Nanda's TransformerLens
    # demo). Same divergence (teacher/student) and
    # same shared tail ('who closed the door') across all three conditions -- only the
    # trigger word right before 'who' changes: an EXACT repeat of the divergent word (the
    # classic induction cue), a semantically NEAR but non-identical repeat (tests whether
    # induction here is literal-token or fuzzy/semantic, cf. Olsson et al.'s literal vs
    # translation induction), or an unrelated NON-DUPLICATE word (control -- no repeat
    # relationship at all). The trigger word itself can't be a checked position for
    # exact/near (it differs by construction between the two sentences -- that's the
    # point); what we read is whether the SHARED tail downstream of it carries more
    # preserved distinguishability when a real repeat occurred.
    ("induction, exact duplicate", "door",
     ["The teacher we all really admire is the teacher who closed the door",
      "The student we all really admire is the student who closed the door"]),
    ("induction, near duplicate (semantic, not literal)", "door",
     ["The teacher we all really admire is the instructor who closed the door",
      "The student we all really admire is the pupil who closed the door"]),
    ("induction, non-duplicate (control)", "door",
     ["The teacher we all really admire is the person who closed the door",
      "The student we all really admire is the person who closed the door"]),
    # SINGLE-SWAP induction test. The three groups above perturb position 1 AND position 8
    # together, so both sentences carry a repeat and the contrast is [both repeat] vs
    # [neither repeats] -- which forces the non-duplicate control down to ONE perturbation
    # against the other conditions' two, leaving it with ~8x less upstream distinguishability
    # at the tail (13.10 / 14.82 nats vs 1.66) and making "control ranks at chance"
    # uninterpretable: there was nothing there to rank.
    #
    # Here only position 1 is perturbed, so the repeat is present in A and absent in B, and
    # the manipulation moves inside the pair. All three conditions now inject the IDENTICAL
    # teacher/student swap at position 1 and differ only in the token at position 8
    # (teacher / instructor / person), so "induction, non-duplicate (control)" above is
    # already the matched control for these two -- no fourth group is needed.
    #
    # Interpretation shift: upstream logBC at the tail is no longer a nuisance to divide out,
    # it is an OUTCOME. If induction carries position-1 identity past position 8, the exact
    # condition's tail tap should be more distinguishable than the control's before any head
    # read is considered. Read the nats metric (logBC_proj - logBC_up) against raw upstream
    # here; the retention RATIO would normalize the effect away. Within-layer head rankings
    # are unaffected (the two metrics rank heads identically).
    ("induction 1-swap, exact duplicate in A only", "door",
     ["The teacher we all really admire is the teacher who closed the door",
      "The student we all really admire is the teacher who closed the door"]),
    ("induction 1-swap, near duplicate in A only", "door",
     ["The teacher we all really admire is the instructor who closed the door",
      "The student we all really admire is the instructor who closed the door"]),
    # COMPLETED induction pattern. The 1-swap groups above contain a repeat of the swapped
    # word but nothing after it matches, so [A][B]...[A] never becomes [A][B]...[A][B] and an
    # induction head firing there would MISpredict. Here the pattern completes:
    #   0:The 1:doctor 2:spoke 3:calmly 4:at 5:first 6:and 7:then 8:the 9:doctor 10:spoke ...
    #   A=doctor@1, B=spoke@2; A repeats@9 and B genuinely follows@10.
    # Sentence B swaps only position 1 (lawyer), so at position 9 there is no earlier
    # "doctor" and induction cannot fire -- the presence/absence of a completable pattern is
    # the manipulation, with everything downstream of position 1 held identical.
    # Structurally distinct columns for a worked example: pos 2 = copy source, pos 9 = where
    # the query is issued, pos 10 = where the copied token lands.
    ("induction, completed AB...AB pattern", "again",
     ["The doctor spoke calmly at first and then the doctor spoke calmly again",
      "The lawyer spoke calmly at first and then the doctor spoke calmly again"]),
]


def head_A_matrices(model, layer_idx):
    """Raw (UN-restricted, ambient d-dim) per-head A matrices for this layer's attention
    input tap (ln_1), folding gamma and the sqrt(d) LN radius, for all 3 projection types
    and all heads. Bias is dropped (translation-invariance of BC, see note). Returns
    dict[(kind, head)] -> A_raw [head_dim, d], plus (tap_name, d, head_dim, n_head)."""
    ln1 = model.transformer.h[layer_idx].ln_1
    attn = model.transformer.h[layer_idx].attn
    gamma = ln1.weight.detach().cpu().double().numpy()          # [d]
    W = attn.c_attn.weight.detach().cpu().double().numpy()      # [d, 3d] (Conv1D: y = x @ W)
    d = W.shape[0]
    n_head = model.config.n_head
    head_dim = d // n_head
    sqrt_d = float(np.sqrt(d))

    out = {}
    for ki, kind in enumerate(("q", "k", "v")):
        Wk = W[:, ki * d:(ki + 1) * d]                            # [d, d]
        Wk_g = Wk * gamma[:, None]                                # fold LN gain
        for h in range(n_head):
            Wh = Wk_g[:, h * head_dim:(h + 1) * head_dim]         # [d, head_dim]
            A_raw = Wh.T * sqrt_d                                 # [head_dim, d]
            out[(kind, h)] = A_raw
    tap_name = f"transformer.h.{layer_idx}.ln_1"
    return out, tap_name, d, head_dim, n_head


def unit(v):
    return v / (np.linalg.norm(v) + 1e-30)


def _patch_crn_prefix(model, depth_order, frozen, target_tap):
    """Patch every tap STRICTLY BEFORE target_tap to consume the shared `frozen` noise
    tensor (dict name -> [1,T,D]) instead of drawing fresh randn -- the CRN mechanism,
    shared between the two forward passes (sentence A, then sentence B) that will use this
    same patch. Patches target_tap itself only to CAPTURE its PRE-noise normalized input
    (its own noise draw is irrelevant here -- see module docstring). Taps after target_tap
    are left unpatched (their output is never read; wasted compute, not incorrect).
    Returns (cap, restore_fn)."""
    named = dict(model.named_modules())
    cap = {}
    restores = []
    idx_target = depth_order.index(target_tap)

    def factory(name, is_target):
        def forward(self, x):
            xhat = self._normalize(x)
            if is_target:
                cap["mu"] = xhat.detach()
            c = self.controller
            if c.enabled and c.sigma is not None:
                s = c.sigma[self.idx]
                fn = frozen.get(name)
                noise = fn.to(x.device, x.dtype) if fn is not None else torch.randn_like(xhat)
                xhat = self._normalize(xhat + s * noise)
            return xhat * self.weight + self.bias
        return forward

    for name in depth_order[:idx_target + 1]:
        mod = named[name]
        restores.append((mod, mod.forward))
        mod.forward = types.MethodType(factory(name, name == target_tap), mod)

    def restore():
        for mod, orig in restores:
            mod.forward = orig
    return cap, restore


@torch.no_grad()
def crn_local_mu_pair(model, controller, policy, depth_order, target_tap, ids_a, ids_b, T,
                      device, gen):
    """One frozen-prefix realization: shared real noise for every tap before target_tap,
    independent for sentence A vs B beyond that (moot -- we only read target_tap, which is
    captured PRE its own noise). Returns (mu_a, mu_b) each [D] on CPU float64, unit-normed
    by the caller (bc_projected normalizes internally too, but we keep raw here)."""
    D = model.config.n_embd
    idx_target = depth_order.index(target_tap)
    frozen = {name: torch.randn(1, T, D, device=device, generator=gen)
              for name in depth_order[:idx_target]}
    cap, restore = _patch_crn_prefix(model, depth_order, frozen, target_tap)
    controller.enabled = True
    controller.sigma = policy.sigmas().detach()
    try:
        model.transformer(input_ids=ids_a)
        mu_a = cap["mu"][0].double().cpu().numpy()
        model.transformer(input_ids=ids_b)
        mu_b = cap["mu"][0].double().cpu().numpy()
    finally:
        restore()
        controller.enabled = False
    return mu_a, mu_b


def target_pos(tok, sentence, target_word):
    ids = tok(sentence)["input_ids"]
    toks = [tok.decode([i]).strip().lower() for i in ids]
    matches = [i for i, t in enumerate(toks) if t == target_word.lower()]
    assert matches, f"'{target_word}' not found as a single token in {toks!r}"
    return ids, matches[-1]


def shared_positions(tok, ids_a, ids_b):
    """Position indices i>0 where sentence A and B have the IDENTICAL token (decoded,
    stripped, lowercased) at the SAME index i -- these are the positions where "read this
    tap at position i for both sentences" is comparing the same word, not two different
    words that happen to share a slot. Position 0 is excluded on principle, not just as an
    empirical non-finding: under GPT-2's causal masking position 0 can only ever attend to
    itself, so its representation is IDENTICAL to the embedding at every layer regardless
    of anything later in the sentence -- BC there is guaranteed exactly 1 by construction,
    not a real measurement of anything upstream."""
    toks_a = [tok.decode([i]).strip().lower() for i in ids_a]
    toks_b = [tok.decode([i]).strip().lower() for i in ids_b]
    return [i for i in range(1, len(ids_a)) if toks_a[i] == toks_b[i]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="PATH/TO/gpt2_checkpoint.pt",
                    help="checkpoint written by train_stoch_layernorm_gpt.py")
    ap.add_argument("--label", default="sigma=1")
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--heads", type=int, nargs="+", default=None)
    ap.add_argument("--n_mc", type=int, default=1500)
    ap.add_argument("--R", type=int, default=8, help="frozen-prefix realizations to average over")
    ap.add_argument("--out", default="outputs/qkv_bc")
    ap.add_argument("--pairs", type=int, nargs="+", default=None,
                    help="restrict to these GLOBAL pair indices (order of PAIR_GROUPS "
                         "expansion). Saved keys keep the global index, and the npz carries a "
                         "`pair_index` array, so a restricted scan stays directly comparable "
                         "to a full one instead of silently renumbering.")
    ap.add_argument("--offsets", type=int, nargs="+", default=None,
                    help="probe ONLY positions this many tokens after the first perturbed "
                         "index (e.g. --offsets 1). Cost is linear in probed positions.")
    ap.add_argument("--battery", default=None,
                    help="module exposing validate(tok) -> PAIR_GROUPS-format list, used "
                         "INSTEAD of the built-in PAIR_GROUPS. Indices are then local to that "
                         "battery (its own npz), not to PAIR_GROUPS.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, controller, policy, depth_order, idx_of, kappas = load_run(args.ckpt, device)
    tok = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
    D = model.config.n_embd - 1
    n_layer = model.config.n_layer
    layers = args.layers if args.layers is not None else list(range(n_layer))
    heads = args.heads if args.heads is not None else list(range(model.config.n_head))

    # build pairs, each carrying EVERY shared-token position (not just the target) that it
    # makes sense to compare -- i.e. positions where A and B have the identical word, so
    # "distance back from the target" can be read off the SAME captured sequence at no
    # extra forward-pass cost (see shared_positions docstring for why position 0 is
    # excluded on principle, and why groups like "bank" end up with only the target).
    groups, id_groups = PAIR_GROUPS, None
    if args.battery:
        import importlib
        mod = importlib.import_module(args.battery)
        # Batteries that build sequences from token ids (random tokens, corpus windows)
        # expose validate_ids() instead of validate(). Going through strings would require
        # decode->re-encode to round-trip exactly, which GPT-2 BPE does not guarantee: a
        # generated pair could silently change length or acquire a second differing index.
        if hasattr(mod, "validate_ids"):
            id_groups = mod.validate_ids(tok)
            print(f"[setup] battery {args.battery!r}: {len(id_groups)} id-built pairs", flush=True)
        else:
            groups = mod.validate(tok)
            print(f"[setup] using battery {args.battery!r} ({len(groups)} validated groups) "
                  f"INSTEAD of PAIR_GROUPS", flush=True)

    pairs = []
    if id_groups is not None:
        for title, ids_a, ids_b in id_groups:
            assert len(ids_a) == len(ids_b), f"{title}: lengths {len(ids_a)} vs {len(ids_b)}"
            positions = shared_positions(tok, ids_a, ids_b)
            words = [tok.decode([ids_a[i]]).strip() for i in positions]
            pairs.append({"gi": len(pairs), "group": title, "target": "",
                         "a": tok.decode(ids_a), "b": tok.decode(ids_b),
                         "pos_target": len(ids_a) - 1,   # no target word; report offsets from end
                         "positions": positions, "words": words, "T": len(ids_a),
                         "ids_a": torch.tensor([ids_a], device=device),
                         "ids_b": torch.tensor([ids_b], device=device)})
    else:
        for title, tgt, sents in groups:
            toks_pos = [target_pos(tok, s, tgt) for s in sents]
            Ts = {len(ids) for ids, _ in toks_pos}
            assert len(Ts) == 1, f"group {title!r} is not length-matched: {[len(i) for i,_ in toks_pos]}"
            for (i, j) in itertools.combinations(range(len(sents)), 2):
                ids_a, pos_target = toks_pos[i]; ids_b, pos_target_b = toks_pos[j]
                assert pos_target == pos_target_b, "target word not at the same index in A/B"
                positions = shared_positions(tok, ids_a, ids_b)
                assert pos_target in positions, "target position must itself be a shared position"
                words = [tok.decode([ids_a[i]]).strip() for i in positions]
                pairs.append({"gi": len(pairs),
                             "group": title, "target": tgt, "a": sents[i], "b": sents[j],
                             "pos_target": pos_target, "positions": positions, "words": words,
                             "T": len(ids_a),
                             "ids_a": torch.tensor([ids_a], device=device),
                             "ids_b": torch.tensor([ids_b], device=device)})

    if args.offsets is not None:
        # Keep only positions a given number of tokens after the FIRST perturbed index.
        # This is what makes a reconnaissance scan cheap: cost is linear in probed positions.
        for p in pairs:
            ia = p["ids_a"][0].tolist(); ib = p["ids_b"][0].tolist()
            first_diff = next(k for k in range(len(ia)) if ia[k] != ib[k])
            keep = {first_diff + o for o in args.offsets}
            sel = [(pos, w) for pos, w in zip(p["positions"], p["words"]) if pos in keep]
            assert sel, f"offsets {args.offsets} select no shared position in {p['group']!r}"
            p["positions"] = [pos for pos, _ in sel]
            p["words"] = [w for _, w in sel]
            p["pos_target"] = first_diff
    if args.pairs is not None:
        keep = set(args.pairs)
        unknown = keep - {p["gi"] for p in pairs}
        assert not unknown, f"--pairs referenced nonexistent global indices {sorted(unknown)}"
        pairs = [p for p in pairs if p["gi"] in keep]
    print(f"[setup] {args.label}  {len(pairs)} length-matched pairs  layers={layers}  "
          f"heads={heads}  R={args.R}  n_mc={args.n_mc}  D={D}", flush=True)
    for p in pairs:
        print(f"  [pair {p['gi']}] [{p['group']}] {p['a']!r} vs {p['b']!r}  T={p['T']}  "
              f"positions checked (back from target): {list(zip(p['positions'], p['words']))}",
              flush=True)

    rng = np.random.default_rng(0)
    results = {}   # (pi, pos, layer, kind, h) -> {"log_bc_mean", "log_bc_std", "log_bc_up_mean"}

    t_start = time.time()
    n_pos_total = sum(len(p["positions"]) for p in pairs)
    n_total = len(layers) * n_pos_total * (1 + 3 * len(heads))
    n_done = 0
    for layer in layers:
        target_tap = f"transformer.h.{layer}.ln_1"
        kappa = float(kappas[idx_of[target_tap]])
        A_dict, tap_name2, d, head_dim, n_head = head_A_matrices(model, layer)
        assert tap_name2 == target_tap

        for p in pairs:
            pi = p["gi"]        # GLOBAL pair index, so --pairs runs stay comparable to full ones
            # R frozen-prefix realizations of the FULL sequence (one forward pass per
            # sentence per realization) -- reused across every checked position below,
            # since we already captured the whole [T,D] tap output, not just one slot.
            mu_full_pairs = []
            for r in range(args.R):
                seed = (layer * 1000 + pi) * 1000 + r
                gen = torch.Generator(device=device).manual_seed(seed)
                mu_a, mu_b = crn_local_mu_pair(model, controller, policy, depth_order,
                                               target_tap, p["ids_a"], p["ids_b"], p["T"],
                                               device, gen)
                mu_full_pairs.append((mu_a, mu_b))

            for pos, word in zip(p["positions"], p["words"]):
                mu_pairs, cos_list = [], []
                for mu_a, mu_b in mu_full_pairs:
                    mu_a_pos, mu_b_pos = unit(mu_a[pos]), unit(mu_b[pos])
                    mu_pairs.append((mu_a_pos, mu_b_pos))
                    cos_list.append(float(np.clip(mu_a_pos @ mu_b_pos, -1.0, 1.0)))
                log_bc_up_r = [float(vmf_utils.log_bhattacharyya(kappa, kappa, c, D)) for c in cos_list]
                log_bc_up_mean = float(logsumexp(log_bc_up_r) - np.log(args.R))
                n_done += 1

                for kind in ("q", "k", "v"):
                    for h in heads:
                        C, rk = bq.whiten(bq.restrict_to_hyperplane(A_dict[(kind, h)]))
                        assert rk == head_dim
                        log_bc_r = []
                        for (mu_a_pos, mu_b_pos) in mu_pairs:
                            res = bq.bc_projected(mu_a_pos, kappa, mu_b_pos, kappa, C, D, args.n_mc, rng)
                            log_bc_r.append(res["log_bc"])
                        log_bc_r = np.array(log_bc_r)
                        log_bc_mean = float(logsumexp(log_bc_r) - np.log(args.R))
                        results[(pi, pos, layer, kind, h)] = {
                            "log_bc_mean": log_bc_mean, "log_bc_std": float(log_bc_r.std()),
                            "log_bc_up_mean": log_bc_up_mean, "word": word,
                            "back": p["pos_target"] - pos}
                        n_done += 1
        elapsed = time.time() - t_start
        rate = n_done / max(elapsed, 1e-9)
        eta = (n_total - n_done) / max(rate, 1e-9)
        print(f"[layer {layer:2d}] kappa={kappa:8.1f}  done {n_done}/{n_total} "
              f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    npz_path = f"{args.out}_{args.label}.npz"
    save = {"layers": np.array(layers), "heads": np.array(heads),
            "pair_index": np.array([p["gi"] for p in pairs]),
            "pair_meta": np.array([f"{p['group']} | {p['a']} vs {p['b']}" for p in pairs])}
    for (pi, pos, layer, kind, h), v in results.items():
        for k2, val in v.items():
            if k2 == "word":
                continue
            save[f"{k2}_{pi}_{pos}_{layer}_{kind}_{h}"] = val
    np.savez(npz_path, **save)
    print(f"[done] wrote {npz_path}", flush=True)

    _plot_overview(pairs, layers, heads, results, args.label, f"{args.out}_{args.label}_overview.png")
    for kind in ("q", "k", "v"):
        _plot_per_head(pairs, layers, heads, results, kind, args.label,
                       f"{args.out}_{args.label}_perhead_{kind}.png")


def _plot_overview(pairs, layers, heads, results, label, out_path):
    """Quick-scan summary: mean-over-heads + min-max shading. Collapses exactly the
    per-head structure that's usually the interesting part of this project's findings --
    use this only for orientation, see _plot_per_head for the real per-head view."""
    max_pos = max(len(p["positions"]) for p in pairs)
    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, max_pos, figsize=(4.6 * max_pos, 3.6 * n_pairs),
                             sharey=True, squeeze=False)
    colors = {"q": "#1f6fb2", "k": "#c1440e", "v": "#2e8b57"}

    for row, p in enumerate(pairs):
        pi = p["gi"]        # results are keyed by GLOBAL index; `row` is this figure's row
        for col in range(max_pos):
            ax = axes[row][col]
            if col >= len(p["positions"]):
                ax.axis("off")
                continue
            pos = p["positions"][col]
            back = results[(pi, pos, layers[0], "q", heads[0])]["back"]
            word = results[(pi, pos, layers[0], "q", heads[0])]["word"]

            up_mean = np.clip(np.exp([results[(pi, pos, l, "q", heads[0])]["log_bc_up_mean"] for l in layers]),
                              BC_FLOOR, 1.0)
            ax.plot(layers, up_mean, "-o", ms=4, color="black", lw=1.8,
                   label="upstream (CRN-local)")
            for kind in ("q", "k", "v"):
                vals = np.array([[results[(pi, pos, l, kind, h)]["log_bc_mean"] for h in heads] for l in layers])
                mean_over_heads = np.clip(np.exp(logsumexp(vals, axis=1) - np.log(len(heads))), BC_FLOOR, 1.0)
                lo = np.clip(np.exp(vals.min(axis=1)), BC_FLOOR, 1.0)
                hi = np.clip(np.exp(vals.max(axis=1)), BC_FLOOR, 1.0)
                ax.plot(layers, mean_over_heads, "-o", ms=3, color=colors[kind],
                       label=f"{kind.upper()} (mean over heads)")
                ax.fill_between(layers, lo, hi, color=colors[kind], alpha=0.15)
            ax.set_yscale("log")
            ax.set_ylim(BC_FLOOR * 0.7, 1.3)
            back_str = "target" if back == 0 else f"target-{back}"
            ax.set_title(f"'{word}' ({back_str})", fontsize=8)
            ax.grid(alpha=0.3, which="both")
            if col == 0:
                ax.set_ylabel(f"{p['group']}\n{p['a']!r} vs\n{p['b']!r}\n\nBC (log)", fontsize=7)
            if pi == n_pairs - 1:
                ax.set_xlabel("attention layer")
            if pi == 0 and col == 0:
                ax.legend(fontsize=6, loc="lower left")
    fig.suptitle(f"Exact per-head Q/K/V pushforward BC vs CRN-local upstream BC, by position -- {label}\n"
                f"columns = shared-token positions (target rightmost among filled columns "
                f"unless the group has back-positions); shaded = min-max across heads", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[done] wrote {out_path}", flush=True)


def _plot_per_head(pairs, layers, heads, results, kind, label, out_path):
    """One figure per Q/K/V type, full per-head resolution: every head gets its own line
    (color = head index, viridis), no averaging. Same (pair, position) grid as the
    overview plot. This is the primary view -- head-to-head heterogeneity has repeatedly
    been the actual finding in this project (e.g. induction-head noise-robustness), not
    something to average away."""
    max_pos = max(len(p["positions"]) for p in pairs)
    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, max_pos, figsize=(4.6 * max_pos, 3.6 * n_pairs),
                             sharey=True, squeeze=False)
    cmap = plt.get_cmap("viridis")
    n_head = len(heads)

    for row, p in enumerate(pairs):
        pi = p["gi"]        # results are keyed by GLOBAL index; `row` is this figure's row
        for col in range(max_pos):
            ax = axes[row][col]
            if col >= len(p["positions"]):
                ax.axis("off")
                continue
            pos = p["positions"][col]
            back = results[(pi, pos, layers[0], kind, heads[0])]["back"]
            word = results[(pi, pos, layers[0], kind, heads[0])]["word"]

            up_mean = np.clip(np.exp([results[(pi, pos, l, kind, heads[0])]["log_bc_up_mean"] for l in layers]),
                              BC_FLOOR, 1.0)
            ax.plot(layers, up_mean, "-", color="black", lw=2.2, alpha=0.8,
                   label="upstream (CRN-local)", zorder=n_head + 1)
            for hi, h in enumerate(heads):
                vals = np.clip(np.exp([results[(pi, pos, l, kind, h)]["log_bc_mean"] for l in layers]),
                               BC_FLOOR, 1.0)
                ax.plot(layers, vals, "-o", ms=2.5, lw=1.1, color=cmap(hi / max(1, n_head - 1)),
                       label=f"head {h}")
            ax.set_yscale("log")
            ax.set_ylim(BC_FLOOR * 0.7, 1.3)
            back_str = "target" if back == 0 else f"target-{back}"
            ax.set_title(f"'{word}' ({back_str})", fontsize=8)
            ax.grid(alpha=0.3, which="both")
            if col == 0:
                ax.set_ylabel(f"{p['group']}\n{p['a']!r} vs\n{p['b']!r}\n\nBC (log)", fontsize=7)
            if pi == n_pairs - 1:
                ax.set_xlabel("attention layer")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=heads[0], vmax=heads[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01, ticks=heads)
    cbar.set_label("head index", fontsize=8)
    fig.suptitle(f"Exact per-head {kind.upper()} pushforward BC vs CRN-local upstream BC (black), "
                f"by position -- {label}\nno averaging: every line is one head", fontsize=10)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
