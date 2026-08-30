"""
Exact per-head Q/K/V Bhattacharyya coefficients for the vMF LayerNorm posterior --
NO Gaussian moment matching, no mixture/KDE density estimation. The pushforward of
vMF(mu,kappa) on S^{D-1} through a linear Q/K/V head projection has an EXACT density
after whitening the projection to have orthonormal rows; the derivation is given in
`bc_projected` below.

Ambient-dimension convention matches vmf_utils.py throughout this project: D is the
ambient dimension of the mean-zero hyperplane (= channel count d minus 1), and z lives
on S^{D-1} inside that hyperplane (vmf_utils calls this "p"). We pass D through to
vmf_utils.log_bhattacharyya etc. as its `p` argument.

Everything below works directly with vectors in their AMBIENT d-dim array representation
(mean-zero, e.g. the captured LayerNorm xhat, or unit-norm mu directions) -- exactly how
the rest of this codebase already represents them (see vmf_utils.py).
There is no need to ever construct an explicit orthonormal basis for the D-dim intrinsic
hyperplane: see `restrict_to_hyperplane` below for why row-centering the
projection matrix in its NATIVE d-dim ambient form gives an identical, correctly-whitened
result (proven in the module docstring of that function, cross-checked in `_validate`).

This module is agnostic to HOW mu/kappa were obtained -- it just needs valid vMF(mu,kappa)
parameters for a single tap. See qkv_bc_gpt.py for the extraction methodology (must use
a CRN-conditioned local posterior under real upstream noise, NOT a clean/deterministic
upstream pass, which is known to give qualitatively wrong answers here).

Reusable primitives (for joint-attention / mixture-posterior extensions):
  - `restrict_to_hyperplane`, `whiten`      -> build C from a raw ambient weight map
  - `project_mu`                            -> (a, |b|) for a given mu and C
  - `log_pushforward_density`               -> the exact density itself (mixtures of
                                                posteriors are just mixtures of this)
  - `bc_projected`                          -> BC estimator + DPI-checked wrapper
"""

import numpy as np
from scipy.special import ive, gammaln, logsumexp

import vmf_utils


# --------------------------------------------------------------------------- Bessel M_n
def log_Mn(x, nu, x_thresh=1e-4):
    """log M_n(x) = log I_nu(x) - nu*log(x), stable for large nu (orders in the hundreds
    for GPT-2-sized D) via scipy's exponentially-scaled `ive`, and stable for small x via
    the analytic series limit (avoids the 0/0 and the catastrophic cancellation that a
    naive log(ive)+x-nu*log(x) would hit as x->0, since log(x) diverges while the true
    quantity converges to a finite limit)."""
    x = np.asarray(x, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    small = x < x_thresh
    xs = np.where(small, 1.0, x)  # dummy safe value where we'll discard the result anyway
    log_I = np.log(ive(nu, xs)) + xs
    log_M_big = log_I - nu * np.log(xs)
    # I_nu(x) = (x/2)^nu/Gamma(nu+1) * (1 + x^2/(4(nu+1)) + O(x^4)), so
    # M_n(x) = I_nu(x)/x^nu -> 1/(2^nu Gamma(nu+1)) * (1 + x^2/(4(nu+1)) + ...)
    log_M_small = -nu * np.log(2.0) - gammaln(nu + 1.0) + np.log1p(x ** 2 / (4.0 * (nu + 1.0)))
    return np.where(small, log_M_small, log_M_big)


# --------------------------------------------------------------------------- projection
def restrict_to_hyperplane(A):
    """A: [m, d] ambient linear map (ROW i = the i-th output direction's weight vector
    over the d channels), acting on ambient d-vectors. Row-CENTER each output direction
    (subtract its own mean across the d channels).

    Why this is exactly the right thing (not an approximation): let P = I_d - (1/d)11^T be
    the orthogonal projector onto the mean-zero hyperplane, and E (d x D) any orthonormal
    basis of that hyperplane (E^T E = I_D, E E^T = P). The "intrinsic" D-dim map is
    A_D = A @ E, and its Gram is A_D A_D^T = A E E^T A^T = A P A^T. Since P acts on the
    RIGHT of A by centering each row (P is symmetric idempotent, (A P)[i,:] = A[i,:] -
    mean(A[i,:])*1), A_restricted := A @ P is literally the row-centered A, and
    A_restricted @ A_restricted^T = A P P^T A^T = A P A^T = A_D A_D^T exactly -- so
    row-centering reproduces the correct intrinsic Gram without ever constructing E.
    Applied to any genuinely mean-zero z (our actual posteriors always are), A_restricted
    @ z = A @ z identically (the centering term contributes (mean row)*sum(z)=0), so this
    is a pure bookkeeping fix for the whitening step, not a change to what gets evaluated."""
    return A - A.mean(axis=-1, keepdims=True)


def whiten(A, rank_tol=1e-8):
    """A: [m, d], rows already hyperplane-restricted. Returns C: [r, d] with C C^T = I_r
    (r = detected rank, normally r=m) via eigendecomposition of the Gram (AA^T), not a
    naive inverse. Flags rank deficiency by returning r < m."""
    G = A @ A.T
    w, V = np.linalg.eigh(G)  # ascending
    tol = rank_tol * max(w.max(), 1e-300)
    keep = w > tol
    r = int(keep.sum())
    if r < A.shape[0]:
        print(f"[bhat_qkv] WARNING: rank-deficient projection ({r}/{A.shape[0]}); "
              f"eigenvalues={w}", flush=True)
    w_inv_sqrt = np.zeros_like(w)
    w_inv_sqrt[keep] = 1.0 / np.sqrt(w[keep])
    M = (V * w_inv_sqrt) @ V.T          # (AA^T)^{-1/2} (pseudo-inverse sqrt if rank-deficient)
    C = M @ A
    if r < A.shape[0]:
        # drop the (numerically null) rows so CC^T = I_r exactly, not I_m with zero rows
        order = np.argsort(-w)          # keep the r largest-eigenvalue directions
        C = (V[:, order[:r]] * (1.0 / np.sqrt(w[order[:r]]))).T @ A
    return C, r


def project_mu(C, mu):
    """C: [m,d], mu: [d] unit-norm ambient direction. Returns (a=[m], |b| scalar)."""
    a = C @ mu
    b_mag = float(np.sqrt(max(0.0, 1.0 - float(a @ a))))
    return a, b_mag


# --------------------------------------------------------------------------- density
def log_pushforward_density(u, a, b_mag, kappa, D, m):
    """Exact log pushforward density of vMF_D(mu,kappa) through C, up to a constant shared
    by every (a,b_mag,kappa) at this (D,m) -- i.e. valid for ratios/differences at fixed
    (D,m), and (separately, additively) for the toy dense-integration check in _validate,
    where that missing constant is calibrated by requiring the density to integrate to 1.
    u: [..., m] points in the open unit ball B^m."""
    u = np.atleast_2d(u)
    a = np.asarray(a, dtype=np.float64)
    u2 = np.clip(np.sum(u * u, axis=-1), 0.0, 1.0 - 1e-15)
    r = np.sqrt(1.0 - u2)
    n = D - m
    assert n > 0, f"n=D-m={n} must be positive (D={D}, m={m})"
    nu = n / 2.0 - 1.0
    term_beta = ((D - m - 2) / 2.0) * np.log1p(-u2)
    term_lin = kappa * (u @ a)
    term_bessel = log_Mn(kappa * b_mag * r, nu)
    return term_beta + term_lin + term_bessel


# --------------------------------------------------------------------------- BC estimator
def bc_projected(mu_p, kappa_p, mu_q, kappa_q, C, D, n_mc, rng, dpi_tol=6.0):
    """Exact-pushforward Bhattacharyya coefficient between vMF_D(mu_p,kappa_p) and
    vMF_D(mu_q,kappa_q) after projection through C ([m,d], CC^T=I_m).

    Derivation: writing the pushforward
    density as (shared universal constant) x C_D(kappa) x f(u; a,b_mag,kappa) [f = exp of
    log_pushforward_density], the shared constant AND the C_D(kappa) normalizer of the
    bridge proposal cancel algebraically, leaving

        BC_projected = BC_upstream * E_{u~bridge}[ sqrt(f_p(u) f_q(u)) / f_bridge(u) ]

    i.e. the exact analytic upstream BC (vmf_utils.log_bhattacharyya, no approximation)
    times an importance-sampling correction factor computed ENTIRELY from log-density
    differences (no need to ever know the missing normalizing constant). This is why the
    DPI check is automatic: the correction factor need only come out >= 1 (within MC/
    numerical tolerance) for BC_projected >= BC_upstream to hold.

    Returns dict with bc, se, log_bc, bc_upstream, correction, correction_rel_se, n_mc.
    Raises AssertionError if the DPI inequality is violated beyond `dpi_tol` MC standard
    errors (or a small absolute floor for near-zero SE)."""
    mu_p = mu_p / np.linalg.norm(mu_p)
    mu_q = mu_q / np.linalg.norm(mu_q)
    cos = float(np.clip(mu_p @ mu_q, -1.0, 1.0))
    assert kappa_p == kappa_q, "same-tap pair must share kappa (see module docstring)"
    kappa = float(kappa_p)
    m = C.shape[0]

    log_bc_up = float(vmf_utils.log_bhattacharyya(kappa, kappa, cos, D))

    # bridge proposal: sqrt(p_Z q_Z) ~ vMF(v, kappa_bridge)
    s = mu_p + mu_q
    s_norm = float(np.linalg.norm(s))
    if s_norm < 1e-12:  # mu_p ~= -mu_q: bridge direction undefined, falls back to either mu
        v = mu_p
        kappa_bridge = 0.0
    else:
        v = s / s_norm
        kappa_bridge = 0.5 * kappa * s_norm

    z_s = vmf_utils.sample_mixture(v[None, :], kappa_bridge, D, n_mc, rng)  # [n_mc, d]
    u_s = z_s @ C.T                                                        # [n_mc, m]

    a_p, b_p = project_mu(C, mu_p)
    a_q, b_q = project_mu(C, mu_q)
    a_v, b_v = project_mu(C, v)

    log_fp = log_pushforward_density(u_s, a_p, b_p, kappa, D, m)
    log_fq = log_pushforward_density(u_s, a_q, b_q, kappa, D, m)
    log_fv = log_pushforward_density(u_s, a_v, b_v, kappa_bridge, D, m)

    log_ratio = 0.5 * (log_fp + log_fq) - log_fv                            # [n_mc]

    # Stay in log-space until the very end: at realistic GPT-2 scale (kappa into the
    # thousands, D~767) bc_up=exp(log_bc_up) can underflow to a hard 0.0 while the IS
    # correction factor mean_corr=exp(log_mean_corr) can simultaneously overflow -- their
    # product as separately-exponentiated floats gives 0.0*inf=nan. Summing the logs first
    # (well-conditioned: both are normal-magnitude finite doubles, only their *individual*
    # exponentials are extreme) and exponentiating once avoids that failure mode entirely.
    log_mean_corr = logsumexp(log_ratio) - np.log(n_mc)
    log_mean_corr2 = logsumexp(2.0 * log_ratio) - np.log(n_mc)
    # relative variance of the correction factor, Var(W)/E[W]^2 = E[W^2]/E[W]^2 - 1, is
    # scale-free (well-conditioned regardless of how extreme mean_corr itself is)
    rel_var_corr = max(0.0, float(np.exp(log_mean_corr2 - 2.0 * log_mean_corr)) - 1.0)
    rel_se_corr = float(np.sqrt(rel_var_corr / n_mc))

    log_bc = log_bc_up + log_mean_corr
    bc = float(np.exp(log_bc))
    se = bc * rel_se_corr
    bc_up = float(np.exp(log_bc_up))
    mean_corr = float(np.exp(min(log_mean_corr, 700.0)))  # diagnostic only; may saturate

    tol = max(dpi_tol * se, 1e-9)
    if bc < bc_up - tol:
        raise AssertionError(
            f"DPI violation: BC_projected={bc:.6g} (se={se:.2g}) < "
            f"BC_upstream={bc_up:.6g} by more than {dpi_tol} SE "
            f"(kappa={kappa:.3g}, cos={cos:.4f}, m={m}, D={D})")

    return {"bc": bc, "se": se, "log_bc": log_bc, "bc_upstream": bc_up,
            "correction": mean_corr, "correction_rel_se": rel_se_corr, "n_mc": n_mc,
            "cos": cos, "kappa": kappa}


# --------------------------------------------------------------------------- self-tests
def _random_unit_hyperplane_vec(d, rng):
    v = rng.normal(size=d)
    v -= v.mean()
    return v / np.linalg.norm(v)


def _check_hyperplane_restriction(rng):
    """restrict_to_hyperplane (row-centering, ambient d-dim) must give the identical Gram
    AA^T as explicitly projecting through an orthonormal basis of the mean-zero hyperplane."""
    from scipy.linalg import null_space
    d, m = 11, 3
    A_raw = rng.normal(size=(m, d))
    E = null_space(np.ones((1, d)))            # [d, d-1], E^T E = I, E E^T = hyperplane projector
    A_D = A_raw @ E
    gram_explicit = A_D @ A_D.T
    A_restricted = restrict_to_hyperplane(A_raw)
    gram_direct = A_restricted @ A_restricted.T
    err = np.abs(gram_explicit - gram_direct).max()
    print(f"  hyperplane-restriction equivalence: max|Gram diff| = {err:.2e}")
    assert err < 1e-9, "row-centering Gram does not match explicit-basis Gram"

    # also check that A_restricted @ z == A_raw @ z for genuine mean-zero z (not just Gram)
    z = _random_unit_hyperplane_vec(d, rng)
    err2 = np.abs(A_restricted @ z - A_raw @ z).max()
    print(f"  hyperplane-restriction action-on-z check: max|diff| = {err2:.2e}")
    assert err2 < 1e-9


def _dense_ball_integral(f_vals, mask, cell_area):
    return float(f_vals[mask].sum() * cell_area)


def _check_dense_integration(rng):
    """Brute-force validation in a tiny toy problem: dense grid quadrature over the m=2
    unit ball, compared against bc_projected's bridge-IS estimator. Also checks that the
    missing normalizing constant (K_D,m * C_D(kappa)) inferred from integrating f_p to 1
    is (a) independent of which mu it's inferred from at fixed kappa, and (b) scales with
    kappa exactly as C_D(kappa) predicts -- i.e. validates the "up to constants" claim in
    log_pushforward_density directly, not just the BC ratio."""
    D, d, m = 8, 9, 2
    A_raw = rng.normal(size=(m, d))
    C, r = whiten(restrict_to_hyperplane(A_raw))
    assert r == m

    mu_p = _random_unit_hyperplane_vec(d, rng)
    mu_q = _random_unit_hyperplane_vec(d, rng)

    n_grid = 500
    lin = np.linspace(-1, 1, n_grid)
    U1, U2 = np.meshgrid(lin, lin)
    u_grid = np.stack([U1.ravel(), U2.ravel()], axis=-1)
    mask = (u_grid ** 2).sum(-1) < 1.0
    cell_area = (lin[1] - lin[0]) ** 2

    def normalizer(a, b_mag, kappa):
        log_f = log_pushforward_density(u_grid, a, b_mag, kappa, D, m)
        f = np.exp(log_f - log_f[mask].max())  # rescale for safety, correct after
        Z_rel = _dense_ball_integral(f, mask, cell_area)
        return Z_rel, log_f[mask].max()

    for kappa in (6.0, 14.0):
        a_p, b_p = project_mu(C, mu_p)
        a_q, b_q = project_mu(C, mu_q)
        Zp_rel, Zp_shift = normalizer(a_p, b_p, kappa)
        Zq_rel, Zq_shift = normalizer(a_q, b_q, kappa)
        # K*C_D(kappa) inferred two ways (from p alone, from q alone) must agree
        logKC_p = -np.log(Zp_rel) - Zp_shift
        logKC_q = -np.log(Zq_rel) - Zq_shift
        print(f"  kappa={kappa:5.1f}  log(K*C_D) from mu_p={logKC_p:.6f}  from mu_q={logKC_q:.6f}"
              f"  diff={abs(logKC_p - logKC_q):.2e}")
        assert abs(logKC_p - logKC_q) < 1e-3, "normalizing constant depends on mu -- should not"

        # BC via dense integration, using the shared (mu-independent) normalizer from p
        log_fp = log_pushforward_density(u_grid, a_p, b_p, kappa, D, m)
        log_fq = log_pushforward_density(u_grid, a_q, b_q, kappa, D, m)
        sqrt_fpfq = np.exp(0.5 * (log_fp + log_fq) + logKC_p)
        bc_dense = _dense_ball_integral(sqrt_fpfq, mask, cell_area)

        res = bc_projected(mu_p, kappa, mu_q, kappa, C, D, n_mc=400_000,
                           rng=np.random.default_rng(0))
        rel_err = abs(bc_dense - res["bc"]) / max(bc_dense, 1e-12)
        print(f"    BC dense-grid={bc_dense:.6f}  BC bridge-IS={res['bc']:.6f} "
              f"+/- {res['se']:.2g}  rel.err={rel_err:.4f}")
        assert rel_err < 0.02, "dense-integration BC and bridge-IS BC disagree"

    # K*C_D(kappa) should scale between the two kappas exactly as vmf_utils.log_C predicts
    a_p, b_p = project_mu(C, mu_p)
    _, shift6 = normalizer(a_p, b_p, 6.0)
    Z6, s6 = normalizer(a_p, b_p, 6.0)
    Z14, s14 = normalizer(a_p, b_p, 14.0)
    logKC6 = -np.log(Z6) - s6
    logKC14 = -np.log(Z14) - s14
    predicted_delta = float(vmf_utils.log_C(14.0, D) - vmf_utils.log_C(6.0, D))
    observed_delta = logKC14 - logKC6
    print(f"  log(K*C_D) delta(kappa 6->14): observed={observed_delta:.4f} "
          f"predicted(vmf_utils.log_C)={predicted_delta:.4f}")
    assert abs(observed_delta - predicted_delta) < 1e-2


def _check_gpt2_scale_no_overflow(rng, n_trials=15):
    """GPT-2-small-scale D/m with kappa pushed into the regime where BC_upstream itself
    underflows to a hard float64 0.0 for well-separated (near-orthogonal) mu -- this is
    exactly the failure mode fixed by summing log_bc_up+log_mean_corr before a single
    final exp() (see bc_projected's comment): the old bc_up*mean_corr product could hit
    0.0*inf=nan here. Confirms bc/se/log_bc all come out finite and in-range instead."""
    D, d, m = 767, 768, 64
    saw_real_underflow = False
    for _ in range(n_trials):
        A_raw = rng.normal(size=(m, d))
        C, r = whiten(restrict_to_hyperplane(A_raw))
        assert r == m
        mu_p = _random_unit_hyperplane_vec(d, rng)
        mu_q = _random_unit_hyperplane_vec(d, rng)
        kappa = float(rng.uniform(2000.0, 20000.0))
        res = bc_projected(mu_p, kappa, mu_q, kappa, C, D, n_mc=4000, rng=rng)
        if res["bc_upstream"] == 0.0:
            saw_real_underflow = True
        assert np.isfinite(res["bc"]) and np.isfinite(res["se"]) and np.isfinite(res["log_bc"]), \
            f"non-finite result at kappa={kappa:.0f}: {res}"
        assert -1e-6 <= res["bc"] <= 1.0 + 1e-6, f"BC out of [0,1]: {res}"
    print(f"  {n_trials} GPT-2-scale (D={D}, m={m}) high-kappa trials: all finite & in [0,1]"
          f" (bc_upstream underflowed to hard 0.0 in at least one trial: {saw_real_underflow})")
    assert saw_real_underflow, \
        "test didn't actually exercise the underflow regime -- raise the kappa range"


def _check_identity_and_bounds(rng):
    D, d, m = 8, 9, 2
    A_raw = rng.normal(size=(m, d))
    C, r = whiten(restrict_to_hyperplane(A_raw))
    mu = _random_unit_hyperplane_vec(d, rng)
    for kappa in (3.0, 50.0, 500.0):
        res = bc_projected(mu, kappa, mu, kappa, C, D, n_mc=20_000, rng=rng)
        print(f"  identity check kappa={kappa:6.1f}: BC={res['bc']:.6f} (want ~1)")
        assert abs(res["bc"] - 1.0) < 1e-3


def _check_dpi_battery(rng, n_trials=25):
    D, d, m = 40, 41, 8
    violations = 0
    corrections = []
    for _ in range(n_trials):
        A_raw = rng.normal(size=(m, d))
        C, r = whiten(restrict_to_hyperplane(A_raw))
        assert r == m
        mu_p = _random_unit_hyperplane_vec(d, rng)
        mu_q = _random_unit_hyperplane_vec(d, rng)
        kappa = float(rng.uniform(5.0, 400.0))
        try:
            res = bc_projected(mu_p, kappa, mu_q, kappa, C, D, n_mc=8000, rng=rng)
        except AssertionError as e:
            violations += 1
            print(f"  DPI VIOLATION: {e}")
            continue
        corrections.append(res["correction"])
        assert -1e-6 <= res["bc"] <= 1.0 + 1e-6, f"BC out of [0,1]: {res['bc']}"
    corrections = np.array(corrections)
    print(f"  {n_trials - violations}/{n_trials} pairs passed DPI; "
          f"correction factor range [{corrections.min():.3f}, {corrections.max():.3f}], "
          f"mean={corrections.mean():.3f}")
    assert violations == 0, f"{violations} DPI violations out of {n_trials}"


def _validate():
    rng = np.random.default_rng(0)
    print("[bhat_qkv] hyperplane restriction vs explicit orthonormal basis")
    _check_hyperplane_restriction(rng)
    print("[bhat_qkv] dense-grid brute-force integration vs bridge-IS estimator")
    _check_dense_integration(rng)
    print("[bhat_qkv] identical posteriors -> BC=1")
    _check_identity_and_bounds(rng)
    print("[bhat_qkv] DPI battery (random pairs/kappas)")
    _check_dpi_battery(rng)
    print("[bhat_qkv] GPT-2-scale high-kappa overflow/underflow stress test")
    _check_gpt2_scale_no_overflow(rng)
    print("[bhat_qkv] all checks passed")


if __name__ == "__main__":
    _validate()
