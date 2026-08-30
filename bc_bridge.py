"""Bounded-variance Monte Carlo Bhattacharyya coefficient between two vMF MIXTURES.

Identity (exact). With m = (p+q)/2,
    E_{z~m}[ 2 sqrt(p(z)q(z)) / (p(z)+q(z)) ]
      = integral (p+q)/2 * 2 sqrt(pq)/(p+q) = integral sqrt(pq) = BC(P,Q).
Sampling from m is trivial: take half the draws from P and half from Q.

Why this and not the existing vmf_utils.log_bc_mc. That one uses E_{x~p}[sqrt(q/p)], whose
integrand is UNBOUNDED -- wherever the sampled x has q(x) >> p(x) the weight blows up, giving
the heavy-tailed variance that made the mixture-of-posteriors BC unusable (it fails a
split-half identity check structurally). Here AM-GM gives sqrt(pq) <= (p+q)/2, so the
integrand is bounded in [0,1] and the estimator has bounded variance by construction, whatever the
mixtures look like.

Everything is done in log space: at kappa in the hundreds and p ~ 383 the densities overflow
float64 individually while their ratios stay perfectly ordinary.
"""
import numpy as np
from scipy.special import logsumexp

import vmf_utils


def log_bc_bridge(mus_a, mus_b, kappa, p, n_mc, rng, return_diag=False):
    """log BC between (1/Ma)sum vMF(mu_i^a,kappa) and (1/Mb)sum vMF(mu_j^b,kappa).

    n_mc total bridge draws, half from each mixture. Returns log BC, or with return_diag a
    dict adding the MC standard error and the effective sample size ESS = (sum h)^2/sum h^2
    (ess_frac = ESS/n_mc; near 1 = the average is carried by most draws, small = a few draws
    dominate and the estimate is unreliable no matter how small its nominal se looks).
    """
    na = n_mc // 2
    za = vmf_utils.sample_mixture(mus_a, kappa, p, na, rng)
    zb = vmf_utils.sample_mixture(mus_b, kappa, p, n_mc - na, rng)
    z = np.concatenate([za, zb], axis=0)
    lp = vmf_utils.mixture_log_density(z, mus_a, kappa, p)
    lq = vmf_utils.mixture_log_density(z, mus_b, kappa, p)
    # h = 2 sqrt(pq)/(p+q); log h = log2 + (lp+lq)/2 - logaddexp(lp,lq) <= 0 by AM-GM
    log_h = np.log(2.0) + 0.5 * (lp + lq) - np.logaddexp(lp, lq)
    log_bc = logsumexp(log_h) - np.log(len(log_h))
    if not return_diag:
        return float(log_bc)
    n = len(log_h)
    # variance of a bounded variate: compute in log space then exponentiate once
    log_m2 = logsumexp(2.0 * log_h) - np.log(n)
    var = max(0.0, float(np.exp(log_m2)) - float(np.exp(log_bc)) ** 2)
    # Effective sample size, ESS = (sum h)^2 / sum h^2, in log space. This replaces an
    # earlier `frac_active` heuristic that counted draws within an ARBITRARY 20 log-units of
    # the mean -- ESS needs no cutoff and is the standard diagnostic. ESS/n near 1 means the
    # average is carried by most draws; ESS/n small means a handful of draws dominate and the
    # estimate is not to be trusted however small its nominal se looks.
    log_ess = 2.0 * logsumexp(log_h) - logsumexp(2.0 * log_h)
    return dict(log_bc=float(log_bc), bc=float(np.exp(log_bc)),
                se=float(np.sqrt(var / n)), n_mc=n,
                ess=float(np.exp(log_ess)), ess_frac=float(np.exp(log_ess) / n),
                max_log_h=float(log_h.max()))


# --------------------------------------------------------------------- validation
def _check_single_component(rng, p=383, D=384):
    """A one-component mixture is just a vMF, so the bridge estimate must match the exact
    closed form. This is the load-bearing check: if it fails, nothing downstream is real."""
    print("  [1] single-component vs CLOSED FORM (the decisive check)")
    print(f"      {'kappa':>7}{'angle':>8}{'exact logBC':>14}{'bridge logBC':>14}"
          f"{'abs err':>10}{'se':>9}{'err/se':>8}")
    worst = 0.0
    for kappa in (50.0, 200.0, 600.0):
        for ang in (2.0, 10.0, 30.0, 60.0):
            mu1 = rng.normal(size=D); mu1 -= mu1.mean(); mu1 /= np.linalg.norm(mu1)
            v = rng.normal(size=D); v -= v.mean()
            v -= (v @ mu1) * mu1; v /= np.linalg.norm(v)
            th = np.radians(ang)
            mu2 = np.cos(th) * mu1 + np.sin(th) * v
            exact = float(vmf_utils.log_bhattacharyya(kappa, kappa, np.cos(th), p))
            r = log_bc_bridge(mu1[None, :], mu2[None, :], kappa, p, 40000, rng,
                              return_diag=True)
            err = abs(r["log_bc"] - exact)
            se_log = r["se"] / max(r["bc"], 1e-300)
            worst = max(worst, err / max(se_log, 1e-12))
            print(f"      {kappa:>7.0f}{ang:>8.0f}{exact:>14.4f}{r['log_bc']:>14.4f}"
                  f"{err:>10.4f}{se_log:>9.4f}{err/max(se_log,1e-12):>8.1f}")
    print(f"      worst |err|/se = {worst:.1f}  (should be a few; large => biased)")
    return worst


def _check_identity(rng, p=383, D=384):
    """Identical mixtures must give BC exactly 1 (log BC = 0) for ANY draw, since then
    h(z) == 1 pointwise. This is an algebraic identity, not a statistical one."""
    print("\n  [2] IDENTITY: same mixture both sides -> log BC must be exactly 0")
    worst = 0.0
    for M in (1, 8, 64):
        mus = rng.normal(size=(M, D)); mus -= mus.mean(-1, keepdims=True)
        mus /= np.linalg.norm(mus, axis=-1, keepdims=True)
        for kappa in (50.0, 600.0):
            lb = log_bc_bridge(mus, mus, kappa, p, 4000, rng)
            worst = max(worst, abs(lb))
            print(f"      M={M:<4} kappa={kappa:>5.0f}   log BC = {lb:+.3e}")
    print(f"      worst |log BC| = {worst:.2e}  (must be ~1e-15)")
    return worst


def _check_vs_old(rng, p=383, D=384):
    """Variance comparison against the unbounded-weight estimator on the same mixtures."""
    print("\n  [3] variance vs the old unbounded importance-sampling estimator")
    M = 16
    base = rng.normal(size=D); base -= base.mean(); base /= np.linalg.norm(base)
    for spread_deg, kappa in ((5.0, 400.0), (20.0, 400.0)):
        def pop(shift_deg):
            out = []
            for _ in range(M):
                v = rng.normal(size=D); v -= v.mean()
                v -= (v @ base) * base; v /= np.linalg.norm(v)
                th = np.radians(shift_deg + rng.normal() * spread_deg)
                out.append(np.cos(th) * base + np.sin(th) * v)
            return np.array(out)
        A, B = pop(0.0), pop(15.0)
        br = [log_bc_bridge(A, B, kappa, p, 8000, np.random.default_rng(s)) for s in range(6)]
        od = [vmf_utils.log_bc_mc(A, B, kappa, p, 8000, np.random.default_rng(s))
              for s in range(6)]
        print(f"      spread={spread_deg:>4.0f}deg  bridge logBC {np.mean(br):+8.3f} "
              f"sd {np.std(br):.4f}   |   old {np.mean(od):+8.3f} sd {np.std(od):.4f}"
              f"   ratio {np.std(od)/max(np.std(br),1e-12):>6.1f}x")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("bc_bridge validation\n")
    w1 = _check_single_component(rng)
    w2 = _check_identity(rng)
    _check_vs_old(rng)
    print(f"\n  VERDICT: closed-form worst err/se {w1:.1f}; identity worst {w2:.1e}")
