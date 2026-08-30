"""
von Mises-Fisher utilities for the LayerNorm-channel bitrate framework.

Reads live on S^{D-2} inside the mean-zero hyperplane of R^D (LayerNorm output:
mean 0, RMS 1). For vMF on that sphere the ambient dimension is p = D-1.

We MODEL each noisy read as vMF(mu, kappa):
  - kappa  = concentration = the rate knob (high kappa = low noise = high rate),
  - rate   = KL(vMF(kappa) || uniform)                    [exact, closed form],
  - rho    = mean resultant length = A_p(kappa) = I_{p/2}(kappa)/I_{p/2-1}(kappa),
  - the actual sampling is add-Gaussian-then-reLN with sigma matched so the mean
    read-cosine equals rho, i.e. sigma = sqrt(1/rho^2 - 1).

Pairwise similarity between two point-posteriors is the vMF Bhattacharyya
coefficient, which factorizes through the cosine mu_i^T mu_j and the concentrations.

All heavy special functions use scipy (CPU); training uses precomputed 1-D maps.
"""

import numpy as np
from scipy.special import ive, gammaln


def log_C(kappa, p):
    """log normalizer of vMF on S^{p-1}: C_p(k)=k^{p/2-1}/((2pi)^{p/2} I_{p/2-1}(k))."""
    kappa = np.asarray(kappa, dtype=np.float64)
    nu = p / 2.0 - 1.0
    logC0 = gammaln(p / 2.0) - np.log(2.0) - (p / 2.0) * np.log(np.pi)  # k->0 limit
    small = kappa < 1e-8
    ksafe = np.where(small, 1.0, kappa)
    log_I = np.log(ive(nu, ksafe)) + ksafe                     # log I_{nu}(k)
    res = (p / 2.0 - 1.0) * np.log(ksafe) - (p / 2.0) * np.log(2.0 * np.pi) - log_I
    return np.where(small, logC0, res)


def A_p(kappa, p):
    """Mean resultant length rho = I_{p/2}(k)/I_{p/2-1}(k) (in [0,1))."""
    kappa = np.asarray(kappa, dtype=np.float64)
    return ive(p / 2.0, kappa) / ive(p / 2.0 - 1.0, kappa)


def rate_kl(kappa, p):
    """KL(vMF(mu,kappa) || uniform) in nats."""
    logC0 = gammaln(p / 2.0) - np.log(2.0) - (p / 2.0) * np.log(np.pi)
    return kappa * A_p(kappa, p) + log_C(kappa, p) - logC0


def sigma_from_kappa(kappa, p):
    """Gaussian-then-reLN sigma matched to vMF(kappa): mean cosine = rho."""
    rho = np.clip(A_p(kappa, p), 1e-8, 1 - 1e-12)
    return np.sqrt(np.clip(1.0 / rho**2 - 1.0, 0.0, None))


def log_bhattacharyya(kappa1, kappa2, cos, p):
    """log Bhattacharyya coefficient between vMF(mu1,k1) and vMF(mu2,k2),
    where cos = mu1^T mu2. Factorizes through cos and the concentrations."""
    km = 0.5 * np.sqrt(kappa1**2 + kappa2**2 + 2.0 * kappa1 * kappa2 * cos)
    return 0.5 * (log_C(kappa1, p) + log_C(kappa2, p)) - log_C(km, p)


def build_rate_sigma_maps(p, kappa_lo=5.0, kappa_hi=1e6, n=4000):
    """Precompute monotone grids for use as differentiable 1-D maps in training:
    returns dict with kappa, rate, sigma (all monotone). kappa_lo default 5 avoids
    Bessel underflow at high p (kappa=5 already gives sigma~38 = essentially pure noise)."""
    kappa = np.geomspace(kappa_lo, kappa_hi, n)
    out = {
        "kappa": kappa,
        "rho": A_p(kappa, p),
        "rate": rate_kl(kappa, p),            # increasing in kappa
        "sigma": sigma_from_kappa(kappa, p),  # decreasing in kappa
    }
    for k, v in out.items():
        assert np.all(np.isfinite(v)), f"non-finite in map '{k}' (raise kappa_lo)"
    assert np.all(np.diff(out["rate"]) > 0), "rate not strictly increasing"
    return out


# ---- sampling (Wood/Ulrich) -----------------------------------------------
# `sample_mixture` is on the hot path of the per-head Q/K/V analysis: it draws the bridge
# distribution inside bhat_qkv.bc_projected, and accounts for roughly 70% of that function's
# runtime, so it dominates the cost of a full layers-by-heads scan. It is also used by
# bc_bridge.py for mixture BC. It is NOT validation-only -- do not remove it as dead code
# on the strength of `_validate` being the only in-module caller.
#
# The sampler is used in preference to a closed-form expected-likelihood kernel (exponent 1
# instead of BC's 0.5, bilinear, so it distributes over mixture sums with no sampling at
# all): that alternative systematically under-reads when high kappa meets moderate true
# similarity -- enough to change a qualitative conclusion, not just a magnitude -- and its
# O(M^2) elementwise Bessel evaluation measured 2.2-2.5x SLOWER in practice than sampling,
# which needs no Bessel calls to sample and only cheap scalar log_C calls to evaluate.

def _wood_sample_w(kappa, p, n, rng):
    """Wood/Ulrich rejection sampler for the mu-aligned coordinate w."""
    b = (-2.0 * kappa + np.sqrt(4.0 * kappa**2 + (p - 1) ** 2)) / (p - 1)
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (p - 1) * np.log(1.0 - x0**2)
    out = np.empty(n)
    filled = 0
    while filled < n:
        m = n - filled
        z = rng.beta((p - 1) / 2.0, (p - 1) / 2.0, size=m)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = rng.uniform(size=m)
        acc = kappa * w + (p - 1) * np.log(1.0 - x0 * w) - c >= np.log(u)
        k = int(acc.sum())
        out[filled:filled + k] = w[acc][:k]
        filled += k
    return out


def sample_mixture(mus, kappa, p, n, rng):
    """n samples from the flat mixture (1/M) sum_i vMF(mu_i, kappa): pick a component
    uniformly per sample, then draw a real vMF sample from it (Wood/Ulrich for the
    mu-aligned coordinate w, plus a uniform direction on the orthogonal (p-2)-sphere within
    the mean-zero hyperplane). mus: [M, D] (D = p+1, e.g. GPT2's LayerNorm-output
    dimension) -- normalized to unit L2 norm internally. Returns [n, D] unit vectors."""
    mus = mus / (np.linalg.norm(mus, axis=-1, keepdims=True) + 1e-12)
    M, D = mus.shape
    idx = rng.integers(0, M, size=n)
    mu = mus[idx]                                              # [n, D]
    w = _wood_sample_w(kappa, p, n, rng)                       # [n]
    z = rng.standard_normal((n, D))
    z = z - z.mean(axis=-1, keepdims=True)                     # project to mean-zero hyperplane
    z = z - (z * mu).sum(-1, keepdims=True) * mu               # project out the mu component
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)
    return w[:, None] * mu + np.sqrt(np.clip(1.0 - w**2, 0.0, None))[:, None] * z


def mixture_log_density(x, mus, kappa, p):
    """log density of the mixture (1/M) sum_i vMF(mu_i,kappa) at each row of x [n,D].
    mus: [M,D] (normalized internally). Returns [n]."""
    from scipy.special import logsumexp
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
    mus = mus / (np.linalg.norm(mus, axis=-1, keepdims=True) + 1e-12)
    cos = np.clip(x @ mus.T, -1.0, 1.0)                        # [n, M]
    log_terms = log_C(kappa, p) + kappa * cos
    return logsumexp(log_terms, axis=1) - np.log(mus.shape[0])


def log_bc_mc(mus_a, mus_b, kappa, p, n_mc, rng):
    """Log-space symmetrized Monte Carlo estimate of log(Bhattacharyya coefficient) between
    two flat mixtures (1/Ma) sum vMF(mu_i^a,kappa) and (1/Mb) sum vMF(mu_j^b,kappa), sharing
    kappa. Returns log(BC) directly (equivalently, -1 * Bhattacharyya distance) -- NEVER
    exponentiate-then-clip, since the true value can be exp(-hundreds) at high kappa or
    diffuse-population taps: a naive linear-space computation can't distinguish "genuinely
    near 1" from "underflowed something modest" from "underflowed something
    astronomically small" -- this can.
    BC(p,q) = integral[sqrt(p*q)] = E_{x~p}[sqrt(q(x)/p(x))] = E_{x~q}[sqrt(p(x)/q(x))];
    averaging both directions (in log-space, via logsumexp of the two log-estimates) reduces
    variance versus either alone."""
    xa = sample_mixture(mus_a, kappa, p, n_mc, rng)
    xb = sample_mixture(mus_b, kappa, p, n_mc, rng)
    from scipy.special import logsumexp
    log_pa_xa = mixture_log_density(xa, mus_a, kappa, p)
    log_pb_xa = mixture_log_density(xa, mus_b, kappa, p)
    log_pa_xb = mixture_log_density(xb, mus_a, kappa, p)
    log_pb_xb = mixture_log_density(xb, mus_b, kappa, p)
    log_est_a = logsumexp(0.5 * (log_pb_xa - log_pa_xa)) - np.log(n_mc)
    log_est_b = logsumexp(0.5 * (log_pa_xb - log_pb_xb)) - np.log(n_mc)
    return float(logsumexp([log_est_a, log_est_b]) - np.log(2))

def _validate():
    import torch
    D = 384                     # ViT-Small LayerNorm width
    p = D - 1
    rng = np.random.default_rng(0)
    print(f"[vmf] p={p} (sphere S^{p-1}, DOF={p-1})")

    kappas = [5.0, 20.0, 80.0, 300.0, 2000.0]
    print("\n kappa     rho=A_p     KL(closed)   KL(MC)     sigma    emp.cos(G+reLN)")
    for k in kappas:
        rho = float(A_p(k, p))
        kl_closed = float(rate_kl(k, p))
        # MC KL: E_z[log p_vMF(z) - log p_unif] = kappa*E[w] + logC(k) - logC(0)
        w = _wood_sample_w(k, p, 20000, rng)
        logC0 = gammaln(p / 2.0) - np.log(2.0) - (p / 2.0) * np.log(np.pi)
        kl_mc = float(k * w.mean() + log_C(k, p) - logC0)
        # empirical mean cosine of the actual training channel (add-Gaussian-then-reLN),
        # using the real LayerNorm convention: RMS 1 (norm sqrt(D)), cosine = <a,b>/D.
        sig = float(sigma_from_kappa(k, p))
        def ln(t):
            t = t - t.mean(-1, keepdim=True)
            return t / t.std(-1, unbiased=False, keepdim=True)
        xhat = ln(torch.randn(4000, D))
        yhat = ln(xhat + sig * torch.randn_like(xhat))
        emp_cos = float((xhat * yhat).sum(-1).mean() / D)
        print(f" {k:7.0f}   {rho:.4f}    {kl_closed:9.3f}   {kl_mc:9.3f}   {sig:.4f}   {emp_cos:.4f}")

    # bits per tap sanity (rate in bits = KL/ln2 * ... here KL already total for the read)
    print("\n rate (bits) for kappa above:", [round(float(rate_kl(k, p)) / np.log(2), 1) for k in kappas])
    # Bhattacharyya sanity: BC(k,k,cos=1)=1; decreasing in angle
    for cos in [1.0, 0.9, 0.5, 0.0]:
        bc = float(np.exp(log_bhattacharyya(80.0, 80.0, cos, p)))
        print(f" BC(k=80,k=80,cos={cos}) = {bc:.4f}")
    # sweep from the same floor build_rate_sigma_maps uses: below it the Bessel term
    # underflows to exactly 0 at large p (see safe_kappa_lo in train_stoch_layernorm_gpt.py) and the
    # monotonicity test would be reading nan/-inf rather than the real curve.
    kgrid = np.geomspace(5.0, 1e5, 200)
    print("\n monotonicity: rate increasing?",
          bool(np.all(np.diff(rate_kl(kgrid, p)) > 0)),
          "| sigma decreasing?",
          bool(np.all(np.diff(sigma_from_kappa(kgrid, p)) < 0)))


if __name__ == "__main__":
    _validate()
