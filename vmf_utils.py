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
Bessel terms fall back to a log-space power series where scipy's ive underflows (log_iv),
so the maps extend down to near-zero rate at any p. channel_rate_mc is an independent,
sample-based check of the rate column against the actual noise channel.
"""

import numpy as np
from scipy.special import ive, gammaln, logsumexp


# scipy's exponentially scaled ive(v, k) underflows to exactly 0 once k << v at large v
# (GPT-2 small, v=382.5: k < ~76). Below this value log_iv switches to the power series.
# 1e-300 sits just above the denormal range, so every value the old scipy-only code could
# represent at full precision stays on the scipy path (old tables are bit-identical).
_IVE_FLOOR = 1e-300


def log_iv(v, kappa):
    """log I_v(kappa) in float64, with no underflow floor.

    Where ive(v, k) > _IVE_FLOOR this is log(ive(v, k)) + k, bit-identical to the old
    scipy-only path. Below that it sums the power series
        I_v(k) = sum_j (k/2)^(2j+v) / (j! Gamma(v+j+1))
    in log space. Terms peak near j = k^2 / (4(v+1)), which is small exactly where ive
    underflows (k << v), so few terms are needed; truncation is asserted rather than assumed.
    Agrees with ive to ~10 digits where both are finite."""
    kappa = np.asarray(kappa, dtype=np.float64)
    k = kappa.reshape(-1)
    s = ive(v, k)
    with np.errstate(divide="ignore"):
        out = np.log(s) + k
    idx = np.flatnonzero(~(s > _IVE_FLOOR) & (k > 0))
    for c in range(0, idx.size, 16384):                        # bound the [rows, terms] matrix
        ii = idx[c:c + 16384]
        out[ii] = _log_iv_series(v, k[ii])
    return out.reshape(kappa.shape)


def _log_iv_series(v, k):
    """log I_v(k) for 1-D k > 0 by the power series in log space (see log_iv)."""
    jstar = k.max() ** 2 / (4.0 * (v + 1.0))
    j = np.arange(int(np.ceil(jstar + 40.0 * np.sqrt(jstar + 1.0) + 60.0)), dtype=np.float64)
    log_terms = (j[None, :] * np.log(k[:, None] ** 2 / 4.0)
                 - gammaln(j + 1.0)[None, :] - gammaln(v + 1.0 + j)[None, :])
    log_sum = logsumexp(log_terms, axis=1)
    assert np.all(log_terms[:, -1] < log_sum - 40.0), "log_iv: power series truncated"
    return v * np.log(k / 2.0) + log_sum


def log_C(kappa, p):
    """log normalizer of vMF on S^{p-1}: C_p(k)=k^{p/2-1}/((2pi)^{p/2} I_{p/2-1}(k))."""
    kappa = np.asarray(kappa, dtype=np.float64)
    nu = p / 2.0 - 1.0
    logC0 = gammaln(p / 2.0) - np.log(2.0) - (p / 2.0) * np.log(np.pi)  # k->0 limit
    small = kappa < 1e-8
    ksafe = np.where(small, 1.0, kappa)
    log_I = log_iv(nu, ksafe)                                  # log I_{nu}(k)
    res = (p / 2.0 - 1.0) * np.log(ksafe) - (p / 2.0) * np.log(2.0 * np.pi) - log_I
    return np.where(small, logC0, res)


def A_p(kappa, p):
    """Mean resultant length rho = I_{p/2}(k)/I_{p/2-1}(k) (in [0,1))."""
    kappa = np.asarray(kappa, dtype=np.float64)
    num, den = ive(p / 2.0, kappa), ive(p / 2.0 - 1.0, kappa)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = num / den
    ok = (num > _IVE_FLOOR) & (den > _IVE_FLOOR)
    if np.all(ok):
        return rho                                             # unchanged scipy path
    return np.where(ok, rho, np.exp(log_iv(p / 2.0, kappa) - log_iv(p / 2.0 - 1.0, kappa)))


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


def build_rate_sigma_maps(p, kappa_lo=0.5, kappa_hi=1e6, n=4000):
    """Precompute monotone grids for use as differentiable 1-D maps in training:
    returns dict with kappa, rate, sigma (all monotone).

    kappa_lo sets the table floor. Below kappa ~ p/10 the rate is kappa^2/(2p) = p/(2 sigma^2)
    nats, so kappa_lo=0.5 gives floor rates of 6.5e-4 (p=191), 3.3e-4 (p=383) and 1.6e-4
    (p=767, sigma ~1500). That matters because RateBudgetPolicy.sigmas() clamps rates below
    the grid: such taps are charged ~0 but run at the floor rate, so the floor bounds that
    unbilled leak. Before log_iv existed the floor had to dodge Bessel underflow: 5.0 for the
    ViTs, safe_kappa_lo(p) for GPT-2 (76.3 at p=767, a 3.74-nat/tap leak). Checkpoints store
    their grids as policy buffers, so old ones still load with their original table.
    Keep n=4000: it is the buffer length, and changing it breaks load_state_dict on
    existing checkpoints."""
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
# Originally validation-only; also used by marginal_bc_probe.py's bc_mc, which estimates
# the Bhattacharyya coefficient between two "flat mixture of posteriors" (one vMF per real
# stochastic redraw at a tap, sharing the tap's own kappa) via Monte Carlo. A closed-form
# alternative (expected-likelihood kernel, exponent 1 instead of BC's 0.5, bilinear so it
# distributes over mixture sums with no sampling) was tried and dropped: it systematically
# under-reads at elevated kappa combined with moderate true similarity (one tap read 0.05
# vs BC's 0.57 there -- a different qualitative conclusion, not just a magnitude gap), and
# turned out not even to be faster in practice (its O(M^2) elementwise Bessel evaluation
# was ~2.2-2.5x SLOWER than this sampling-based approach, which needs no Bessel calls at
# all for sampling and only cheap scalar log_C calls for density evaluation).

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
    exponentiate-then-clip, since the true value can be exp(-hundreds) at high kappa/
    diffuse-population taps (see result-marginal-bc-bandwidth-failure memory): a naive
    linear-space computation can't distinguish "genuinely near 1" from "underflowed
    something modest" from "underflowed something astronomically small" -- this can.
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


def bc_mc(mus_a, mus_b, kappa, p, n_mc, rng):
    """Linear-space BC -- see log_bc_mc's docstring for why the log-space version should be
    preferred whenever the true value might be extremely small (it usually should be)."""
    return float(np.exp(log_bc_mc(mus_a, mus_b, kappa, p, n_mc, rng)))


def channel_rate_mc(sigma, D, n=2_000_000_000, n_bins=4000, chunk=100_000_000, device=None, seed=0):
    """Sample-based rate (nats) of the ACTUAL training channel, x -> normalize(x + sigma*randn),
    as an independent check on rate_kl(kappa) at kappa matched to sigma. Returns (kl, se).

    Everything is rotationally symmetric about mu, so KL(channel || uniform) on S^{p-1}
    equals the 1-D KL of w = cos(read, mu):
      - channel: 2 scalars per draw. The clean read has norm sqrt(D); the mean-centred noise
        is isotropic in the p = D-1 dim hyperplane, so w = s / sqrt(s^2 + sigma^2 c) with
        s = sqrt(D) + sigma*z, z ~ N(0,1), c ~ chi2(p-1).
      - uniform: (w+1)/2 ~ Beta((p-1)/2, (p-1)/2). Bin probabilities are computed in LOG
        space (Simpson on the log density), because cdf differences round to 0 in the tail.
    Histogram plug-in KL with a Miller-Madow correction. Relative se ~ sqrt(2/(KL*n)), and
    the correction is ~n_bins/(2n) nats, so trust it down to rates of ~1e3 * n_bins/n.
    Checked (2e9 draws) against exact 1-D quadrature of the channel density: within 1e-5
    relative for sigma 2-100 at D=768, and the 2-scalar draw matches the real 768-D op."""
    import math
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    p = D - 1
    gen = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)
    chi2 = torch.distributions.Chi2(torch.tensor(float(p - 1), device=device, dtype=torch.float64))

    def draw(m):
        s = math.sqrt(D) + sigma * torch.randn(m, device=device, dtype=torch.float64, generator=gen)
        return s / torch.sqrt(s * s + sigma * sigma * chi2.sample((m,)))

    pilot = draw(1_000_000)                      # window = pilot range padded by 50% each side
    lo_p, hi_p = pilot.min().item(), pilot.max().item()
    lo, hi = max(-1.0, lo_p - 0.5 * (hi_p - lo_p)), min(1.0, hi_p + 0.5 * (hi_p - lo_p))
    edges = torch.linspace(lo, hi, n_bins + 1, device=device, dtype=torch.float64)
    counts = torch.zeros(n_bins + 2, device=device, dtype=torch.float64)
    for c in range(0, n, chunk):
        idx = torch.bucketize(draw(min(chunk, n - c)), edges)
        counts += torch.bincount(idx, minlength=n_bins + 2).double()
    counts = counts.cpu().numpy()
    assert counts[0] == 0 and counts[-1] == 0, "channel_rate_mc: draws fell outside the window"
    q = counts[1:-1] / n
    ed = edges.cpu().numpy()
    t = ed[:-1, None] + (ed[1:] - ed[:-1])[:, None] * np.linspace(0.0, 1.0, 65)[None, :]
    log_f = (gammaln(p / 2.0) - 0.5 * np.log(np.pi) - gammaln((p - 1) / 2.0)
             + (p - 3) / 2.0 * np.log1p(-np.clip(t * t, 0.0, 1.0 - 1e-16)))
    wts = np.ones(65); wts[1:-1:2] = 4.0; wts[2:-1:2] = 2.0
    log_u = logsumexp(log_f + np.log(wts)[None, :], axis=1) + np.log((ed[1:] - ed[:-1]) / 64 / 3)
    m = q > 0
    lr = np.log(q[m]) - log_u[m]
    kl = float(np.sum(q[m] * lr))
    se = math.sqrt(max(float(np.sum(q[m] * lr ** 2)) - kl ** 2, 0.0) / n)
    return kl - (int(m.sum()) - 1) / (2.0 * n), se


def _validate():
    import torch
    D = 192
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
    print("\n monotonicity: rate increasing?",
          bool(np.all(np.diff(rate_kl(np.geomspace(0.5, 1e5, 400), p)) > 0)),
          "| sigma decreasing?",
          bool(np.all(np.diff(sigma_from_kappa(np.geomspace(0.5, 1e5, 400), p)) < 0)))

    # low-kappa extension: series/scipy crossover, table floor, and the sample-based rate check
    k_x = np.geomspace(2.0, 400.0, 400)
    k_x = k_x[ive(p / 2.0 - 1.0, k_x) > 1e-200]                 # where scipy is still finite
    err = np.abs(_log_iv_series(p / 2.0 - 1.0, k_x) - (np.log(ive(p / 2.0 - 1.0, k_x)) + k_x))
    print(f"\n log I_nu series vs scipy over kappa [{k_x[0]:.1f}, {k_x[-1]:.0f}]: max |diff| = {err.max():.1e}")
    maps = build_rate_sigma_maps(p)
    print(f" table floor: kappa={maps['kappa'][0]:.2f} sigma={maps['sigma'][0]:.1f} "
          f"rate={maps['rate'][0]:.3e} nats (kappa^2/(2p)={maps['kappa'][0]**2/(2*p):.3e})")
    try:
        import torch
        have_cuda = torch.cuda.is_available()
    except ImportError:
        have_cuda = False
    if have_cuda:
        print(" channel_rate_mc (2e9 draws) vs rate_kl:")
        for k in [maps["kappa"][0], 5.0, 80.0]:
            sig = float(sigma_from_kappa(k, p))
            kl, se = channel_rate_mc(sig, p + 1)
            print(f"   kappa={k:6.2f} sigma={sig:8.2f}: MC={kl:.4e} +- {se:.1e}  rate_kl={float(rate_kl(k, p)):.4e}"
                  f"  ratio={kl / float(rate_kl(k, p)):.4f}")
    else:
        print(" (no CUDA: skipping channel_rate_mc)")



if __name__ == "__main__":
    _validate()
