"""Bayesian (MAP) estimation of (A0, half-life, background) from binned counting data.

Method summary
--------------
1. Find the maximum a posteriori (MAP) point by direct numerical optimization of the
   negative log-posterior (log-likelihood + priors), restarted from a few candidate
   starting points. This scales as O(grid-free), independent of the number of counts
   or channels beyond a single pass per optimizer evaluation -- unlike a brute-force
   grid search, it does not degrade as the posterior becomes very narrow (which it does
   as the count number grows).
2. Build a small grid LOCAL to the MAP (already well-centred, so no fragile iterative
   grid search is needed to locate it) to get the posterior covariance by direct
   second-moment integration -- exact given the grid, no Gaussian/Laplace
   approximation, and robust even when a parameter (typically background) sits at its
   physical floor.
3. Smooth the displayed marginal posteriors via importance sampling: draw from a
   Gaussian proposal matching the covariance (so draws concentrate along whatever
   correlation ridge the posterior has), re-weight by the true posterior density, and
   summarize with a weighted KDE / weighted quantiles. This also gives a non-parametric
   (non-Gaussian, correctly skewed where relevant) 95% credible interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, multivariate_normal, norm
from tqdm import tqdm

from .model import expected_observed_counts, loglinear_fit
from .priors import Priors, log_prior_background, log_prior_gaussian, log_prior_student_t


@dataclass
class FitConfig:
    dead_time_model: str = "nonparalyzable"   # or "paralyzable"
    n_checkpoints: int = 20       # number of points in the convergence trace
    n_vis: int = 61               # local covariance grid resolution per dimension
    edge_mass_tol: float = 1e-4   # local-grid widening stop criterion
    max_widen: int = 10           # max local-grid widen/shrink attempts
    min_effective_fraction: float = 0.25  # local-grid shrinking stop criterion (see fit())
    n_is_samples: int = 100_000   # importance-sampling draws for marginal smoothing
    quadrature_points: int = 1    # >1 corrects the per-channel dead-time averaging bias
    # (see model.expected_observed_counts); 1 = today's exact behaviour, no extra cost.
    priors: Priors = field(default_factory=Priors)


@dataclass
class FitResult:
    t_start: np.ndarray
    t_end: np.ndarray
    bin_width: np.ndarray
    counts: np.ndarray

    A0_loglinear: float
    half_life_loglinear: float
    sigma_A0_loglinear: float
    sigma_half_life_loglinear: float

    A0: float
    half_life: float
    background: float
    cov: np.ndarray  # 3x3, order (A0, half_life, background)
    corr: np.ndarray

    trace_t: np.ndarray
    trace_A0: np.ndarray
    trace_half_life: np.ndarray
    trace_background: np.ndarray

    A0_grid: np.ndarray
    A0_marginal: np.ndarray
    half_life_grid: np.ndarray
    half_life_marginal: np.ndarray
    background_grid: np.ndarray
    background_marginal: np.ndarray

    A0_ci95: tuple
    half_life_ci95: tuple
    background_ci95: tuple

    converged: bool
    dead_time_model: str

    @property
    def u_A0(self):
        return float(np.sqrt(self.cov[0, 0]))

    @property
    def u_half_life(self):
        return float(np.sqrt(self.cov[1, 1]))

    @property
    def u_background(self):
        return float(np.sqrt(self.cov[2, 2]))

    @property
    def decay_constant(self):
        return np.log(2) / self.half_life

    @property
    def u_decay_constant(self):
        """Propagated (GUM law of propagation of uncertainty) from u(half_life)."""
        d_lam_d_t12 = -np.log(2) / self.half_life**2
        return abs(d_lam_d_t12) * self.u_half_life

    def convergence_cutoff(self, n_sigma=3.0):
        """Where the convergence trace (``trace_t``/``trace_A0``/...) has stabilized:
        the earliest point from which EVERY later checkpoint stays within
        ``n_sigma`` posterior standard deviations of the final estimate, for each of
        A0, half-life, and background. Returns a dict with a time (in ``trace_t``'s
        units, ``None`` if that parameter never stabilizes within the observed trace)
        per parameter, plus ``"overall"`` = the latest of the three (the point by
        which all three have settled).

        This is a DISPLAY marker, not a data-truncation point. Unlike MCMC burn-in,
        every checkpoint here is fit from genuinely valid data -- just a growing
        amount of it -- so early, higher-variance points are not "wrong" samples to
        discard before refitting; for this problem the earliest, highest-rate data is
        usually the most informative for A0, so cutting it would typically only
        throw that away rather than improve anything.
        """
        def cutoff_for(trace, final_value, sigma):
            if sigma <= 0:
                return float(self.trace_t[0])
            within = np.abs(np.asarray(trace) - final_value) <= n_sigma * sigma
            bad = np.where(~within)[0]
            if len(bad) == 0:
                return float(self.trace_t[0])
            last_bad = bad[-1]
            if last_bad + 1 >= len(self.trace_t):
                return None
            return float(self.trace_t[last_bad + 1])

        per_param = {
            "A0": cutoff_for(self.trace_A0, self.A0, self.u_A0),
            "half_life": cutoff_for(self.trace_half_life, self.half_life, self.u_half_life),
            "background": cutoff_for(self.trace_background, self.background, self.u_background),
        }
        settled = [t for t in per_param.values() if t is not None]
        per_param["overall"] = max(settled) if settled else None
        return per_param


def _log_prior_B(B, priors):
    """Background prior: Gaussian centred on ``priors.background_prior_mean`` when
    that informative override is set (see Priors.informative()), else the default
    exponential(``b_prior_scale``) -- mode at B=0, "probably negligible"."""
    if priors.background_prior_mean is not None and priors.background_prior_sigma is not None:
        return log_prior_gaussian(B, priors.background_prior_mean, priors.background_prior_sigma)
    return log_prior_background(B, priors.b_prior_scale)


def _prior_mean_sigma(informative_mean, informative_sigma, fallback_mean, fallback_sigma):
    """Resolve one parameter's effective prior (mean, sigma): the informative
    override from Priors.informative() when both its mean and sigma are set, else
    the weakly-informative fallback (the data's own log-linear fit, widened)."""
    if informative_mean is not None and informative_sigma is not None:
        return informative_mean, informative_sigma
    return fallback_mean, fallback_sigma


def neg_log_posterior(
    params, t_s, t_e, n, tau_d, dead_time_model, quadrature_points,
    A0_mean, A0_sigma, t12_mean, t12_sigma, priors,
):
    A0, half_life, B = params
    if A0 <= 0 or half_life <= 0 or B < 0:
        return np.inf
    lam = np.log(2) / half_life
    mu = expected_observed_counts(A0, lam, B, t_s, t_e, tau_d, model=dead_time_model, n_quad=quadrature_points)
    # Floor away from exactly 0: with the paralyzable model (which is not monotonic in
    # the true rate, see deadtime.py) the optimizer's numerical-gradient probing can
    # occasionally land on a combination that underflows mu to 0.0, which would
    # otherwise raise a RuntimeWarning on log(mu) without changing the result.
    mu = np.maximum(mu, 1e-300)
    neg_loglik = -np.sum(n * np.log(mu) - mu)
    log_prior_total = (
        log_prior_student_t(A0, A0_mean, A0_sigma, priors.prior_df)
        + log_prior_student_t(half_life, t12_mean, t12_sigma, priors.prior_df)
        + _log_prior_B(B, priors)
    )
    return neg_loglik - log_prior_total


def _fit_map(x0, args, bounds):
    """Multi-start MAP search: with strongly correlated parameters (e.g. an
    acquisition time on the order of one half-life), a single local optimizer run can
    get stuck on a near-flat likelihood plateau near its starting point. We restart
    from several plausible background guesses AND, since A0 and half-life trade off
    against each other along that same near-flat ridge, from a couple of points
    nudged in opposite directions along it too -- varying only B is not always enough
    to escape it (a weakly-informative prior, by design, does not do much of that
    escaping for us). Keep whichever restart reaches the lowest negative-log-posterior."""
    A0_0, t12_0, B_0 = x0
    candidates = [
        (A0_0, t12_0, B_0),
        (A0_0, t12_0, 0.1 * A0_0),
        (A0_0, t12_0, 0.02 * A0_0),
        (A0_0 * 1.1, t12_0 * 0.9, B_0),
        (A0_0 * 0.9, t12_0 * 1.1, B_0),
    ]
    return min(
        (minimize(neg_log_posterior, list(c), args=args, method="L-BFGS-B", bounds=bounds) for c in candidates),
        key=lambda r: r.fun,
    )


def _numerical_hessian(f, x0, h):
    """Central-difference Hessian of scalar function ``f`` at ``x0`` (length-n),
    with per-dimension step sizes ``h``. O(n^2) evaluations of ``f`` -- only ever
    called here with n in {2, 3}, negligible next to a single local-grid pass."""
    n = len(x0)
    x0 = np.asarray(x0, dtype=float)
    f0 = f(x0)
    H = np.zeros((n, n))
    for i in range(n):
        xp, xm = x0.copy(), x0.copy()
        xp[i] += h[i]
        xm[i] -= h[i]
        H[i, i] = (f(xp) - 2 * f0 + f(xm)) / h[i] ** 2
    for i in range(n):
        for j in range(i + 1, n):
            xpp, xpm, xmp, xmm = x0.copy(), x0.copy(), x0.copy(), x0.copy()
            xpp[i] += h[i]; xpp[j] += h[j]
            xpm[i] += h[i]; xpm[j] -= h[j]
            xmp[i] -= h[i]; xmp[j] += h[j]
            xmm[i] -= h[i]; xmm[j] -= h[j]
            H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h[i] * h[j])
    return H


def _laplace_sigma(A0_final, t12_final, B_final, args_full):
    """Cheap (a handful of neg_log_posterior evaluations -- negligible next to a
    single local-grid pass) local quadratic (Laplace) approximation of the
    marginal standard deviations at the MAP, from the numerical Hessian of
    neg_log_posterior.

    Used ONLY to size the local covariance grid's window well -- the grid's own
    second-moment integration remains the actual reported uncertainty. Without
    this, the window is sized from a fixed fraction of the MAP value, which at
    very high statistics can be orders of magnitude wider than the true
    posterior, leaving the grid so coarse that almost all its mass piles onto a
    single cell -- underestimating the reported uncertainty despite the window
    safely containing the posterior (see the git history for this function's
    introduction for the full diagnosis).

    B often sits at its physical floor, where a quadratic approximation doesn't
    apply (a central difference would probe B < 0, where neg_log_posterior is
    +inf) -- in that case only (A0, half-life) are sized this way, with B held
    fixed at its MAP value, and the caller falls back to its own heuristic for
    B's window. But when B is well away from the floor AND the statistics are
    high enough, B can be constrained to a width far narrower than that fixed
    heuristic expects (e.g. a background of several hundred cps resolved to a
    fraction of a cps): leaving its window badly oversized relative to the
    others then corrupts A0/half-life's marginals too, since marginalizing over
    a badly-under-resolved, strongly-correlated B axis drags them down with it
    (this is what the 3-parameter Hessian below is for -- sizing all three
    windows from one consistent quadratic approximation when it is safe to).

    Returns ``None`` if the Hessian isn't usable (not finite / not positive
    definite), in which case the caller falls back to the old fixed-fraction
    windows entirely -- still safe, just potentially needing more widen passes.
    Otherwise returns a 2-tuple ``(sigma_A0, sigma_t12)`` (B at its floor) or a
    3-tuple ``(sigma_A0, sigma_t12, sigma_B)``.
    """
    def neg_post(params):
        return neg_log_posterior(list(params), *args_full)

    h_A0 = max(abs(A0_final) * 1e-4, 1e-3)
    h_t12 = max(abs(t12_final) * 1e-4, 1e-6)
    h_B = max(abs(B_final) * 1e-4, 1e-3)

    n_dims = 2 if B_final <= h_B else 3
    x0 = [A0_final, t12_final] if n_dims == 2 else [A0_final, t12_final, B_final]
    h = [h_A0, h_t12] if n_dims == 2 else [h_A0, h_t12, h_B]
    f = (lambda p: neg_post([p[0], p[1], B_final])) if n_dims == 2 else neg_post

    try:
        H = _numerical_hessian(f, x0, h)
        variances = np.diag(np.linalg.inv(H))
    except np.linalg.LinAlgError:
        return None
    # A non-positive-definite Hessian (a negative "variance") is an expected,
    # ordinary outcome here -- not every MAP is a clean quadratic bowl in finite
    # differences -- so check for it before sqrt rather than let it warn.
    if not np.all(np.isfinite(variances)) or np.any(variances <= 0):
        return None
    sigma = np.sqrt(variances)
    return tuple(sigma)


def _weighted_quantile(samples_1d, weights, q):
    order = np.argsort(samples_1d)
    s, w = samples_1d[order], weights[order]
    cum = np.cumsum(w)
    cum /= cum[-1]
    return np.interp(q, cum, s)


def _weighted_marginal(samples_1d, grid, weights):
    keep = weights > 0
    kde = gaussian_kde(samples_1d[keep], weights=weights[keep])
    return kde(grid)


def fit(bin_edges, counts, tau_d, config=None, show_progress=True):
    """Run the full Bayesian estimation pipeline on binned counting data.

    Parameters
    ----------
    bin_edges : array_like, shape (n_bins + 1,)
    counts : array_like, shape (n_bins,)
    tau_d : float
        Dead time (same time unit as ``bin_edges``).
    config : FitConfig, optional
    show_progress : bool
        Whether to display tqdm progress bars for the expensive loops.

    Returns
    -------
    FitResult
    """
    config = config or FitConfig()
    priors = config.priors
    bounds = [(1e-3, None), (1e-3, None), (0.0, None)]

    t_start, t_end = bin_edges[:-1], bin_edges[1:]
    bin_width = t_end - t_start
    bin_centers = 0.5 * (t_start + t_end)
    n_bins = len(counts)

    # -- Log-linear reference fit + its use as prior location/width -----------------
    # (unless Priors.informative() overrides a given parameter -- see _prior_mean_sigma)
    A0_lin, t12_lin, B_lin, sigma_A0_lin, sigma_t12_lin = loglinear_fit(
        bin_centers, counts, bin_width, bin_edges[-1], tau_d=tau_d, dead_time_model=config.dead_time_model
    )
    A0_prior_mean, A0_prior_sigma = _prior_mean_sigma(
        priors.A0_prior_mean, priors.A0_prior_sigma, A0_lin, priors.prior_widen_k * sigma_A0_lin
    )
    t12_prior_mean, t12_prior_sigma = _prior_mean_sigma(
        priors.half_life_prior_mean, priors.half_life_prior_sigma, t12_lin, priors.prior_widen_k * sigma_t12_lin
    )

    # -- MAP point estimate -----------------------------------------------------------
    args_full = (
        t_start, t_end, counts, tau_d, config.dead_time_model, config.quadrature_points,
        A0_prior_mean, A0_prior_sigma, t12_prior_mean, t12_prior_sigma, priors,
    )
    res = _fit_map([A0_lin, t12_lin, B_lin], args_full, bounds)
    A0_final, t12_final, B_final = res.x

    # -- Convergence trace: re-fit using only the first k channels, for growing k ----
    # The optimizer's STARTING guess still comes from the (per-checkpoint) local
    # log-linear fit regardless of prior mode -- just a search heuristic. The PRIOR
    # centre/width, though, stays fixed at the informative override when one is
    # given (external knowledge doesn't change as more of this dataset arrives);
    # only the weakly-informative fallback re-derives it from the growing data.
    checkpoints = np.unique(np.linspace(n_bins // config.n_checkpoints, n_bins, config.n_checkpoints, dtype=int))
    trace_t, trace_A0, trace_t12, trace_B = [], [], [], []
    for k in tqdm(checkpoints, desc="  Convergence trace", unit="point", disable=not show_progress):
        A0_k, t12_k, _, sigma_A0_k, sigma_t12_k = loglinear_fit(
            bin_centers[:k], counts[:k], bin_width[:k], bin_edges[-1],
            tau_d=tau_d, dead_time_model=config.dead_time_model,
        )
        A0_prior_mean_k, A0_prior_sigma_k = _prior_mean_sigma(
            priors.A0_prior_mean, priors.A0_prior_sigma, A0_k, priors.prior_widen_k * sigma_A0_k
        )
        t12_prior_mean_k, t12_prior_sigma_k = _prior_mean_sigma(
            priors.half_life_prior_mean, priors.half_life_prior_sigma, t12_k, priors.prior_widen_k * sigma_t12_k
        )
        args_k = (
            t_start[:k], t_end[:k], counts[:k], tau_d, config.dead_time_model, config.quadrature_points,
            A0_prior_mean_k, A0_prior_sigma_k, t12_prior_mean_k, t12_prior_sigma_k, priors,
        )
        res_k = _fit_map([A0_k, t12_k, 0.0], args_k, bounds)
        trace_A0.append(res_k.x[0])
        trace_t12.append(res_k.x[1])
        trace_B.append(res_k.x[2])
        trace_t.append(t_end[k - 1])

    # -- Local covariance grid, centred on the MAP ------------------------------------
    def local_grid_pass(half_A0, half_t12, half_B, desc):
        A0_grid = np.linspace(max(0.0, A0_final - half_A0), A0_final + half_A0, config.n_vis)
        t12_grid = np.linspace(max(1e-6, t12_final - half_t12), t12_final + half_t12, config.n_vis)
        B_grid = np.linspace(max(0.0, B_final - half_B), B_final + half_B, config.n_vis)
        AA, TT, BB = np.meshgrid(A0_grid, t12_grid, B_grid, indexing="ij")
        LL = np.log(2) / TT
        log_prior = (
            log_prior_student_t(AA, A0_prior_mean, A0_prior_sigma, priors.prior_df)
            + log_prior_student_t(TT, t12_prior_mean, t12_prior_sigma, priors.prior_df)
            + _log_prior_B(BB, priors)
        )
        posterior = np.exp(log_prior - np.max(log_prior))
        posterior /= np.sum(posterior)
        for i in tqdm(range(n_bins), desc=desc, unit="channel", leave=False, disable=not show_progress):
            mu_grid = expected_observed_counts(
                AA, LL, BB, t_start[i], t_end[i], tau_d,
                model=config.dead_time_model, n_quad=config.quadrature_points,
            )
            mu_grid = np.maximum(mu_grid, 1e-300)  # A0 can sit at the grid's lower edge (0)
            log_likelihood = counts[i] * np.log(mu_grid) - mu_grid
            log_posterior = np.log(posterior + 1e-300) + log_likelihood
            log_posterior -= np.max(log_posterior)
            posterior = np.exp(log_posterior)
            posterior /= np.sum(posterior)
        return A0_grid, t12_grid, B_grid, AA, TT, BB, posterior

    def edge_mass_bad(p, grid, floor):
        lo_bad = p[0] > config.edge_mass_tol and grid[0] > floor + 1e-12
        hi_bad = p[-1] > config.edge_mass_tol
        return lo_bad or hi_bad

    def effective_points(p):
        """Participation ratio (inverse Simpson index) of a normalized marginal
        pmf: how many of the n_vis grid cells meaningfully carry posterior mass.
        A window much wider than the TRUE posterior passes edge_mass_bad trivially
        (both tails are ~0) while still being badly under-resolved -- almost all of
        the mass piles onto a handful of cells near the centre, and the resulting
        second-moment (variance) integral underestimates the true posterior width.
        edge_mass_bad alone can't see this: it only ever looks at the two end
        cells, not how concentrated the mass is in between."""
        p = p / p.sum()
        return 1.0 / np.sum(p * p)

    min_effective = config.min_effective_fraction * config.n_vis
    # Half-width (in units of the LOCAL covariance grid's own sigma estimate) that
    # a roughly-Gaussian marginal needs to reach min_effective effective grid cells:
    # for n_vis points spanning +/-k*sigma, n_eff ~= sqrt(pi) * (n_vis - 1) / k.
    target_k = np.sqrt(np.pi) * (config.n_vis - 1) / min_effective

    # -- Initial window: Laplace (Hessian-at-the-MAP) sizing -------------------------
    # A fixed-fraction-of-the-MAP window (the old ``max(0.05 * A0_final, 10.0)``,
    # and B's own heuristic below) can be orders of magnitude wider than the true
    # posterior at very high statistics. Sizing from the local curvature instead
    # gets the window right from the start in the common (well-behaved, roughly
    # Gaussian) case, so the widen loop below rarely needs to do anything -- it
    # stays as a safety net for whichever dimension isn't Laplace-sized (B, when
    # it sits at its physical floor, where this quadratic approximation doesn't
    # apply) and for cases where the Laplace estimate undershoots (e.g. a
    # non-Gaussian posterior).
    half_A0 = max(0.05 * A0_final, 10.0)
    half_t12 = max(0.05 * t12_final, 1.0)
    half_B = max(0.5 * B_final + 10.0, 10.0)
    laplace = _laplace_sigma(A0_final, t12_final, B_final, args_full)
    if laplace is not None:
        half_A0 = max(target_k * laplace[0], 10.0)
        half_t12 = max(target_k * laplace[1], 1.0)
        if len(laplace) == 3:
            half_B = max(target_k * laplace[2], 10.0)

    # -- Containment: widen (all three dimensions in lockstep) until no marginal's
    # tails spill past the window edges. Lockstep, not per-dimension, because A0
    # and half-life are strongly correlated (a longer half-life + lower A0 can
    # mimic a shorter half-life + higher A0 over a finite window): resizing one
    # dimension's window changes what the OTHER marginals integrate over, so
    # independent per-dimension resizing was found to oscillate indefinitely
    # rather than converge, particularly at extreme statistics (see git history).
    converged = False
    for attempt in range(config.max_widen):
        grid_A0, grid_t12, grid_B, AA, TT, BB, posterior = local_grid_pass(
            half_A0, half_t12, half_B, desc=f"  Local grid (pass {attempt + 1}/{config.max_widen})"
        )
        marg_A0 = posterior.sum(axis=(1, 2))
        marg_t12 = posterior.sum(axis=(0, 2))
        marg_B = posterior.sum(axis=(0, 1))
        if not (
            edge_mass_bad(marg_A0, grid_A0, 0.0)
            or edge_mass_bad(marg_t12, grid_t12, 0.0)
            or edge_mass_bad(marg_B, grid_B, 0.0)
        ):
            converged = True
            break
        half_A0 *= 2
        half_t12 *= 2
        half_B *= 2

    # -- Resolution safety net: if contained but still under-resolved (the Laplace
    # estimate undershot, or B -- not Laplace-sized -- is the culprit), shrink all
    # three windows together by the SAME ratio (driven by the worst-resolved
    # dimension) rather than independently, for the same lockstep-not-independent
    # reason as above. Bounded and rare in practice given the Laplace-sized start.
    if converged:
        for _ in range(3):
            n_eff = min(effective_points(marg_A0), effective_points(marg_t12), effective_points(marg_B))
            if n_eff >= min_effective:
                break
            ratio = max(n_eff / min_effective, 0.2)
            half_A0 *= ratio
            half_t12 *= ratio
            half_B *= ratio
            grid_A0, grid_t12, grid_B, AA, TT, BB, posterior = local_grid_pass(
                half_A0, half_t12, half_B, desc="  Local grid (resolution refinement)"
            )
            marg_A0 = posterior.sum(axis=(1, 2))
            marg_t12 = posterior.sum(axis=(0, 2))
            marg_B = posterior.sum(axis=(0, 1))
            if (
                edge_mass_bad(marg_A0, grid_A0, 0.0)
                or edge_mass_bad(marg_t12, grid_t12, 0.0)
                or edge_mass_bad(marg_B, grid_B, 0.0)
            ):
                # Overshot back into truncation -- widen back once and stop refining
                # rather than risk oscillating between the two failure modes.
                half_A0 *= 2
                half_t12 *= 2
                half_B *= 2
                grid_A0, grid_t12, grid_B, AA, TT, BB, posterior = local_grid_pass(
                    half_A0, half_t12, half_B, desc="  Local grid (re-widen)"
                )
                marg_A0 = posterior.sum(axis=(1, 2))
                marg_t12 = posterior.sum(axis=(0, 2))
                marg_B = posterior.sum(axis=(0, 1))
                break

    dA0, dT12, dB = AA - A0_final, TT - t12_final, BB - B_final
    cov = np.empty((3, 3))
    cov[0, 0] = np.sum(posterior * dA0 * dA0)
    cov[1, 1] = np.sum(posterior * dT12 * dT12)
    cov[2, 2] = np.sum(posterior * dB * dB)
    cov[0, 1] = cov[1, 0] = np.sum(posterior * dA0 * dT12)
    cov[0, 2] = cov[2, 0] = np.sum(posterior * dA0 * dB)
    cov[1, 2] = cov[2, 1] = np.sum(posterior * dT12 * dB)
    corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))

    # -- Importance-sampling smoothing of the displayed marginals + credible interval
    def log_posterior_batch(A0, half_life, B, chunk=2000):
        out = np.full(len(A0), -np.inf)
        for s in tqdm(
            range(0, len(A0), chunk), desc="  Importance sampling", unit="batch", leave=False,
            disable=not show_progress,
        ):
            sl = slice(s, s + chunk)
            a, t, b = A0[sl], half_life[sl], B[sl]
            valid = (a > 0) & (t > 0) & (b >= 0)
            if not np.any(valid):
                continue
            av, tv, bv = a[valid], t[valid], b[valid]
            lam = np.log(2) / tv
            mu = expected_observed_counts(
                av[:, None], lam[:, None], bv[:, None], t_start[None, :], t_end[None, :], tau_d,
                model=config.dead_time_model, n_quad=config.quadrature_points,
            )
            mu = np.maximum(mu, 1e-300)
            ll = (
                np.sum(counts[None, :] * np.log(mu) - mu, axis=1)
                + log_prior_student_t(av, A0_prior_mean, A0_prior_sigma, priors.prior_df)
                + log_prior_student_t(tv, t12_prior_mean, t12_prior_sigma, priors.prior_df)
                + _log_prior_B(bv, priors)
            )
            tmp = out[sl]
            tmp[valid] = ll
            out[sl] = tmp
        return out

    # Regularize JUST the IS proposal's covariance, not the reported `cov`: when a
    # parameter (background, almost always) is resolved extremely tightly -- e.g.
    # the sub-microcps precision reachable at very high statistics -- `cov` can be
    # ill-conditioned enough (a near-zero eigenvalue next to a much larger one)
    # that both sampling and the proposal's log-density evaluation lose precision
    # in that eigendirection, occasionally underflowing to -inf for samples that
    # are genuinely near the mode and producing `-inf - (-inf) = nan` importance
    # weights -- which silently corrupts the weighted quantiles (a credible
    # interval that doesn't even bracket the point estimate), not just a cosmetic
    # warning. Flooring the smallest eigenvalue relative to the largest keeps the
    # proposal's shape (and its sampling/logpdf consistency) numerically sound
    # while barely widening it.
    eigval, eigvec = np.linalg.eigh(cov)
    eigval_floored = np.maximum(eigval, 1e-8 * eigval[-1])
    cov_is = (eigvec * eigval_floored) @ eigvec.T

    rng = np.random.default_rng()
    mean_vec = np.array([A0_final, t12_final, B_final])
    is_samples = rng.multivariate_normal(mean_vec, cov_is, size=config.n_is_samples)
    A0_s, t12_s, B_s = is_samples[:, 0], is_samples[:, 1], is_samples[:, 2]

    log_w = log_posterior_batch(A0_s, t12_s, B_s) - multivariate_normal(
        mean=mean_vec, cov=cov_is, allow_singular=True
    ).logpdf(is_samples)
    finite = np.isfinite(log_w)
    log_w[~finite] = -np.inf
    log_w -= np.max(log_w[finite])
    weights = np.where(finite, np.exp(log_w), 0.0)
    weights /= weights.sum()

    # Effective sample size (Kish's ESS, 1/sum(w^2) for normalized weights): how many
    # of the n_is_samples draws are actually carrying the reweighting, as opposed to
    # a handful dominating while the rest contribute ~0. IS in 3D against a proposal
    # this precisely targeted can degenerate badly once the posterior itself gets
    # very tight/ill-conditioned (very high statistics): a single lucky draw can end
    # up with nearly all the weight, making the "smoothed" quantiles/marginals built
    # from it noise rather than a real description of the posterior -- in one
    # observed case, a 95% CI that didn't even bracket the MAP point estimate.
    #
    # That same regime (tight, well-behaved likelihood, huge counts) is exactly
    # where asymptotic normality (Bernstein-von Mises) makes the local grid's own
    # Gaussian second-moment `cov` an accurate description on its own -- so when IS
    # has collapsed, fall back to it directly instead of reporting quantiles built
    # from a degenerate weighted sample.
    ess = 1.0 / np.sum(weights**2)
    is_degenerate = ess < max(0.01 * config.n_is_samples, 20)

    if is_degenerate:
        z = 1.9599639845400545  # norm.ppf(0.975)

        def _gaussian_marginal(mean, sigma, floor=None, n=200, n_sigma=4.0):
            lo = mean - n_sigma * sigma
            if floor is not None:
                lo = max(lo, floor)
            grid = np.linspace(lo, mean + n_sigma * sigma, n)
            return grid, norm.pdf(grid, mean, sigma)

        sigma_A0, sigma_t12, sigma_B = np.sqrt(np.diag(cov))
        plot_A0_grid, A0_marginal = _gaussian_marginal(A0_final, sigma_A0, floor=0.0)
        plot_t12_grid, half_life_marginal = _gaussian_marginal(t12_final, sigma_t12, floor=1e-6)
        plot_B_grid, background_marginal = _gaussian_marginal(B_final, sigma_B, floor=0.0)
        A0_ci95 = (A0_final - z * sigma_A0, A0_final + z * sigma_A0)
        half_life_ci95 = (t12_final - z * sigma_t12, t12_final + z * sigma_t12)
        background_ci95 = (max(0.0, B_final - z * sigma_B), B_final + z * sigma_B)
    else:
        A0_lo, A0_hi = _weighted_quantile(A0_s, weights, [0.001, 0.999])
        t12_lo, t12_hi = _weighted_quantile(t12_s, weights, [0.001, 0.999])
        B_lo, B_hi = _weighted_quantile(B_s, weights, [0.001, 0.999])
        n_plot = 200
        plot_A0_grid = np.linspace(max(0.0, A0_lo), A0_hi, n_plot)
        plot_t12_grid = np.linspace(max(1e-6, t12_lo), t12_hi, n_plot)
        plot_B_grid = np.linspace(max(0.0, B_lo), B_hi, n_plot)
        A0_marginal = _weighted_marginal(A0_s, plot_A0_grid, weights)
        half_life_marginal = _weighted_marginal(t12_s, plot_t12_grid, weights)
        background_marginal = _weighted_marginal(B_s, plot_B_grid, weights)
        A0_ci95 = tuple(_weighted_quantile(A0_s, weights, [0.025, 0.975]))
        half_life_ci95 = tuple(_weighted_quantile(t12_s, weights, [0.025, 0.975]))
        background_ci95 = tuple(_weighted_quantile(B_s, weights, [0.025, 0.975]))

    return FitResult(
        t_start=t_start, t_end=t_end, bin_width=bin_width, counts=counts,
        A0_loglinear=A0_lin, half_life_loglinear=t12_lin,
        sigma_A0_loglinear=sigma_A0_lin, sigma_half_life_loglinear=sigma_t12_lin,
        A0=A0_final, half_life=t12_final, background=B_final, cov=cov, corr=corr,
        trace_t=np.array(trace_t), trace_A0=np.array(trace_A0),
        trace_half_life=np.array(trace_t12), trace_background=np.array(trace_B),
        A0_grid=plot_A0_grid, A0_marginal=A0_marginal,
        half_life_grid=plot_t12_grid, half_life_marginal=half_life_marginal,
        background_grid=plot_B_grid, background_marginal=background_marginal,
        A0_ci95=A0_ci95, half_life_ci95=half_life_ci95, background_ci95=background_ci95,
        converged=converged, dead_time_model=config.dead_time_model,
    )
