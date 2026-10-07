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
from scipy.stats import gaussian_kde, multivariate_normal
from tqdm import tqdm

from .deadtime import apply_deadtime
from .model import expected_counts, loglinear_fit
from .priors import Priors, log_prior_background, log_prior_student_t


@dataclass
class FitConfig:
    dead_time_model: str = "nonparalyzable"   # or "paralyzable"
    n_checkpoints: int = 20       # number of points in the convergence trace
    n_vis: int = 61               # local covariance grid resolution per dimension
    edge_mass_tol: float = 1e-4   # local-grid widening stop criterion
    max_widen: int = 6            # max local-grid widening attempts
    n_is_samples: int = 100_000   # importance-sampling draws for marginal smoothing
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


def neg_log_posterior(params, t_s, t_e, w, n, tau_d, dead_time_model, A0_mean, A0_sigma, t12_mean, t12_sigma, priors):
    A0, half_life, B = params
    if A0 <= 0 or half_life <= 0 or B < 0:
        return np.inf
    lam = np.log(2) / half_life
    mu_true = expected_counts(A0, lam, B, t_s, t_e)
    mu = apply_deadtime(mu_true, w, tau_d, model=dead_time_model)
    # Floor away from exactly 0: with the paralyzable model (which is not monotonic in
    # the true rate, see deadtime.py) the optimizer's numerical-gradient probing can
    # occasionally land on a combination that underflows mu to 0.0, which would
    # otherwise raise a RuntimeWarning on log(mu) without changing the result.
    mu = np.maximum(mu, 1e-300)
    neg_loglik = -np.sum(n * np.log(mu) - mu)
    log_prior_total = (
        log_prior_student_t(A0, A0_mean, A0_sigma, priors.prior_df)
        + log_prior_student_t(half_life, t12_mean, t12_sigma, priors.prior_df)
        + log_prior_background(B, priors.b_prior_scale)
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
    A0_lin, t12_lin, B_lin, sigma_A0_lin, sigma_t12_lin = loglinear_fit(
        bin_centers, counts, bin_width, bin_edges[-1], tau_d=tau_d, dead_time_model=config.dead_time_model
    )
    A0_prior_sigma = priors.prior_widen_k * sigma_A0_lin
    t12_prior_sigma = priors.prior_widen_k * sigma_t12_lin

    # -- MAP point estimate -----------------------------------------------------------
    args_full = (
        t_start, t_end, bin_width, counts, tau_d, config.dead_time_model,
        A0_lin, A0_prior_sigma, t12_lin, t12_prior_sigma, priors,
    )
    res = _fit_map([A0_lin, t12_lin, B_lin], args_full, bounds)
    A0_final, t12_final, B_final = res.x

    # -- Convergence trace: re-fit using only the first k channels, for growing k ----
    checkpoints = np.unique(np.linspace(n_bins // config.n_checkpoints, n_bins, config.n_checkpoints, dtype=int))
    trace_t, trace_A0, trace_t12, trace_B = [], [], [], []
    for k in tqdm(checkpoints, desc="  Convergence trace", unit="point", disable=not show_progress):
        A0_k, t12_k, _, sigma_A0_k, sigma_t12_k = loglinear_fit(
            bin_centers[:k], counts[:k], bin_width[:k], bin_edges[-1],
            tau_d=tau_d, dead_time_model=config.dead_time_model,
        )
        args_k = (
            t_start[:k], t_end[:k], bin_width[:k], counts[:k], tau_d, config.dead_time_model,
            A0_k, priors.prior_widen_k * sigma_A0_k, t12_k, priors.prior_widen_k * sigma_t12_k, priors,
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
            log_prior_student_t(AA, A0_lin, A0_prior_sigma, priors.prior_df)
            + log_prior_student_t(TT, t12_lin, t12_prior_sigma, priors.prior_df)
            + log_prior_background(BB, priors.b_prior_scale)
        )
        posterior = np.exp(log_prior - np.max(log_prior))
        posterior /= np.sum(posterior)
        for i in tqdm(range(n_bins), desc=desc, unit="channel", leave=False, disable=not show_progress):
            mu_true_grid = expected_counts(AA, LL, BB, t_start[i], t_end[i])
            mu_grid = apply_deadtime(mu_true_grid, bin_width[i], tau_d, model=config.dead_time_model)
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

    half_A0 = max(0.05 * A0_final, 10.0)
    half_t12 = max(0.05 * t12_final, 1.0)
    half_B = max(0.5 * B_final + 10.0, 10.0)
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
            mu_true = expected_counts(av[:, None], lam[:, None], bv[:, None], t_start[None, :], t_end[None, :])
            mu = apply_deadtime(mu_true, bin_width[None, :], tau_d, model=config.dead_time_model)
            mu = np.maximum(mu, 1e-300)
            ll = (
                np.sum(counts[None, :] * np.log(mu) - mu, axis=1)
                + log_prior_student_t(av, A0_lin, A0_prior_sigma, priors.prior_df)
                + log_prior_student_t(tv, t12_lin, t12_prior_sigma, priors.prior_df)
                + log_prior_background(bv, priors.b_prior_scale)
            )
            tmp = out[sl]
            tmp[valid] = ll
            out[sl] = tmp
        return out

    rng = np.random.default_rng()
    mean_vec = np.array([A0_final, t12_final, B_final])
    is_samples = rng.multivariate_normal(mean_vec, cov, size=config.n_is_samples)
    A0_s, t12_s, B_s = is_samples[:, 0], is_samples[:, 1], is_samples[:, 2]

    log_w = log_posterior_batch(A0_s, t12_s, B_s) - multivariate_normal(
        mean=mean_vec, cov=cov, allow_singular=True
    ).logpdf(is_samples)
    finite = np.isfinite(log_w)
    log_w[~finite] = -np.inf
    log_w -= np.max(log_w[finite])
    weights = np.where(finite, np.exp(log_w), 0.0)
    weights /= weights.sum()

    A0_lo, A0_hi = _weighted_quantile(A0_s, weights, [0.001, 0.999])
    t12_lo, t12_hi = _weighted_quantile(t12_s, weights, [0.001, 0.999])
    B_lo, B_hi = _weighted_quantile(B_s, weights, [0.001, 0.999])
    n_plot = 200
    plot_A0_grid = np.linspace(max(0.0, A0_lo), A0_hi, n_plot)
    plot_t12_grid = np.linspace(max(1e-6, t12_lo), t12_hi, n_plot)
    plot_B_grid = np.linspace(max(0.0, B_lo), B_hi, n_plot)

    return FitResult(
        t_start=t_start, t_end=t_end, bin_width=bin_width, counts=counts,
        A0_loglinear=A0_lin, half_life_loglinear=t12_lin,
        sigma_A0_loglinear=sigma_A0_lin, sigma_half_life_loglinear=sigma_t12_lin,
        A0=A0_final, half_life=t12_final, background=B_final, cov=cov, corr=corr,
        trace_t=np.array(trace_t), trace_A0=np.array(trace_A0),
        trace_half_life=np.array(trace_t12), trace_background=np.array(trace_B),
        A0_grid=plot_A0_grid, A0_marginal=_weighted_marginal(A0_s, plot_A0_grid, weights),
        half_life_grid=plot_t12_grid, half_life_marginal=_weighted_marginal(t12_s, plot_t12_grid, weights),
        background_grid=plot_B_grid, background_marginal=_weighted_marginal(B_s, plot_B_grid, weights),
        A0_ci95=tuple(_weighted_quantile(A0_s, weights, [0.025, 0.975])),
        half_life_ci95=tuple(_weighted_quantile(t12_s, weights, [0.025, 0.975])),
        background_ci95=tuple(_weighted_quantile(B_s, weights, [0.025, 0.975])),
        converged=converged, dead_time_model=config.dead_time_model,
    )
