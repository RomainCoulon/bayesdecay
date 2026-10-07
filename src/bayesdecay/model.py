"""Decay-curve model, data simulation, and the classical log-linear reference fit."""

from __future__ import annotations

import numpy as np

from .deadtime import apply_deadtime, invert_deadtime, rate_transform

_GL_CACHE = {}


def _gauss_legendre(n):
    """Gauss-Legendre nodes/weights on [-1, 1], cached (there are only a handful of
    distinct ``n_quad`` values any caller will ever pass)."""
    if n not in _GL_CACHE:
        _GL_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _GL_CACHE[n]


def expected_counts(A0, lam, B, t_start, t_end):
    """Analytic integral of ``A0 * exp(-lam * t) + B`` between ``t_start`` and
    ``t_end`` (dead-time-free, i.e. the "true" expected count in that time span)."""
    return (A0 / lam) * (np.exp(-lam * t_start) - np.exp(-lam * t_end)) + B * (t_end - t_start)


def expected_observed_counts(A0, lam, B, t_start, t_end, tau_d, model="nonparalyzable", n_quad=1):
    """Expected OBSERVED (dead-time-corrected) counts per channel.

    With ``n_quad=1`` (the default), this is exactly ``apply_deadtime(expected_counts(...))``
    -- the channel's average true rate, dead-time-corrected once -- at zero extra
    cost over calling those two directly.

    The dead-time rate transform (:func:`bayesdecay.deadtime.rate_transform`) is
    NONLINEAR, so applying it once to a channel's *average* rate is not exactly the
    same as integrating it against the true, continuously time-varying rate within
    the channel: a small, systematic (Jensen's-inequality-style) bias that
    :func:`bayesdecay.model.auto_bin_count` keeps negligible for typical use by
    bounding the per-channel rate change, but which can become statistically
    resolvable at very high count rates/statistics. ``n_quad>1`` corrects for this by
    evaluating the rate transform at ``n_quad`` Gauss-Legendre sub-points within each
    channel and integrating those instead -- trading ``n_quad``-times the rate-
    transform evaluations (this is called inside the hot per-channel loops in
    :mod:`bayesdecay.fit`) for reduced bias. A modest value (3-5) is normally enough,
    since the per-channel rate change is already kept small by construction.
    """
    if n_quad <= 1:
        mu_true = expected_counts(A0, lam, B, t_start, t_end)
        width = np.asarray(t_end) - np.asarray(t_start)
        return apply_deadtime(mu_true, width, tau_d, model=model)

    nodes, weights = _gauss_legendre(n_quad)
    t_start = np.asarray(t_start, dtype=float)
    t_end = np.asarray(t_end, dtype=float)
    half = 0.5 * (t_end - t_start)
    mid = 0.5 * (t_end + t_start)
    t_sub = mid[..., None] + half[..., None] * nodes  # (..., n_quad)

    A0b = np.asarray(A0, dtype=float)[..., None]
    lamb = np.asarray(lam, dtype=float)[..., None]
    Bb = np.asarray(B, dtype=float)[..., None]
    rate_true_sub = A0b * np.exp(-lamb * t_sub) + Bb
    rate_obs_sub = rate_transform(rate_true_sub, tau_d, model=model)

    # Gauss-Legendre weights on [-1, 1] sum to 2, so this is the weighted average.
    rate_obs_avg = np.sum(rate_obs_sub * weights, axis=-1) / 2.0
    width = t_end - t_start
    return rate_obs_avg * width


def simulate_binned(
    A0, lam, B, t_max, tau_d, n_bins, dead_time_model="nonparalyzable", rng=None, quadrature_points=1
):
    """Simulate binned (histogrammed) counts directly, without enumerating individual
    events. Cost is O(n_bins), independent of the total number of counts -- this is
    what makes the estimator usable for measurements spanning hours or days, where
    enumerating every event would be computationally prohibitive.

    ``quadrature_points`` (see :func:`expected_observed_counts`) controls how
    precisely the dead-time correction accounts for the rate varying within each
    channel; the default (1) matches the fitter's own default, so simulated and
    fitted data use the same approximation unless you deliberately raise this (e.g.
    to check how a higher-fidelity "ground truth" affects recovered parameters).
    """
    rng = np.random.default_rng() if rng is None else rng
    bin_edges = np.linspace(0.0, t_max, n_bins + 1)
    t_start, t_end = bin_edges[:-1], bin_edges[1:]

    mu_obs = expected_observed_counts(
        A0, lam, B, t_start, t_end, tau_d, model=dead_time_model, n_quad=quadrature_points
    )
    counts = rng.poisson(mu_obs)
    return bin_edges, counts


def loglinear_fit(t_c, c, w, t_max_fallback, tau_d=0.0, dead_time_model="nonparalyzable"):
    """Regress log(counts / channel width) against time, over non-empty channels.

    Gives a fast estimate of ``(A0, half_life)`` that ignores the background ``B``
    (see the module-level comparison printed by the CLI). ``B`` is deliberately not
    estimated here: a late-acquisition plateau only reflects a real background once
    ``t_max`` extends well past the half-life -- when the acquisition is only on the
    order of one half-life, the "tail" is still decaying, and reading a plateau there
    would give an inflated, wrong background estimate. The full Bayesian estimator
    (see :mod:`bayesdecay.fit`) recovers B properly from the data and prior.

    If ``tau_d`` > 0, the observed rate is first corrected back to an estimated TRUE
    rate (:func:`bayesdecay.deadtime.invert_deadtime`) before the regression. Without
    this, the fit is biased even with zero background: dead-time losses are
    themselves rate-dependent (heaviest at the high rates right after t=0, tapering
    off as the source decays), so a plain log-linear fit on the raw observed rate
    systematically reads a smaller A0 and a shorter half-life than the truth -- the
    decay looks artificially "faster" than it really is because the earliest,
    highest-rate channels are disproportionately thinned by dead time. This also
    matters for the Bayesian estimator: since this fit sets both its starting point
    AND its prior centre, an uncorrected bias here otherwise has to be walked back by
    the data alone, most visibly as a transient in the convergence trace during the
    first few checkpoints (where there's not yet enough data to fully override it).

    Also returns the regression standard errors on A0 and half_life (propagated from
    the fitted-coefficient covariance, via ``numpy.polyfit(..., cov=True)`` and the
    delta method) -- used as the width of the weakly-informative Bayesian prior on
    A0 and half_life (see :mod:`bayesdecay.fit`).

    Returns
    -------
    A0_hat, half_life_hat, B_hat, sigma_A0, sigma_half_life
    """
    rate = c / w
    if tau_d > 0:
        rate = invert_deadtime(rate, tau_d, model=dead_time_model)
    # Poisson noise on a channel can occasionally push its observed rate right to (or
    # past, before clipping) the edge of a dead-time model's valid domain -- e.g. near
    # the paralyzable model's peak rate 1/tau_d. invert_deadtime() clips rather than
    # raising, but guard here too: exclude anything that still came out non-finite or
    # non-positive, on top of the existing "non-empty channel" filter.
    valid = (c > 0) & np.isfinite(rate) & (rate > 0)
    if valid.sum() < 3:
        a0_fallback = 0.5 * max(rate.max(), 1.0)
        return a0_fallback, t_max_fallback, 0.0, 0.5 * a0_fallback, 0.5 * t_max_fallback
    (slope, intercept), pcov = np.polyfit(t_c[valid], np.log(rate[valid]), 1, cov=True)
    sigma_slope, sigma_intercept = np.sqrt(np.diag(pcov))
    lam_hat = max(-slope, 1e-12)
    half_life_hat = np.log(2) / lam_hat
    a0_hat = np.exp(intercept)
    # Delta method: half_life = ln2/lam, A0 = exp(intercept).
    sigma_half_life = (np.log(2) / lam_hat**2) * sigma_slope
    sigma_a0 = a0_hat * sigma_intercept
    return a0_hat, half_life_hat, 0.0, sigma_a0, sigma_half_life


def auto_bin_count(
    A0, lam, B, t_max, tau_d, dead_time_model="nonparalyzable",
    rate_change_tol=0.01, n_bins_prelim=200, n_bins_min=200, n_bins_max=20_000, rng=None,
):
    """Automatically choose the number of channels so that the dead-time formula's
    "rate approximately constant per channel" assumption holds to within
    ``rate_change_tol`` (relative).

    For a pure exponential ``A0 * exp(-lam * t)``, the RELATIVE rate of change is the
    constant ``-lam`` at every ``t`` (adding a constant background only damps this
    further at late times, so bounding with ``lam`` alone is a safe, conservative
    choice). Requiring ``lam * dt <= rate_change_tol`` over a channel of width ``dt``
    gives ``n_bins = ceil(t_max * ln2 / (rate_change_tol * half_life))``.

    Since the half-life is not known in advance, a cheap preliminary pass (coarse,
    just enough channels for a log-linear fit) gives a first estimate, which is then
    used to choose the channel count for the real analysis.

    Returns
    -------
    n_bins : int
    half_life_prelim : float
        The preliminary half-life estimate used to derive ``n_bins`` (useful for
        logging/diagnostics).
    """
    _, counts_prelim = simulate_binned(
        A0, lam, B, t_max, tau_d, n_bins_prelim, dead_time_model=dead_time_model, rng=rng
    )
    edges_prelim = np.linspace(0.0, t_max, n_bins_prelim + 1)
    tc_prelim = 0.5 * (edges_prelim[:-1] + edges_prelim[1:])
    w_prelim = np.diff(edges_prelim)
    return bin_count_from_coarse_pass(
        tc_prelim, counts_prelim, w_prelim, t_max, rate_change_tol, n_bins_min, n_bins_max,
        tau_d=tau_d, dead_time_model=dead_time_model,
    )


def bin_count_from_coarse_pass(
    bin_centers, counts, bin_width, t_max, rate_change_tol=0.01, n_bins_min=200, n_bins_max=20_000,
    tau_d=0.0, dead_time_model="nonparalyzable",
):
    """Same channel-count rule as :func:`auto_bin_count`, but starting from an already
    binned coarse pass (real or simulated) rather than running a new simulation --
    used for real experimental data, where there is no "true" model to simulate from.

    Returns
    -------
    n_bins : int
    half_life_prelim : float
    """
    _, half_life_prelim, *_ = loglinear_fit(
        bin_centers, counts, bin_width, t_max, tau_d=tau_d, dead_time_model=dead_time_model
    )
    n_bins = int(np.clip(
        np.ceil(t_max * np.log(2) / (rate_change_tol * half_life_prelim)), n_bins_min, n_bins_max
    ))
    return n_bins, half_life_prelim
