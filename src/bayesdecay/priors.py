"""Prior distributions used by the Bayesian estimator.

Two ways to build a :class:`Priors`:

``Priors.weakly_informative()`` (also plain ``Priors()`` -- this is the default)
    A0 and half-life: Student-t, centred on THIS measurement's own classical
    log-linear fit (:func:`bayesdecay.model.loglinear_fit`), with scale equal to
    ``prior_widen_k`` times the regression standard error (not an arbitrary
    number) and ``prior_df`` degrees of freedom -- deliberately weak in both
    width and shape (see below). Background: exponential, mean ``b_prior_scale``,
    mode exactly at B=0 ("probably zero, but let the data override that").

    Because the prior's CENTRE comes from the same dataset the likelihood also
    uses, this is a convenient safety net when no reliable external knowledge of
    A0/half-life/background exists ahead of the measurement -- but it also means
    the reported credible interval doesn't separately account for the prior
    centre's own sampling uncertainty ("double dipping" on one dataset), which
    can show up as under-coverage in low-information regimes (e.g. a short
    measurement time, where the log-linear fit itself is quite uncertain).

``Priors.informative(...)``
    Centres the prior on EXTERNALLY supplied knowledge instead -- e.g. a
    half-life from a nuclear-data evaluation (DDEP, ENSDF) independent of this
    measurement, or a background rate from a separate blank measurement. Avoids
    the double-dipping issue above for whichever parameter(s) are supplied (any
    left as ``None`` fall back to the weakly-informative, data-centred
    behaviour). Should be tighter (smaller sigma) than the default's widened
    safety-net scale, reflecting genuine prior confidence rather than a
    fallback -- and defaults ``prior_df`` much higher (nearer-Gaussian) since
    that heavy-tailed hedging exists specifically to protect against the
    log-linear fit being a poor guess, which doesn't apply to external
    knowledge trusted independently of this dataset.

A0 and half-life shape (both modes)
    Student-t rather than Gaussian. A Gaussian's tails decay as ``exp(-x^2)`` --
    even a "wide" Gaussian pushes back on the MAP with exponentially growing
    force the farther it sits from the prior mean, which is exactly what makes a
    Gaussian prior feel "too strong" once the likelihood disagrees with the
    prior centre. A Student-t's tails decay only polynomially
    (``~ x^-(df+1)``), so even a low ``prior_df`` still discourages the
    optimizer from wandering to implausible values in a poorly-conditioned
    regime (e.g. acquisition time on the order of one half-life), while barely
    resisting the data once they are actually informative. ``prior_df ->
    infinity`` recovers a Gaussian; lower values give progressively heavier
    tails (``prior_df = 1`` is a Cauchy).

Background shape
    Exponential by default (mode at B=0, matching "probably negligible"); when
    ``background_prior_mean``/``background_prior_sigma`` are supplied (via
    ``informative()``), a Gaussian centred there instead -- appropriate once a
    real background rate, not just "probably small", is independently known.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as _student_t


@dataclass
class Priors:
    b_prior_scale: float = 5.0       # cps -- adjust to the background typically expected
    prior_widen_k: float = 5.0       # widening factor applied to the log-linear regression SE
    prior_df: float = 4.0            # Student-t degrees of freedom for the A0/half-life prior (lower = heavier tails = less informative)

    # Optional INFORMATIVE overrides: when set (both mean and sigma) for a given
    # parameter, they REPLACE that parameter's weakly-informative, data-centred
    # prior entirely -- see Priors.informative(). Left at None (the default
    # Priors()/weakly_informative() case), that parameter keeps the
    # weakly-informative behaviour described above.
    A0_prior_mean: float | None = None
    A0_prior_sigma: float | None = None
    half_life_prior_mean: float | None = None
    half_life_prior_sigma: float | None = None
    background_prior_mean: float | None = None
    background_prior_sigma: float | None = None

    @classmethod
    def weakly_informative(cls, prior_widen_k=5.0, prior_df=4.0, b_prior_scale=5.0):
        """The default behaviour, spelled out: priors centred on THIS measurement's
        own log-linear fit, widened as a safety net rather than a real constraint.
        Use when there's no reliable external knowledge of A0/half-life/background
        ahead of the measurement -- the common case."""
        return cls(prior_widen_k=prior_widen_k, prior_df=prior_df, b_prior_scale=b_prior_scale)

    @classmethod
    def informative(
        cls, half_life_mean=None, half_life_sigma=None,
        A0_mean=None, A0_sigma=None, background_mean=None, background_sigma=None,
        prior_df=30.0, b_prior_scale=5.0,
    ):
        """Priors centred on EXTERNAL knowledge (e.g. a DDEP-evaluated half-life, or
        a background rate from a separate blank measurement), not on this
        measurement's own log-linear fit -- avoids the "double dipping" the default
        weakly-informative prior incurs (see the module docstring). Any parameter
        left at ``None`` falls back to the usual weakly-informative, data-centred
        prior for just that parameter -- e.g. pass only ``half_life_mean``/
        ``half_life_sigma`` to use an external half-life while A0 and background
        still use the default data-driven behaviour.

        ``prior_df`` defaults much higher (30, nearer-Gaussian) than
        ``weakly_informative``'s default of 4: the heavy Student-t tails exist to
        hedge against the log-linear fit being a poor guess, which doesn't apply to
        external knowledge trusted independently of this dataset.
        """
        return cls(
            prior_df=prior_df, b_prior_scale=b_prior_scale,
            half_life_prior_mean=half_life_mean, half_life_prior_sigma=half_life_sigma,
            A0_prior_mean=A0_mean, A0_prior_sigma=A0_sigma,
            background_prior_mean=background_mean, background_prior_sigma=background_sigma,
        )


def log_prior_background(B, scale):
    return -B / scale - np.log(scale)


def log_prior_gaussian(x, mean, sigma):
    return -0.5 * ((x - mean) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)


def log_prior_student_t(x, mean, scale, df):
    return _student_t.logpdf(x, df, loc=mean, scale=scale)
