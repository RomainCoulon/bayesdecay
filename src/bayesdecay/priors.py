"""Prior distributions used by the Bayesian estimator.

Background B
    Exponential, mean ``b_prior_scale``. Its mode is EXACTLY at B=0 (an exponential
    density decreases monotonically from B=0, which is therefore the single most
    probable value a priori) -- matching "B is very probably zero" while still letting
    the data win if they clearly indicate a real background. This is the standard
    prior for a non-negative quantity expected to be small (low-level counting).

A0 and half-life
    Student-t, centred on the classical log-linear fit (:func:`bayesdecay.model.loglinear_fit`),
    with scale equal to ``prior_widen_k`` times the REGRESSION standard error (not an
    arbitrary number) and ``prior_df`` degrees of freedom. This is deliberately a
    WEAKLY informative prior, in both width and shape:

    - width: widened (factor 5 by default) so it is a safety net, not a constraint;
    - shape: a Student-t rather than a Gaussian. A Gaussian's tails decay as
      ``exp(-x^2)`` -- even a "wide" Gaussian pushes back on the MAP with
      exponentially growing force the farther it sits from the prior mean, which is
      exactly what makes a Gaussian prior feel "too strong" once the likelihood
      disagrees with the log-linear estimate (expected, since the log-linear fit
      ignores dead time and background). A Student-t's tails decay only
      polynomially (``~ x^-(df+1)``), so with a low ``prior_df`` (4 by default) it
      still discourages the optimizer from wandering to implausible values in a
      poorly-conditioned regime (e.g. acquisition time on the order of one
      half-life), while barely resisting the data once they are actually
      informative. ``prior_df -> infinity`` recovers a Gaussian; lower values give
      progressively heavier tails (``prior_df = 1`` is a Cauchy).
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


def log_prior_background(B, scale):
    return -B / scale - np.log(scale)


def log_prior_gaussian(x, mean, sigma):
    return -0.5 * ((x - mean) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)


def log_prior_student_t(x, mean, scale, df):
    return _student_t.logpdf(x, df, loc=mean, scale=scale)
