"""Prior distributions used by the Bayesian estimator.

Background B
    Exponential, mean ``b_prior_scale``. Its mode is EXACTLY at B=0 (an exponential
    density decreases monotonically from B=0, which is therefore the single most
    probable value a priori) -- matching "B is very probably zero" while still letting
    the data win if they clearly indicate a real background. This is the standard
    prior for a non-negative quantity expected to be small (low-level counting).

A0 and half-life
    Gaussian, centred on the classical log-linear fit (:func:`bayesdecay.model.loglinear_fit`),
    with standard deviation equal to ``prior_widen_k`` times the REGRESSION standard
    error (not an arbitrary number). Deliberately widened (factor 5 by default) so
    this acts as a weak safety net rather than a strong constraint: it keeps the
    optimizer from wandering to implausible values when the data alone leave the
    problem poorly conditioned (strongly correlated parameters -- e.g. an acquisition
    time on the order of one half-life), without pulling the result away from what the
    data actually indicate when they are informative (the much narrower likelihood
    then dominates the prior).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Priors:
    b_prior_scale: float = 5.0       # cps -- adjust to the background typically expected
    prior_widen_k: float = 5.0       # widening factor applied to the log-linear regression SE


def log_prior_background(B, scale):
    return -B / scale - np.log(scale)


def log_prior_gaussian(x, mean, sigma):
    return -0.5 * ((x - mean) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
