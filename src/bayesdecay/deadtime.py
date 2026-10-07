"""Detector dead-time models.

Two standard models are supported, selectable via the ``model`` argument accepted
throughout this package (``"nonparalyzable"`` or ``"paralyzable"``):

Non-paralyzable (non-extending)
    After each *recorded* event the detector is blind for a fixed duration ``tau_d``.
    Events arriving during that blind window are lost but do not extend it. In steady
    state, if ``n`` is the true event rate, the recorded rate ``m`` satisfies
    ``m = n / (1 + n * tau_d)`` -- a saturating curve, monotonically increasing in ``n``.

Paralyzable (extending)
    Every true event -- recorded or not -- restarts a dead period of duration
    ``tau_d``. An event is recorded only if the gap since the previous true event
    exceeds ``tau_d``; for a Poisson process with rate ``n`` that gap exceeds ``tau_d``
    with probability ``exp(-n * tau_d)``, giving ``m = n * exp(-n * tau_d)``. Unlike the
    non-paralyzable case this is *not* monotonic: it peaks at ``n = 1 / tau_d`` and
    decreases beyond that ("paralyzed" detector at very high true rates), so a given
    observed rate can in principle correspond to two different true rates. This package
    never needs to invert the relation (simulation and the likelihood both go from an
    assumed true rate to the implied observed rate), so the non-monotonicity is not a
    numerical problem here -- it is simply a real feature of the physics worth knowing
    about when interpreting results at very high count rates.

Both formulas assume the true rate is approximately constant over the time span they
are applied to (one histogram channel here) -- see :func:`bayesdecay.model.auto_bin_count`
for how the channel width is chosen to keep that assumption valid.
"""

from __future__ import annotations

import numpy as np

DEAD_TIME_MODELS = ("nonparalyzable", "paralyzable")


def _nonparalyzable_rate(rate_true, tau_d):
    return rate_true / (1.0 + rate_true * tau_d)


def _paralyzable_rate(rate_true, tau_d):
    return rate_true * np.exp(-rate_true * tau_d)


_RATE_FUNCS = {
    "nonparalyzable": _nonparalyzable_rate,
    "paralyzable": _paralyzable_rate,
}


def apply_deadtime(true_counts, width, tau_d, model="nonparalyzable"):
    """Convert true (dead-time-free) expected counts over a channel into the expected
    *observed* (dead-time-corrected) counts, under the chosen dead-time model.

    Parameters
    ----------
    true_counts : array_like
        Expected number of true events in each channel (no dead time applied).
    width : array_like
        Channel width(s), same shape as ``true_counts`` (or broadcastable).
    tau_d : float
        Dead time, in the same time unit as ``width``.
    model : {"nonparalyzable", "paralyzable"}
        Which dead-time model to apply.
    """
    if model not in _RATE_FUNCS:
        raise ValueError(f"Unknown dead-time model {model!r}; choose one of {DEAD_TIME_MODELS}")
    rate_true = true_counts / width
    rate_obs = _RATE_FUNCS[model](rate_true, tau_d)
    return rate_obs * width
