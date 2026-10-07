"""bayesdecay: Bayesian estimation of radioactive half-life, initial activity, and
background from counting data (simulated or real list-mode timestamps), accounting
for detector dead time (paralyzable or non-paralyzable).
"""

from .deadtime import DEAD_TIME_MODELS, apply_deadtime
from .fit import FitConfig, FitResult, fit
from .io import load_timestamps, timestamps_to_binned
from .model import auto_bin_count, bin_count_from_coarse_pass, expected_counts, loglinear_fit, simulate_binned
from .priors import Priors, log_prior_background, log_prior_gaussian
from .report import format_summary_table, save_results_csv

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DEAD_TIME_MODELS",
    "apply_deadtime",
    "FitConfig",
    "FitResult",
    "fit",
    "load_timestamps",
    "timestamps_to_binned",
    "auto_bin_count",
    "bin_count_from_coarse_pass",
    "expected_counts",
    "loglinear_fit",
    "simulate_binned",
    "Priors",
    "log_prior_background",
    "log_prior_gaussian",
    "format_summary_table",
    "save_results_csv",
]
