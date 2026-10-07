# bayesdecay

Bayesian estimation of **radioactive half-life, initial activity, and background**
from counting data, with proper **detector dead-time correction** — scales from a
few seconds of acquisition to measurements spanning hours or days.

## Why

Fitting a decaying count rate sounds simple until you need to do it properly:

- **Dead time matters.** At count rates where dead-time losses are non-negligible, a
  naive fit (e.g. linear regression on log-counts) is systematically biased. This
  package models the dead-time-corrected expected count per channel directly in the
  likelihood, for either a **non-paralyzable** (non-extending) or **paralyzable**
  (extending) detector.
- **Background is rarely well separated from the decay tail**, especially when the
  acquisition time is only a few half-lives long. A weakly-informative prior (B is
  *a priori* most probably zero, but the data can override it) regularizes this
  instead of letting the fit chase noise.
- **Long acquisitions shouldn't mean long compute.** The estimator works on binned
  (histogrammed) data with a channel count chosen automatically so that the
  "rate ≈ constant per channel" approximation underlying the dead-time formula stays
  valid — cost scales with the number of channels, not the number of counts, so hours-
  or days-long list-mode acquisitions stay tractable.
- **Uncertainty should mean something.** Point estimates come with a full posterior
  covariance (propagated to the fitted decay curve via the GUM law of propagation of
  uncertainty) and both a standard uncertainty and a non-parametric 95% credible
  interval for every parameter.

## Method, briefly

1. **Binning.** Data (simulated or loaded from a real timestamp CSV) is histogrammed
   into channels. If not given explicitly, the channel count is chosen automatically
   from a quick preliminary half-life estimate, so that the per-channel rate change
   stays under 1% (configurable) — this is what keeps the dead-time formula valid.
2. **Classical reference fit.** A log-linear regression (ignoring dead time and
   background) gives a fast point estimate, with regression standard errors.
3. **Bayesian MAP.** The posterior (binned Poisson likelihood with the proper
   dead-time formula and background term, times weak priors — see below) is
   maximized directly by numerical optimization, restarted from a few starting points
   for robustness in strongly correlated regimes (e.g. acquisition time comparable to
   one half-life).
4. **Covariance.** A small grid local to the MAP (auto-widened until it safely
   contains the posterior mass) gives the covariance matrix by direct integration —
   no Gaussian/Laplace approximation, robust even when background sits at its
   physical floor (B ≥ 0).
5. **Reporting.** Importance sampling (proposal = Gaussian matching the covariance)
   gives smooth marginal posterior plots and an empirical (non-Gaussian) 95% credible
   interval for each parameter.

Priors: A0 and T1/2 get a weakly-informative **Student-t** prior centred on the
log-linear fit, with scale equal to 5× its own regression standard error and 4
degrees of freedom by default (`--prior-widen-k`, `--prior-df`). A Student-t rather
than a Gaussian: a Gaussian's tails decay exponentially, so even a "wide" one
resists the data with rapidly growing force the farther the MAP sits from the prior
mean — a Student-t's tails decay only polynomially, so it still discourages the
optimizer from wandering to implausible values in a poorly-conditioned regime,
without fighting the likelihood once the data are actually informative. Background B
gets an exponential prior — its mode is exactly zero, matching "B is probably
negligible" while still letting a real
background win if the data indicate one.

## Installation

```bash
pip install bayesdecay          # once published to PyPI
# or, from a clone of this repository:
pip install -e .
```

Requires Python ≥ 3.9. Dependencies: `numpy`, `scipy`, `matplotlib`, `tqdm`.

## Command-line usage

### Validate against simulated data

```bash
bayesdecay simulate --a0 5000 --half-life 500 --background 0 \
    --acquisition-time 2000 --dead-time 20e-6 \
    --dead-time-model nonparalyzable --output-dir results/
```

Prints a summary table (log-linear vs. Bayesian MAP, each with 1σ and 95% CI, next
to the known true values) and saves three figures plus a results CSV to
`results/`.

### Fit real experimental data

```bash
bayesdecay fit events.csv --timestamp-unit seconds --dead-time 20e-6 \
    --dead-time-model nonparalyzable --output-dir results/
```

`events.csv` is a single column of event timestamps (a header row is auto-detected;
use `--column` to pick a specific column by name or index). **The timestamp unit
matters** — `--timestamp-unit` accepts `seconds`, `milliseconds`, `microseconds`, or
`nanoseconds` and converts internally; every other quantity (`--dead-time`,
`--acquisition-time`) is in seconds.

Run `bayesdecay simulate --help` or `bayesdecay fit --help` for the full option list
(prior widths, number of importance-sampling draws, convergence-trace resolution,
`--no-plots`, `--quiet`, ...).

## Python API

```python
import numpy as np
from bayesdecay import simulate_binned, fit, FitConfig, Priors
from bayesdecay.report import format_summary_table

lam = np.log(2) / 500.0  # half-life = 500 s
bin_edges, counts = simulate_binned(
    A0=5000.0, lam=lam, B=0.0, t_max=2000.0, tau_d=20e-6, n_bins=300,
    dead_time_model="nonparalyzable",
)

config = FitConfig(
    dead_time_model="nonparalyzable",
    priors=Priors(b_prior_scale=5.0, prior_widen_k=5.0),
)
result = fit(bin_edges, counts, tau_d=20e-6, config=config)

print(format_summary_table(result))
print(result.A0, result.u_A0, result.A0_ci95)
print(result.half_life, result.u_half_life, result.half_life_ci95)
```

Loading real data uses `bayesdecay.io`:

```python
from bayesdecay.io import load_timestamps, timestamps_to_binned

timestamps = load_timestamps("events.csv", unit="seconds")
bin_edges, counts, half_life_prelim = timestamps_to_binned(timestamps)
```

## Output

Each run produces:

- `bayesdecay_results.csv` — one row per parameter (A0, half-life, background), with
  the log-linear and Bayesian point estimates, standard uncertainties, and 95%
  interval bounds.
- `bayesdecay_activity.png` — fitted decay curve (with propagated ±1σ band) vs. the
  classical log-linear fit, over the raw binned counts.
- `bayesdecay_convergence.png` — how each estimate evolves as more of the
  acquisition is used.
- `bayesdecay_posteriors.png` — smoothed marginal posteriors and the
  parameter correlation matrix.

## Development

```bash
git clone https://github.com/RomainCoulon/bayesdecay.git
cd bayesdecay
pip install -e ".[test]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
