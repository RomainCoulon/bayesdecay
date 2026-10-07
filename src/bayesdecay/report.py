"""Human-readable summary table and CSV export of a :class:`bayesdecay.fit.FitResult`."""

from __future__ import annotations

import csv as _csv

Z95 = 1.959963984540054  # normal 97.5th-percentile quantile (95% CI = value +/- Z95 * sigma)


def _fmt(value, sigma, unit, decimals):
    return f"{value:.{decimals}f} ± {sigma:.{decimals}f} {unit}"


def _fmt_ci(ci, decimals):
    return f"[{ci[0]:.{decimals}f}, {ci[1]:.{decimals}f}]"


def _rows(result, true_values=None):
    """Build the raw (unformatted) per-parameter rows shared by the printed table and
    the CSV export. ``true_values`` is an optional dict with keys "A0", "half_life",
    "background" -- used only for validation runs against simulated data."""
    true_values = true_values or {}
    A0_lin_ci = (
        result.A0_loglinear - Z95 * result.sigma_A0_loglinear,
        result.A0_loglinear + Z95 * result.sigma_A0_loglinear,
    )
    t12_lin_ci = (
        result.half_life_loglinear - Z95 * result.sigma_half_life_loglinear,
        result.half_life_loglinear + Z95 * result.sigma_half_life_loglinear,
    )
    return [
        (
            "A0", "cps", 1, true_values.get("A0"),
            (result.A0_loglinear, result.sigma_A0_loglinear, A0_lin_ci),
            (result.A0, result.u_A0, result.A0_ci95),
        ),
        (
            "half_life", "s", 3, true_values.get("half_life"),
            (result.half_life_loglinear, result.sigma_half_life_loglinear, t12_lin_ci),
            (result.half_life, result.u_half_life, result.half_life_ci95),
        ),
        (
            "background", "cps", 2, true_values.get("background"),
            None,
            (result.background, result.u_background, result.background_ci95),
        ),
    ]


def format_summary_table(result, true_values=None):
    """Return the summary table (log-linear reference vs. Bayesian MAP, each with its
    standard uncertainty and 95% interval) as a printable string.

    Note on the two 95% intervals: the log-linear one is the standard normal
    approximation (value +/- 1.96*regression SE) -- that is all a plain OLS fit gives
    you. The Bayesian one is an EMPIRICAL credible interval (2.5th/97.5th percentile of
    the importance-weighted posterior samples) -- not a Gaussian approximation, so it
    correctly reflects skew (e.g. background pinned near its physical floor at 0).
    """
    rows = _rows(result, true_values)
    has_true = any(r[3] is not None for r in rows)

    headers = ["Parameter"]
    if has_true:
        headers.append("True value")
    headers += ["Log-linear (1σ)", "95% CI (log-lin.)", "Bayesian MAP (1σ)", "95% CI (credible)"]

    table_rows = [headers]
    for name, unit, dec, true_val, lin, bay in rows:
        row = [name]
        if has_true:
            row.append("--" if true_val is None else f"{true_val:.{dec}f} {unit}")
        if lin is None:
            row += ["-- (not estimated)", "--"]
        else:
            v, s, ci = lin
            row += [_fmt(v, s, unit, dec), _fmt_ci(ci, dec)]
        v, s, ci = bay
        row += [_fmt(v, s, unit, dec), _fmt_ci(ci, dec)]
        table_rows.append(row)

    col_w = [max(len(r[i]) for r in table_rows) for i in range(len(headers))]
    sep = "-+-".join("-" * w for w in col_w)
    lines = [sep, " | ".join(h.ljust(w) for h, w in zip(table_rows[0], col_w)), sep]
    for row in table_rows[1:]:
        lines.append(" | ".join(c.ljust(w) for c, w in zip(row, col_w)))
    lines.append(sep)
    return "\n".join(lines)


def check_consistency(result):
    """Sanity-check the MAP point estimates against their own 95% credible intervals.

    In a well-behaved fit the point estimate sits comfortably inside its interval.
    If it does not, that is a sign the posterior is not well approximated by the local
    Gaussian used to build the importance-sampling proposal (or that the MAP
    optimizer landed on a different, poorer local mode) -- most likely in a severely
    poorly-conditioned fit (e.g. an acquisition time only on the order of one
    half-life). Returns a list of human-readable warning strings (empty if nothing
    looks off).

    All three parameters have a physical floor at 0. When a point estimate sits
    exactly at (or essentially at) that floor -- the expected, healthy outcome when a
    parameter is legitimately consistent with zero, background most commonly -- its
    empirical credible interval's lower bound naturally lands a little above 0 (a
    one-sided pile-up at a hard boundary); that is not a sign of a poor fit, so the
    LOWER bound is not checked in that case.
    """
    warnings = []
    checks = [
        ("A0", result.A0, result.A0_ci95),
        ("half_life", result.half_life, result.half_life_ci95),
        ("background", result.background, result.background_ci95),
    ]
    for name, value, ci in checks:
        at_floor = value <= 1e-9
        low_bad = not at_floor and value < ci[0]
        high_bad = value > ci[1]
        if low_bad or high_bad:
            warnings.append(
                f"{name}: the MAP point estimate ({value:.4g}) falls outside its own 95% "
                f"credible interval ({ci[0]:.4g}, {ci[1]:.4g}). The fit may be poorly "
                "conditioned (e.g. an acquisition time only on the order of one "
                "half-life) -- consider a tighter prior (higher --prior-df or lower "
                "--prior-widen-k) or a longer acquisition."
            )
    return warnings


def save_results_csv(result, path, true_values=None):
    """Write the raw numeric values behind :func:`format_summary_table` to a CSV file
    (one row per parameter), for downstream use in a spreadsheet or another script."""
    rows = _rows(result, true_values)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow([
            "parameter", "unit", "true_value",
            "loglinear_value", "loglinear_u", "loglinear_ci95_low", "loglinear_ci95_high",
            "map_value", "map_u", "map_ci95_low", "map_ci95_high",
        ])
        for name, unit, _dec, true_val, lin, bay in rows:
            v_bay, s_bay, ci_bay = bay
            if lin is None:
                v_lin = s_lin = ci_lin_lo = ci_lin_hi = ""
            else:
                v_lin, s_lin, ci_lin = lin
                ci_lin_lo, ci_lin_hi = ci_lin
            writer.writerow([
                name, unit, "" if true_val is None else true_val,
                v_lin, s_lin, ci_lin_lo, ci_lin_hi,
                v_bay, s_bay, ci_bay[0], ci_bay[1],
            ])
