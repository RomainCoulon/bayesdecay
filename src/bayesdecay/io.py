"""Loading experimental list-mode event timestamps from a CSV file, and histogramming
them into the binned-channel format the estimator (:mod:`bayesdecay.fit`) consumes.

This is the entry point for real data, as opposed to :func:`bayesdecay.model.simulate_binned`
used for simulated validation data -- both ultimately produce the same
``(bin_edges, counts)`` pair, so the rest of the pipeline (auto-binning, fitting,
reporting, plotting) does not need to know which one it got.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .model import bin_count_from_coarse_pass

TIMESTAMP_UNITS = {
    "s": 1.0, "seconds": 1.0,
    "ms": 1e-3, "milliseconds": 1e-3,
    "us": 1e-6, "microseconds": 1e-6,
    "ns": 1e-9, "nanoseconds": 1e-9,
}


def load_timestamps(path, column=None, unit="seconds", delimiter=","):
    """Load a single column of event timestamps from a CSV file into a sorted array
    of seconds, time-zeroed to the first event.

    Parameters
    ----------
    path : str or Path
    column : str or int, optional
        Column name (if the file has a header row) or a 0-based column index. If
        omitted, the first column is used.
    unit : {"seconds"/"s", "milliseconds"/"ms", "microseconds"/"us", "nanoseconds"/"ns"}
        Unit the timestamps are recorded in. Every other quantity in this package
        (dead time, acquisition time, rates in cps) is in seconds, so timestamps are
        converted on load -- get this wrong and every downstream result (half-life,
        dead-time correction, ...) will be off by the corresponding factor.
    delimiter : str

    Returns
    -------
    numpy.ndarray
        Sorted timestamps in seconds, with the first event at t=0.
    """
    if unit not in TIMESTAMP_UNITS:
        raise ValueError(f"Unknown timestamp unit {unit!r}; choose one of {sorted(TIMESTAMP_UNITS)}")
    scale = TIMESTAMP_UNITS[unit]

    path = Path(path)
    with path.open(newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = bool(sample.strip()) and csv.Sniffer().has_header(sample)
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader) if has_header else None

        if column is None:
            col_idx = 0
        elif isinstance(column, int):
            col_idx = column
        elif header is not None and column in header:
            col_idx = header.index(column)
        else:
            raise ValueError(f"Column {column!r} not found; header was {header!r}")

        values = [float(row[col_idx]) for row in reader if row]

    if not values:
        raise ValueError(f"No timestamp rows found in {path}")

    timestamps = np.sort(np.asarray(values, dtype=float)) * scale
    timestamps -= timestamps[0]
    return timestamps


def timestamps_to_binned(
    timestamps, t_max=None, n_bins=None,
    rate_change_tol=0.01, n_bins_prelim=200, n_bins_min=200, n_bins_max=20_000,
):
    """Histogram a sorted array of event timestamps (seconds) into fixed-width channels.

    If ``n_bins`` is not given, it is chosen automatically the same way as for
    simulated data (see :func:`bayesdecay.model.auto_bin_count`): a coarse preliminary
    histogram gives a quick half-life estimate, from which the channel width that
    keeps the dead-time formula's "rate ~ constant per channel" assumption valid is
    derived.

    Parameters
    ----------
    timestamps : array_like
        Sorted event times in seconds (as returned by :func:`load_timestamps`).
    t_max : float, optional
        Acquisition duration. Defaults to the last timestamp.
    n_bins : int, optional
        Fixed channel count; if omitted, chosen automatically (recommended).

    Returns
    -------
    bin_edges : numpy.ndarray, shape (n_bins + 1,)
    counts : numpy.ndarray, shape (n_bins,)
    half_life_prelim : float or None
        The preliminary half-life estimate used for auto-binning (``None`` if
        ``n_bins`` was given explicitly).
    """
    t_max = float(timestamps[-1]) if t_max is None else t_max
    half_life_prelim = None

    if n_bins is None:
        edges_prelim = np.linspace(0.0, t_max, n_bins_prelim + 1)
        counts_prelim, _ = np.histogram(timestamps, bins=edges_prelim)
        centers_prelim = 0.5 * (edges_prelim[:-1] + edges_prelim[1:])
        width_prelim = np.diff(edges_prelim)
        n_bins, half_life_prelim = bin_count_from_coarse_pass(
            centers_prelim, counts_prelim, width_prelim, t_max, rate_change_tol, n_bins_min, n_bins_max
        )

    bin_edges = np.linspace(0.0, t_max, n_bins + 1)
    counts, _ = np.histogram(timestamps, bins=bin_edges)
    return bin_edges, counts.astype(int), half_life_prelim
