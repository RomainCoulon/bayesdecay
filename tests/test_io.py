import numpy as np
import pytest

from bayesdecay.io import load_timestamps, timestamps_to_binned


def test_load_timestamps_with_header(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text("timestamp\n1.0\n2.5\n0.2\n10.0\n")
    ts = load_timestamps(path, unit="seconds")
    assert list(ts) == pytest.approx([0.0, 0.8, 2.3, 9.8])  # sorted, zeroed to first event


def test_load_timestamps_without_header(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text("1.0\n2.0\n3.0\n")
    ts = load_timestamps(path, unit="seconds")
    assert list(ts) == pytest.approx([0.0, 1.0, 2.0])


def test_load_timestamps_unit_conversion(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text("1000.0\n2000.0\n")  # milliseconds
    ts = load_timestamps(path, unit="milliseconds")
    assert list(ts) == pytest.approx([0.0, 1.0])  # converted to seconds


def test_load_timestamps_unknown_unit_raises(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text("1.0\n2.0\n")
    with pytest.raises(ValueError):
        load_timestamps(path, unit="fortnights")


def test_timestamps_to_binned_auto_bins():
    rng = np.random.default_rng(0)
    # Synthetic exponential-ish arrivals via thinning a homogeneous Poisson process.
    lam = np.log(2) / 20.0
    t_max = 100.0
    n_candidates = 20_000
    t_cands = np.sort(rng.uniform(0, t_max, n_candidates))
    rate_max = 500.0
    probs = (rate_max * np.exp(-lam * t_cands)) / rate_max
    timestamps = t_cands[rng.random(n_candidates) < probs]

    bin_edges, counts, half_life_prelim = timestamps_to_binned(timestamps, t_max=t_max)
    assert bin_edges[0] == 0.0
    assert bin_edges[-1] == pytest.approx(t_max)
    assert counts.sum() == len(timestamps)
    assert half_life_prelim is not None and half_life_prelim > 0


def test_timestamps_to_binned_fixed_n_bins():
    timestamps = np.array([0.1, 0.5, 1.2, 1.9])
    bin_edges, counts, half_life_prelim = timestamps_to_binned(timestamps, t_max=2.0, n_bins=4)
    assert len(counts) == 4
    assert half_life_prelim is None
