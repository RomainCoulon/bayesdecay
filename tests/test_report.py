import numpy as np

from bayesdecay.fit import FitConfig, fit
from bayesdecay.model import simulate_binned
from bayesdecay.report import check_consistency, format_summary_table, save_results_csv


def _well_conditioned_result():
    rng = np.random.default_rng(0)
    A0, half_life, tau_d = 3000.0, 40.0, 1e-6
    lam = np.log(2) / half_life
    bin_edges, counts = simulate_binned(A0, lam, 1.0, 240.0, tau_d, n_bins=150, rng=rng)
    return fit(bin_edges, counts, tau_d, config=FitConfig(n_checkpoints=5, n_is_samples=5_000), show_progress=False)


def test_check_consistency_quiet_for_a_well_conditioned_fit():
    result = _well_conditioned_result()
    assert check_consistency(result) == []


def test_format_summary_table_contains_parameter_names():
    result = _well_conditioned_result()
    table = format_summary_table(result)
    assert "A0" in table
    assert "half_life" in table
    assert "background" in table


def test_format_summary_table_with_true_values_adds_column():
    result = _well_conditioned_result()
    table = format_summary_table(result, true_values={"A0": 3000.0, "half_life": 40.0, "background": 1.0})
    assert "True value" in table


def test_save_results_csv_writes_expected_header(tmp_path):
    result = _well_conditioned_result()
    path = tmp_path / "out.csv"
    save_results_csv(result, path)
    header = path.read_text().splitlines()[0]
    assert header == (
        "parameter,unit,true_value,loglinear_value,loglinear_u,loglinear_ci95_low,"
        "loglinear_ci95_high,map_value,map_u,map_ci95_low,map_ci95_high"
    )
