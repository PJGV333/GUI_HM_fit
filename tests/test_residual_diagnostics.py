"""Serial correlation of residuals: the symptom under which every reported SE is too small."""
import os

import numpy as np
import pandas as pd
import pytest

from hmfit_core.utils.residual_diagnostics import lag1_autocorrelation, serial_correlation_note


def _ar1(rng, n, rho, ncol=1):
    e = rng.normal(size=(n, ncol))
    out = np.empty_like(e)
    out[0] = e[0]
    for i in range(1, n):
        out[i] = rho * out[i - 1] + np.sqrt(1 - rho**2) * e[i]
    return out


def test_independent_noise_is_not_flagged():
    rng = np.random.default_rng(0)
    flagged = sum(lag1_autocorrelation(rng.normal(size=30))["level"] != "ok" for _ in range(400))
    assert flagged <= 12  # nominal one-sided rate at r1 >= 2/sqrt(30) is ~2 %


def test_correlated_residuals_are_flagged_and_explained():
    rng = np.random.default_rng(1)
    flagged = sum(lag1_autocorrelation(_ar1(rng, 60, 0.8))["level"] in {"warn", "strong"} for _ in range(200))
    assert flagged >= 190
    diag = lag1_autocorrelation(np.sin(np.linspace(0, 6, 40)))
    assert diag["level"] == "strong" and diag["r1"] > 0.9
    note = serial_correlation_note(diag, "time")
    assert "lag-1 autocorrelation" in note and "too small" in note and "along time" in note
    assert serial_correlation_note(lag1_autocorrelation(rng.normal(size=40))) == ""


def test_channels_do_not_add_significance_and_ordering_is_applied():
    rng = np.random.default_rng(2)
    # 200 channels of independent noise over 30 times: still "ok"
    assert lag1_autocorrelation(rng.normal(size=(30, 200)))["level"] == "ok"
    smooth = np.sin(np.linspace(0, 6, 40))
    perm = rng.permutation(40)
    shuffled = lag1_autocorrelation(smooth[perm])
    restored = lag1_autocorrelation(smooth[perm], order=np.linspace(0, 6, 40)[perm])
    assert restored["r1"] == pytest.approx(lag1_autocorrelation(smooth)["r1"])
    assert shuffled["r1"] < restored["r1"]


def test_blocks_missing_values_and_short_series():
    rng = np.random.default_rng(3)
    a, b = _ar1(rng, 40, 0.9, 3), _ar1(rng, 25, 0.9, 2)
    pooled = lag1_autocorrelation([a, b])
    assert pooled["n_points"] == 40 and pooled["level"] in {"warn", "strong"}
    holes = a.copy()
    holes[5, 0] = np.nan
    holes[17, :] = np.nan
    assert np.isfinite(lag1_autocorrelation(holes)["r1"])
    assert lag1_autocorrelation(rng.normal(size=5))["level"] == "insufficient"
    assert lag1_autocorrelation(np.zeros(20))["level"] == "insufficient"
    with pytest.raises(ValueError):
        lag1_autocorrelation(np.zeros((3, 3, 3)))


def _titrate(tmp_path, ct_assumed, name):
    from hmfit_core.acid_base import make_simple_acid_base_system
    from hmfit_core.potentiometry import PotentiometryExperiment, simulate_pH_titration
    from hmfit_core.run_acid_base import run_acid_base

    pka = [9.5, 4.2]
    vol = np.linspace(0.0, 25.0, 40)
    system = make_simple_acid_base_system(analytical_concentration=1e-2, pka=pka, base_charge=-2)
    exp = PotentiometryExperiment(initial_volume=10.0, titrant_volumes=vol, analyte_concentration=1e-2,
                                  titrant_concentration=1e-2, titrant_type="base")
    pH = simulate_pH_titration(exp, system) + np.random.default_rng(0).normal(0.0, 0.005, vol.size)
    path = os.path.join(tmp_path, name)
    pd.DataFrame({"volume_mL": vol, "pH": pH}).to_csv(path, index=False)
    cfg = {"file_path": path, "data_type": "potentiometry", "titrant_concentration": ct_assumed,
           "initial_volume": 10.0, "titrant_type": "base", "analyte_concentration": 1e-2,
           "base_charge": -2, "pka_initial": "8.0,5.0"}
    return run_acid_base(cfg, progress_cb=lambda _m: None)


def test_potentiometry_reports_a_systematic_error_the_se_cannot_see(tmp_path):
    """
    Titrant concentration entered 2 % too high: pKa1 is off by ~0.09 (5 reported SE) and its
    95 % interval excludes the true value. Nothing warned about it; now the residual
    autocorrelation does, in the report, the statistics and the Errors summary.
    """
    from hmfit_core.acid_base_errors import compute_errors_acid_base_from_context

    good = _titrate(str(tmp_path), 1.0e-2, "good.csv")
    assert good["warnings"] == []
    assert abs(good["statistics"]["residual_lag1_autocorrelation"]) < 0.3

    bad = _titrate(str(tmp_path), 1.02e-2, "bad.csv")
    pka1 = next(c["value"] for c in bad["constants"] if c["parameter"] == "pKa1")
    out = compute_errors_acid_base_from_context(bad["errors_context"], {"method": "analytic"})
    row = next(r for r in out["results_rows"] if r[0] == "pKa1")
    assert abs(pka1 - 9.5) > 4.0 * float(row[2])          # the SE does not cover the error
    assert bad["statistics"]["residual_lag1_autocorrelation"] > 0.5
    assert any("correlated along the titrant volume" in w for w in bad["warnings"])
    assert "correlated along the titrant volume" in bad["results_text"]
    assert "correlated along the titrant volume" in out["summary"]
