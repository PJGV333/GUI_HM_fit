from __future__ import annotations

import numpy as np
import pandas as pd

from hmfit_core.acid_base import (
    NMRAcidBaseDataset,
    SpectroscopicAcidBaseDataset,
    distribution_fractions_from_pH,
    fit_nmr_acid_base,
    fit_spectroscopy_acid_base,
    log_beta_to_pka,
    make_simple_acid_base_system,
    pka_to_log_beta,
)
from hmfit_core.potentiometry import (
    PotentiometryExperiment,
    fit_potentiometry,
    simulate_pH_titration,
)
from hmfit_core.run_acid_base import run_acid_base


def test_pka_log_beta_roundtrip():
    pka = [4.0, 7.0]
    log_beta = pka_to_log_beta(pka)
    pka_back = log_beta_to_pka(log_beta)
    assert np.allclose(pka_back, pka, atol=1e-12, rtol=0.0)


def test_distribution_fractions_monoprotic():
    fractions = distribution_fractions_from_pH(np.asarray([2.0, 5.0, 8.0]), [5.0])
    assert np.allclose(fractions.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
    assert np.allclose(fractions[1], [0.5, 0.5], atol=1e-12, rtol=0.0)
    assert fractions[0, 1] > 0.999
    assert fractions[2, 0] > 0.999


def test_synthetic_potentiometry_recovers_pka():
    rng = np.random.default_rng(1234)
    true_system = make_simple_acid_base_system(
        analytical_concentration=1.0e-2,
        pka=[5.0],
        base_charge=-1,
    )
    volumes = np.linspace(0.0, 18.0, 45)
    experiment_template = PotentiometryExperiment(
        initial_volume=10.0,
        titrant_volumes=volumes,
        analyte_concentration=1.0e-2,
        titrant_concentration=1.0e-2,
        titrant_type="base",
    )
    pH = simulate_pH_titration(experiment_template, true_system)
    measured = pH + rng.normal(0.0, 1.0e-3, size=pH.shape)
    experiment = PotentiometryExperiment(
        initial_volume=10.0,
        titrant_volumes=volumes,
        measured_pH=measured,
        analyte_concentration=1.0e-2,
        titrant_concentration=1.0e-2,
        titrant_type="base",
    )
    start_system = make_simple_acid_base_system(
        analytical_concentration=1.0e-2,
        pka=[4.4],
        base_charge=-1,
    )
    result = fit_potentiometry(experiment, start_system, initial_pka=[4.4])
    assert result.success
    assert abs(result.fitted_pka[0] - 5.0) < 0.03


def test_synthetic_spectroscopy_recovers_pka():
    rng = np.random.default_rng(5678)
    pH = np.linspace(2.0, 8.0, 41)
    fractions = distribution_fractions_from_pH(pH, [5.0])
    signal = fractions @ np.asarray([0.15, 1.25])
    signal = signal + rng.normal(0.0, 1.0e-4, size=signal.shape)
    dataset = SpectroscopicAcidBaseDataset(pH=pH, signal=signal)
    start_system = make_simple_acid_base_system(pka=[4.2])
    result = fit_spectroscopy_acid_base(dataset, start_system, initial_pka=[4.2])
    assert result.success
    assert abs(result.fitted_pka[0] - 5.0) < 0.02


def test_synthetic_nmr_recovers_pka():
    rng = np.random.default_rng(9012)
    pH = np.linspace(2.0, 8.0, 41)
    fractions = distribution_fractions_from_pH(pH, [5.0])
    limiting_shifts = np.asarray(
        [
            [7.10, 8.20, 6.95],
            [8.80, 6.40, 7.55],
        ]
    )
    shifts = fractions @ limiting_shifts
    shifts = shifts + rng.normal(0.0, 1.0e-4, size=shifts.shape)
    dataset = NMRAcidBaseDataset(pH=pH, shifts=shifts, nuclei_labels=["H1", "H2", "H3"])
    start_system = make_simple_acid_base_system(pka=[5.8])
    result = fit_nmr_acid_base(dataset, start_system, initial_pka=[5.8])
    assert result.success
    assert abs(result.fitted_pka[0] - 5.0) < 0.02


def test_xlsx_potentiometry_import_recovers_pka(tmp_path):
    true_system = make_simple_acid_base_system(
        analytical_concentration=1.0e-2,
        pka=[5.0],
        base_charge=-1,
    )
    volumes = np.linspace(0.0, 18.0, 30)
    experiment = PotentiometryExperiment(
        initial_volume=10.0,
        titrant_volumes=volumes,
        analyte_concentration=1.0e-2,
        titrant_concentration=1.0e-2,
        titrant_type="base",
    )
    pH = simulate_pH_titration(experiment, true_system)
    path = tmp_path / "acid_base.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"volume_mL": volumes, "pH": pH}).to_excel(
            writer,
            sheet_name="Pot",
            index=False,
        )

    result = run_acid_base(
        {
            "file_path": str(path),
            "sheet_name": "Pot",
            "data_type": "potentiometry",
            "pka_initial": "4.2",
            "analyte_concentration": 1.0e-2,
            "titrant_concentration": 1.0e-2,
            "initial_volume": 10.0,
            "titrant_type": "base",
        },
        progress_cb=lambda _msg: None,
    )
    assert result["success"] is True
    assert abs(float(result["constants"][0]["value"]) - 5.0) < 1.0e-6
