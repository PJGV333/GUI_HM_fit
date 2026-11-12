"""Potentiometry report uses analytical errors and unweighted signal RMS."""
import numpy as np
import pandas as pd
import pytest

from hmfit_core.acid_base import make_simple_acid_base_system
from hmfit_core.potentiometry import PotentiometryExperiment, simulate_pH_titration
from hmfit_core.run_acid_base import run_acid_base
from hmfit_core.exports import write_results_xlsx


@pytest.mark.parametrize("emf", [False, True])
def test_potentiometry_summary_fixed_free_rms_and_export(tmp_path, emf):
    volumes = np.linspace(0., 15., 30)
    system = make_simple_acid_base_system(analytical_concentration=.01, pka=[8., 4.], base_charge=-2)
    exp = PotentiometryExperiment(initial_volume=10., titrant_volumes=volumes,
                                   analyte_concentration=.01, titrant_concentration=.01, titrant_type="base")
    ph = simulate_pH_titration(exp, system)
    signal = (-59.16 * ph if emf else ph) + np.random.default_rng(4).normal(0., .01, len(ph))
    cfg = {"data_type": "potentiometry", "titrant_volume": volumes, "observed_signal": signal,
           "signal_type": "mV" if emf else "pH", "pka_initial": "8,4", "pka_fit_mask": [True, False],
           "analyte_concentration": .01, "titrant_concentration": .01, "initial_volume": 10.,
           "base_charge": -2, "titrant_type": "base", "sigma_E" if emf else "sigma_pH": .1}
    if emf:
        cfg.update({"electrode_e0": 0., "electrode_slope": -59.16})
    result = run_acid_base(cfg, progress_cb=lambda _: None)
    text = result["results_text"]
    lines = text.splitlines()
    assert any("pKa1 (HL ⇌ L + H⁺)" in line and "±" in line for line in lines)
    assert any("pKa2 (H2L ⇌ HL + H⁺)" in line and "fixed" in line for line in lines)
    assert text.index("pKa1") < text.index("pKa2") < text.index("RMS =") < text.index("Estadísticos adicionales")
    assert "σ = 1" in text and ("mV²" if emf else "pH²") in text
    assert result["statistics"]["RMS_unit"] == ("mV" if emf else "pH")
    residuals = np.asarray(result["errors_context"]["residuals"]) * .1
    assert result["statistics"]["RMS"] == pytest.approx(np.sqrt(np.mean(residuals ** 2)))
    assert result["statistics"]["N"] == len(volumes)
    assert result["statistics"]["dof"] == len(volumes) - 1
    path = tmp_path / "pot.xlsx"
    write_results_xlsx(path, constants=result["constants"], statistics=result["statistics"],
                       results_text=text, export_data=result["export_data"])
    summary = pd.read_excel(path, sheet_name="Fit_summary", header=1)
    exported = "\n".join(summary["line"].dropna().astype(str))
    assert "pKa1" in exported and "±" in exported and "fixed" in exported
    assert "RMS =" in exported and "Estadísticos adicionales" in exported
