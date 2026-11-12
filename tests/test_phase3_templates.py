"""Fit synthetic template mechanisms from their untouched automatic table."""
import numpy as np
import pytest
pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtWidgets import QApplication
from hmfit.kinetics.data.fit_dataset import KineticsFitDataset
from hmfit.kinetics.mechanism_editor.parser import parse_mechanism
from hmfit.kinetics.model.kinetics_model import KineticsModel
from hmfit_gui_qt.tabs.kinetics_tab import KineticsTab, _automatic_rate_parameters, _run_kinetics_fit
from hmfit_gui_qt.widgets.kinetics_model_opt_plots import MECHANISM_TEMPLATES

CASES = MECHANISM_TEMPLATES + [("Eyring", "species: A, B\nA -> B ; k1\neyring: k1(dH=H1, dS=S1)\n")]


@pytest.mark.parametrize("label,text", CASES, ids=[case[0] for case in CASES])
def test_synthetic_templates_reach_noise_minimum(label, text):
    app = QApplication.instance() or QApplication([])
    model = KineticsModel(parse_mechanism(text))
    t = np.linspace(0, 100, 36)
    temps = [285., 300., 315.] if model.mechanism.temp_models else [300.]
    rng = np.random.default_rng(73)
    data = []
    for temp in temps:
        y = {sp: (.01 if sp in ("A", "E") else .02 if sp in ("B", "S") else 0.) for sp in model.dynamic_species}
        fixed = {sp: .02 for sp in model.fixed_species}
        data.append(KineticsFitDataset(t=t, y0=y, fixed_conc=fixed, temperature=temp,
                    D=np.zeros((len(t), 4)), time_unit="s", concentration_unit="M"))
    truth = _automatic_rate_parameters(model, data)
    for name in truth:
        if name.startswith("Ea") or name.startswith("H"):
            truth[name] = 1000.
        elif name.startswith("S1"):
            truth[name] -= 2.
        else:
            truth[name] *= 1.3
    noise_ssq = 0.
    for dataset in data:
        C = model.solve_concentrations(t, dataset.y0, truth, dataset)
        noise = rng.normal(0, 2e-5, (len(t), 4))
        dataset.D = C @ rng.uniform(.2, 2., (len(model.dynamic_species), 4)) + noise
        noise_ssq += np.sum(noise ** 2)
    tab = KineticsTab()
    tab._datasets = data
    tab.model_opt_plots.set_mechanism_text(text)
    tab._on_validate_mechanism()
    result = _run_kinetics_fit(tab._collect_fit_config(data))
    assert result["ssq"] <= 1.05 * noise_ssq, (label, result["ssq"], noise_ssq, result["params"])
    tab.deleteLater()
    assert app
