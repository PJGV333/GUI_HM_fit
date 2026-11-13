"""Automatic kinetics starts, unit metadata and temperature laws."""
import numpy as np
import pytest
from hmfit.kinetics.data.fit_dataset import KineticsFitDataset
from hmfit.kinetics.mechanism_editor.parser import parse_mechanism
from hmfit.kinetics.model.kinetics_model import KineticsModel
from hmfit.kinetics.fit.rate_parameters import _automatic_rate_parameters, _eyring_dataset_warnings


def ds(tmax=100, temp=298.15, time="s", conc="M"):
    return KineticsFitDataset(t=np.linspace(0, tmax, 11), y0={"A": .001, "B": .002, "C": 0., "P": 0.},
                              D=np.ones((11, 1)), temperature=temp, time_unit=time, concentration_unit=conc)


def test_automatic_values_refresh_preserve_user_and_legacy_config():
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from hmfit_gui_qt.tabs.kinetics_tab import KineticsTab
    app = QApplication.instance() or QApplication([])
    tab = KineticsTab()
    tab.model_opt_plots.set_mechanism_text("species: A, B, C\nA -> B ; k1\nB -> C ; k2\n")
    tab._on_validate_mechanism()
    tab._datasets = [ds()]
    tab._refresh_dataset_list()
    w = tab.model_opt_plots
    state = w.get_params_state()
    assert state.values == pytest.approx([.01, .01 / 3])
    assert state.log_mask == [True, True]
    assert "s⁻¹" in w.params_table.item(0, 6).text()
    w.params_table.item(0, 1).setText(".2")
    w.params_table.item(1, 5).setCheckState(Qt.CheckState.Unchecked)
    tab._datasets[0].t = np.linspace(0, 200, 11)
    tab._refresh_dataset_list()
    state = w.get_params_state()
    assert state.values == pytest.approx([.2, .005 / 3])
    assert state.log_mask == [True, False]
    tab._on_validate_mechanism()
    assert w.get_params_state().values[0] == pytest.approx(.2)
    tab._apply_config({"mechanism_text": w.mechanism_text(), "params": {"k1": {"value": .7, "log": False}, "k2": {"value": .3, "log": True}}})
    tab._refresh_dataset_list()
    assert w.get_params_state().values == pytest.approx([.7, .3])
    assert w.get_params_state().log_mask == [False, True]
    tab.deleteLater()
    assert app


def test_units_displayed_in_table_and_fit_results():
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication
    from hmfit_gui_qt.tabs.kinetics_tab import KineticsTab, _run_kinetics_fit
    app = QApplication.instance() or QApplication([])
    tab = KineticsTab()
    text = "species: A, B, P\nA + B -> P ; k\n"
    model = KineticsModel(parse_mechanism(text))
    d = ds(time="min", conc="mM")
    d.y0 = {"A": 1., "B": 2., "P": 0.}
    c = model.solve_concentrations(d.t, d.y0, {"k": .005}, d)
    d.D = c @ np.array([[.2, 1.], [.6, .4], [1.4, .8]])
    tab._datasets = [d]
    tab.model_opt_plots.set_mechanism_text(text)
    tab._on_validate_mechanism()
    assert tab.model_opt_plots.params_table.item(0, 6).text() == "mM⁻¹·min⁻¹"
    result = _run_kinetics_fit(tab._collect_fit_config([d]))
    assert "mM⁻¹·min⁻¹" in result["results_text"]
    tab.deleteLater()
    assert app


def test_first_order_start_and_factor_three_without_gui():
    m = KineticsModel(parse_mechanism("species: A, B, C\nA -> B ; k1\nB -> C ; k2\n"))
    assert _automatic_rate_parameters(m, [ds()]) == pytest.approx({"k1": .01, "k2": .01 / 3})


def test_shared_rate_uses_max_and_intermediate_fallback():
    m = KineticsModel(parse_mechanism("species: A, B, C, P\nA + B -> P ; k\nA + C -> P ; k\n"))
    d = ds()
    d.y0 = {"A": .001, "B": .002, "C": .01, "P": 0}
    assert _automatic_rate_parameters(m, [d])["k"] == pytest.approx(1.)
    m = KineticsModel(parse_mechanism("species: A, B, C, P\nB + C -> P ; k\n"))
    d.y0 = {"A": .004, "B": 0, "C": 0, "P": 0}
    assert _automatic_rate_parameters(m, [d])["k"] == pytest.approx(2.5)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("kind", ["arrhenius", "eyring"])
def test_temperature_law_inverts_to_k0(order, kind):
    reaction = "A -> P" if order == 1 else "A + B -> P"
    law = "arrhenius: k(A=A0, Ea=Ea0)" if kind == "arrhenius" else "eyring: k(dH=H, dS=S)"
    m = KineticsModel(parse_mechanism(f"species: A, B, P\n{reaction} ; k\n{law}\n"))
    datasets = [ds(temp=t, time="min", conc="mM") for t in (280., 300., 330.)]
    p = _automatic_rate_parameters(m, datasets)
    k0 = 1 / (100 * (.002 if order == 2 else 1))
    rate_si = m.param_resolver.resolve(p, 300.)["k"]
    factor = m._eyring_unit_factors(datasets[0]).get("k", 1.)
    assert rate_si * factor == pytest.approx(k0, rel=1e-12)


def test_eyring_unit_scaling_and_warnings():
    from hmfit.kinetics.model.temperature import R_GAS
    from hmfit.kinetics.data.units import eyring_unit_factors
    one_min = ds(time="min", conc="M")
    bimol_mm = ds(time="s", conc="mM")
    fmin, _ = eyring_unit_factors(one_min, 1)
    fmm, _ = eyring_unit_factors(bimol_mm, 2)
    assert R_GAS * np.log(fmin) == pytest.approx(34.0422748)
    assert R_GAS * np.log(fmm) == pytest.approx(-57.4342730)
    m = KineticsModel(parse_mechanism("species: A, B\nA -> B ; k\neyring: k(dH=H, dS=S)\n"))
    empty = ds(time="", conc="")
    warnings = _eyring_dataset_warnings(m, [empty])
    assert len(warnings) == 2
    assert all(part in warnings[1] for part in ("assuming M", "mM", "57", "bimolecular"))
    with pytest.raises(ValueError, match="Unrecognized time"):
        _eyring_dataset_warnings(m, [ds(time="fortnight")])
    with pytest.raises(ValueError, match="Unrecognized concentration"):
        _eyring_dataset_warnings(m, [ds(conc="bananas")])


def test_eyring_equivalent_trajectories_in_minutes_and_millimolar():
    text = "species: A, B, P\nA + B -> P ; k\neyring: k(dH=H, dS=S)\n"
    model = KineticsModel(parse_mechanism(text))
    seconds = np.linspace(0, 120, 20)
    si = ds(time="s", conc="M")
    si.y0 = {"A": .001, "B": .002, "P": 0.}
    other = ds(time="min", conc="mM")
    other.y0 = {"A": 1., "B": 2., "P": 0.}
    params = {"H": 0., "S": -220.}
    a = model.solve_concentrations(seconds, si.y0, params, si)
    b = model.solve_concentrations(seconds / 60, other.y0, params, other)
    np.testing.assert_allclose(a, b / 1000, rtol=2e-7, atol=3e-11)
