"""Experimental JAX backend contract and time-grid limitations."""
import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtWidgets import QApplication
from hmfit_gui_qt.tabs.kinetics_tab import KineticsTab


def test_backend_key_and_legacy_configuration():
    app = QApplication.instance() or QApplication([])
    tab = KineticsTab()
    combo = tab.model_opt_plots.combo_backend
    assert combo.currentData() == "scipy"
    tab._apply_config({"backend": "JAX"})
    assert combo.currentData() == "jax"
    assert "experimental" in combo.currentText().lower()
    assert "without error control" in combo.toolTip()
    tab.deleteLater()
    assert app is not None


def test_jax_rk4_fine_and_coarse_grid():
    pytest.importorskip("jax")
    from hmfit.kinetics.model.kinetics_model import KineticsModel
    from hmfit.kinetics.mechanism_editor.parser import parse_mechanism
    model = KineticsModel(parse_mechanism("species: A, B, C\nA -> B ; k1\nB -> C ; k2\n"))
    y0 = {"A": 1.0, "B": 0.0, "C": 0.0}
    params = {"k1": 1.0, "k2": 0.2}
    fine = np.linspace(0, 8, 401)
    a = np.asarray(model.solve_concentrations_jax(fine, y0, params))
    b = model.solve_concentrations(fine, y0, params)
    assert np.max(np.abs(a - b)) < 2e-5
    coarse = np.linspace(0, 8, 5)
    a = np.asarray(model.solve_concentrations_jax(coarse, y0, params))
    b = model.solve_concentrations(coarse, y0, params)
    assert np.max(np.abs(a - b)) > 0.05


def test_jax_projection_drops_roundoff_rank_for_all_weight_shapes():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from hmfit_gui_qt.tabs.kinetics_tab import _jax_solve_linear_matrix
    from hmfit.kinetics.observation.linear_matrix import solve_linear_matrix
    c = np.array([[1., 1.], [0., 1e-12], [0., 0.]])
    d = np.array([[1., 2.], [1., -1.], [0., 0.]])
    with jax.enable_x64():
        # The old default pinv retains this round-off direction (coefficients ~1e12).
        assert np.max(np.abs(np.asarray(jnp.linalg.pinv(jnp.asarray(c)) @ jnp.asarray(d)))) > 1e11
        for w in (None, np.ones((3, 1)), np.ones((3, 2))):
            expected, _ = solve_linear_matrix(c, d, weights=w)
            got = _jax_solve_linear_matrix(jnp.asarray(c), jnp.asarray(d), None if w is None else jnp.asarray(w))
            np.testing.assert_allclose(np.asarray(got), expected, atol=1e-10)
