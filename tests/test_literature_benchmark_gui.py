"""Literature benchmark: the harness uses the same GUI paths as the NMR tab.

* the stoichiometry map is the one ``NMRTab._inject_stoichiometry_map`` builds;
* the "analytic" errors used for sigma, correlations and limiting shifts equal those
  of the Errors tab (``ModelOptPlotsWidget._compute_errors_from_context``) for the
  context the NMR tab hands over;
* the Errors tab uses all observations with a valid stoichiometric denominator (§6.4).
"""
from __future__ import annotations

import os
import tempfile
import types

import numpy as np
import pytest

from validation import literature_benchmark as lb

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _errors_tab_output(case):
    """Fit with run_nmr_fit, hand the result to the NMR tab's context builder and run the
    Errors tab's analytic method on it (no reimplementation)."""
    from hmfit_core.api import run_nmr_fit
    from hmfit_gui_qt.tabs.nmr_tab import NMRTab
    from hmfit_gui_qt.widgets.model_opt_plots import ModelOptPlotsWidget

    _qapp()
    conc, shifts = lb.numeric_table(case)
    with tempfile.TemporaryDirectory() as td:
        cfg = lb.gui_config(case, os.path.join(td, "c.xlsx"), conc, shifts)
        result = run_nmr_fit(cfg, progress_cb=lambda _m: None)
    captured = {}
    fake = types.SimpleNamespace(
        _last_config=dict(cfg), _last_fit_context=None,
        model_opt_plots=types.SimpleNamespace(
            set_errors_context=lambda ctx, auto_compute=False: captured.update(ctx or {})))
    fake._refit_from_data = lambda *a, **k: None
    NMRTab._update_errors_context_from_result(fake, result)
    widget = ModelOptPlotsWidget()
    out = widget._compute_errors_from_context(captured, options={"method": "analytic"})
    return result, cfg, out


@pytest.mark.parametrize("cid", [c for c in lb.case_ids("fit") if lb.supported(lb.load_case(c))[0]])
def test_stoichiometry_map_is_the_nmr_tab_map(cid):
    from hmfit_gui_qt.tabs.nmr_tab import NMRTab

    case = lb.load_case(cid)
    with tempfile.TemporaryDirectory() as td:
        conc, shifts = lb.numeric_table(case)
        cfg = lb.gui_config(case, os.path.join(td, "c.xlsx"), conc, shifts)
    expected = cfg.pop("stoichiometry_map")
    fake = types.SimpleNamespace(log=types.SimpleNamespace(append_text=lambda *_a: None))
    NMRTab._inject_stoichiometry_map(fake, cfg, debug=False)
    np.testing.assert_array_equal(np.asarray(cfg["stoichiometry_map"], float), np.asarray(expected, float))


def test_harness_errors_equal_the_errors_tab():
    case = lb.load_case("r09a-r1-5sp")
    result, cfg, out = _errors_tab_output(case)
    mine = lb.errors_tab_analytic(result, cfg)
    np.testing.assert_allclose(np.asarray(out["se_log10"], float), mine["se"], rtol=1e-10)
    np.testing.assert_allclose(np.asarray(out["corr"], float), mine["corr"], rtol=1e-10, atol=1e-12)
    assert int(out["dof"]) == mine["dof"]


def test_errors_tab_uses_every_observed_value():
    case = lb.load_case("r09a-r1-11-me")
    _result, _cfg, out = _errors_tab_output(case)
    dof = case["dof_identity"]
    assert int(out["dof"]) == dof["m"] - dof["p"]
