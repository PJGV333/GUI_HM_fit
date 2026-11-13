"""Kinetics fit and Errors must share the same residual variance convention."""
import types
from typing import Any, cast

import numpy as np
import pytest


@pytest.mark.gui
def test_nonpositive_kinetics_dof_never_produces_residual_variance():
    from tests.test_phase2_invariants_matrix import _kinetics_case

    context, _ = _kinetics_case("dof_le_0")
    assert context["dof"] <= 0
    assert context["sigma2"] is None or np.isnan(context["sigma2"])


@pytest.mark.gui
@pytest.mark.parametrize("configuration", ["none_fixed", "mixed", "all_fixed"])
def test_positive_dof_variance_matches_fit_residuals(configuration):
    from tests.test_phase2_invariants_matrix import _kinetics_case

    context, _ = _kinetics_case(configuration)
    assert context["dof"] > 0
    residuals = np.asarray(context["residuals"])
    assert context["sigma2"] == pytest.approx(float(residuals @ residuals) / context["dof"])


@pytest.mark.gui
def test_kinetics_errors_context_uses_authoritative_dof_for_variance():
    from hmfit_gui_qt.tabs.kinetics_tab import KineticsTab

    captured = {}
    fake = types.SimpleNamespace(model_opt_plots=types.SimpleNamespace(
        set_errors_context=lambda ctx, auto_compute=False: captured.update(ctx)))
    result = {"param_names": ["k1", "k2"], "params": {"k1": 1.0, "k2": 2.0},
              "fixed_mask": [False, True], "n_linear": 2, "dof": 2,
              "residuals": np.array([1., 2., 0., 0., 0.])}
    KineticsTab._update_errors_context_from_result(cast(Any, fake), result)
    assert captured["sigma2"] == pytest.approx(2.5)
