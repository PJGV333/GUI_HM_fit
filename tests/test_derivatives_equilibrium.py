"""Independent references for the experimental 1:1 equilibrium sensitivity."""
import numpy as np
import pytest
from scipy.optimize import brentq


def _association_11(total, log_k):
    htot, gtot = total
    K = 10.0 ** log_k
    complex_ = brentq(lambda x: x - K * (htot - x) * (gtot - x),
                      0.0, min(htot, gtot), xtol=1e-28, rtol=np.float64(1e-14))
    return np.array([htot - complex_, gtot - complex_, complex_])


@pytest.mark.parametrize("total,log_k", [
    ([1e-3, 2e-3], 3.0),
    ([1e-6, 1e-4], 6.0),
    ([2e-2, 1e-5], 2.0),
])
def test_11_implicit_sensitivity_matches_analytic_and_centered_differences(total, log_k):
    from hmfit_core.derivatives.equilibrium import association_11_logk_sensitivity

    c = _association_11(total, log_k)
    deriv, diagnostics = association_11_logk_sensitivity(total, c[:2], log_k)
    K = 10.0 ** log_k
    db = np.log(10.0) * c[2] / (1.0 + K * (c[0] + c[1]))
    np.testing.assert_allclose(deriv, [-db, -db, db], rtol=2e-12, atol=1e-20)
    h = 1e-4
    fd = (_association_11(total, log_k + h) - _association_11(total, log_k - h)) / (2*h)
    np.testing.assert_allclose(deriv, fd, rtol=2e-7, atol=1e-14)
    np.testing.assert_allclose([deriv[0] + deriv[2], deriv[1] + deriv[2]], 0, atol=1e-18)
    assert diagnostics["condition_number"] >= 1.0


def test_11_uses_existing_equilibrium_solver_at_tight_tolerance():
    from hmfit_core.derivatives.equilibrium import association_11_logk_sensitivity
    from hmfit_core.solvers import NewtonRaphson

    total = np.array([0.001, 0.002])
    model = np.array([[1., 0., 1.], [0., 1., 1.]])
    solver = NewtonRaphson(total[None, :], model, nas=[], model_sett="Free", tol=1e-16)
    _, concentrations = solver.concentraciones(np.array([3.0]))
    expected = _association_11(total, 3.0)
    np.testing.assert_allclose(concentrations[0], expected, rtol=1e-10, atol=1e-16)
    derivative, _ = association_11_logk_sensitivity(total, concentrations[0, :2], 3.0)
    h = 1e-4
    fd = (_association_11(total, 3.0+h) - _association_11(total, 3.0-h))/(2*h)
    np.testing.assert_allclose(derivative, fd, rtol=2e-7, atol=1e-14)


def test_11_rejects_ill_conditioned_balances():
    from hmfit_core.derivatives.equilibrium import association_11_logk_sensitivity

    free = np.array([1e-14, 1e-3])
    bound = 1e16 * free[0] * free[1]
    with pytest.raises(ValueError, match="ill-conditioned"):
        association_11_logk_sensitivity(free + bound, free, 16.0)


def test_11_rejects_unbalanced_or_negative_concentrations():
    from hmfit_core.derivatives.equilibrium import association_11_logk_sensitivity

    with pytest.raises(ValueError, match="balance"):
        association_11_logk_sensitivity([1e-3, 2e-3], [1e-3, 2e-3], 3.)
    with pytest.raises(ValueError, match="nonnegative"):
        association_11_logk_sensitivity([1e-3, 2e-3], [-1e-4, 1e-3], 3.)
