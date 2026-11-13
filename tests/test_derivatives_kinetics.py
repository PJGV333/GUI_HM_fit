"""First-order kinetic sensitivities against analytic and independent SciPy fits."""
import numpy as np
import pytest


@pytest.mark.parametrize("k,a0,b0", [(0.0, 2.0, 0.1), (0.8, 1.0, 0.3), (10.0, 0.002, 0.001)])
def test_first_order_direct_sensitivity_matches_analytic_and_scipy_difference(k, a0, b0):
    from hmfit_core.derivatives.kinetics import solve_first_order_sensitivity
    from hmfit.kinetics.mechanism_editor.parser import parse_mechanism
    from hmfit.kinetics.model.kinetics_model import KineticsModel

    times = np.linspace(0.2, 2.2, 21)
    initial = np.array([a0, b0])
    c, sens, diagnostic = solve_first_order_sensitivity(
        times, initial, k, rtol=1e-10, atol=1e-13)
    dt = times - times[0]
    a = a0 * np.exp(-k * dt)
    exact_c = np.column_stack([a, b0 + a0 - a])
    exact_s = np.column_stack([-dt * a, dt * a])
    np.testing.assert_allclose(c, exact_c, rtol=2e-8, atol=2e-12)
    np.testing.assert_allclose(sens, exact_s, rtol=2e-7, atol=2e-11)
    np.testing.assert_allclose(sens.sum(axis=1), 0, atol=2e-11)
    assert diagnostic["nfev"] > 0

    model = KineticsModel(parse_mechanism("species: A, B\nA -> B ; k1\n"))
    step = 1e-4
    plus = model.solve_concentrations(times, initial, {"k1": k + step}, rtol=1e-10, atol=1e-13)
    minus = model.solve_concentrations(times, initial, {"k1": k - step}, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(sens, (plus-minus)/(2*step), rtol=1e-5, atol=1e-9)


def test_first_order_initial_condition_parameter_dependence():
    from hmfit_core.derivatives.kinetics import solve_first_order_sensitivity
    from hmfit.kinetics.mechanism_editor.parser import parse_mechanism
    from hmfit.kinetics.model.kinetics_model import KineticsModel

    t = np.linspace(0., 3., 25)
    initial = np.array([1., 0.2])
    d_initial = np.array([0.3, -0.1])
    k, step = 0.7, 1e-4
    c, sens, _ = solve_first_order_sensitivity(t, initial, k, initial_sensitivity=d_initial,
                                               rtol=1e-11, atol=1e-14)
    model = KineticsModel(parse_mechanism("species: A, B\nA -> B ; k1\n"))
    plus = model.solve_concentrations(t, initial + step*d_initial, {"k1": k+step}, rtol=1e-11, atol=1e-14)
    minus = model.solve_concentrations(t, initial - step*d_initial, {"k1": k-step}, rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(c[0], initial)
    np.testing.assert_allclose(sens[0], d_initial)
    np.testing.assert_allclose(sens, (plus-minus)/(2*step), rtol=2e-6, atol=1e-9)


def test_first_order_rejects_invalid_time_and_rate():
    from hmfit_core.derivatives.kinetics import solve_first_order_sensitivity

    with pytest.raises(ValueError, match="time"):
        solve_first_order_sensitivity([0., 0.], [1., 0.], 1.)
    with pytest.raises(ValueError, match="rate"):
        solve_first_order_sensitivity([0., 1.], [1., 0.], -1.)
