"""
Reliability of the kinetics core (no GUI): defects found in the November 2026 review.

Each defect was reproduced on commit 277b255 before the fix; the docstrings give the old behaviour.
"""
import numpy as np
import pytest

from hmfit.kinetics.data.fit_dataset import KineticsFitDataset
from hmfit.kinetics.fit.objective import GlobalKineticsObjective
from hmfit.kinetics.fit.optimizer import (
    convergence_notes,
    exchange_equivalent_parameters,
    fit_global,
    randomize_start,
    scaled_least_squares,
)
from hmfit.kinetics.mechanism_editor.parser import parse_mechanism
from hmfit.kinetics.model.kinetics_model import KineticsModel
from hmfit.kinetics.observation.linear_matrix import effective_rank


def _case(mech, k_true, y0, t, nch=6, seed=2, amp=1.0, sigma=2e-6):
    model = KineticsModel(parse_mechanism(mech))
    C = model.solve_concentrations(t, y0, k_true)
    A = (np.random.default_rng(seed).random((C.shape[1], nch)) + 0.2) * amp
    D0 = C @ A
    noise = np.random.default_rng(seed + 100).normal(0.0, sigma * amp, D0.shape)
    return model, C, D0, noise


def _objective(model, t, y0, D, names, **kw):
    return GlobalKineticsObjective(model, [KineticsFitDataset(t=t, y0=y0, D=D)], param_names=names, **kw)


BIMOLECULAR = "species: A, B, C, D\nA + B -> C ; k1\nC -> D ; k2\n"
CONSECUTIVE = "species: A, B, C\nA -> B ; k1\nB -> C ; k2\n"


def test_conservation_laws_do_not_stall_the_fit():
    """
    A + B -> C -> D has two conservation laws: the 4 concentration columns span 3 dimensions.
    The round-off singular value (~1e-13) was kept or dropped erratically by lstsq, the
    objective was noisy and the optimiser returned the initial k1 (20.000 for a true 30)
    as a "converged" fit.
    """
    t = np.linspace(0.0, 600.0, 60)
    y0 = {"A": 1e-3, "B": 2e-3, "C": 0.0, "D": 0.0}
    model, C, D0, noise = _case(BIMOLECULAR, {"k1": 30.0, "k2": 0.01}, y0, t)
    assert effective_rank(C) == 3
    obj = _objective(model, t, y0, D0 + noise, ["k1", "k2"])
    for start in ([20.0, 0.02], [100.0, 0.001]):
        res = fit_global(obj, np.array(start), bounds=(0.0, np.inf), max_nfev=300)
        assert res.ssq <= 1.05 * float(np.sum(noise**2))
        assert res.params["k1"] == pytest.approx(30.0, rel=0.02)
        assert res.params["k2"] == pytest.approx(0.01, rel=0.02)
        assert abs(res.params["k1"] - 30.0) < 4.0 * res.errors[0]
        # 3 independent concentration profiles x 6 channels, not 4 x 6
        assert res.result.n_linear == 18
        assert not res.result.convergence_notes


def test_objective_is_smooth_with_rank_deficient_concentrations():
    """Forward-difference Jacobian with SciPy's default step agrees with a wide central one."""
    t = np.linspace(0.0, 600.0, 60)
    y0 = {"A": 1e-3, "B": 2e-3, "C": 0.0, "D": 0.0}
    model, _C, D0, noise = _case(BIMOLECULAR, {"k1": 30.0, "k2": 0.01}, y0, t)
    obj = _objective(model, t, y0, D0 + noise, ["k1", "k2"])
    x = np.array([30.0, 0.01])

    def jac(h, central):
        cols = []
        for i in range(2):
            e = np.zeros(2)
            e[i] = h * x[i]
            if central:
                cols.append((obj.residuals(x + e) - obj.residuals(x - e)) / (2 * e[i]))
            else:
                cols.append((obj.residuals(x + e) - obj.residuals(x)) / e[i])
        return np.column_stack(cols)

    ref = jac(1e-4, True)
    err = np.linalg.norm(jac(1.5e-8, False) - ref, axis=0) / np.linalg.norm(ref, axis=0)
    assert np.all(err < 1e-3)  # was ~2000 (pure noise)


def test_fit_does_not_depend_on_the_units_of_the_signal():
    """
    gtol is an absolute tolerance on J^T r: with mM-scale signals the fit stopped early
    (k1 = 0.050571 instead of 0.049864) or at the initial values; the same data x1000 converged.
    """
    t = np.linspace(0.0, 600.0, 60)
    y0 = {"A": 1e-3, "B": 0.0, "C": 0.0}
    model, _C, D0, noise = _case(CONSECUTIVE, {"k1": 0.05, "k2": 0.004}, y0, t, seed=1)
    results = []
    for factor in (1.0, 1.0e3):
        obj = _objective(model, t, y0, (D0 + noise) * factor, ["k1", "k2"])
        results.append(fit_global(obj, np.array([0.2, 0.02]), bounds=(0.0, np.inf), max_nfev=300))
    small, large = results
    for name in ("k1", "k2"):
        assert small.params[name] == pytest.approx(large.params[name], rel=1e-5)
    assert small.ssq * 1.0e6 == pytest.approx(large.ssq, rel=1e-4)
    assert np.allclose(small.errors, large.errors, rtol=1e-3)
    assert small.ssq <= 1.02 * float(np.sum(noise**2))


def test_scaled_least_squares_reports_original_units():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = X @ np.array([2.0, -1.0]) + rng.normal(0, 0.01, 30)
    y_small = y * 1e-4

    def residual(theta):
        return y_small - (X @ theta) * 1e-4

    res = scaled_least_squares(residual, np.zeros(2), scale=float(np.sqrt(np.mean(y_small**2))))
    assert np.allclose(res.fun, residual(res.x))
    assert res.cost == pytest.approx(0.5 * float(np.sum(residual(res.x) ** 2)))
    assert np.allclose(res.jac, -X * 1e-4, rtol=1e-5, atol=1e-12)
    assert np.allclose(res.x, np.linalg.lstsq(X, y, rcond=None)[0], rtol=1e-6)
    # analytic Jacobian is scaled consistently
    res2 = scaled_least_squares(residual, np.zeros(2), scale=1e-4, jac=lambda th: -X * 1e-4)
    assert np.allclose(res2.x, res.x, rtol=1e-8)
    assert np.allclose(res2.jac, -X * 1e-4)


def test_unconverged_and_unmoved_fits_are_reported():
    t = np.linspace(0.0, 600.0, 60)
    y0 = {"A": 1e-3, "B": 0.0, "C": 0.0}
    model, _C, D0, noise = _case(CONSECUTIVE, {"k1": 0.05, "k2": 0.004}, y0, t, seed=1)
    obj = _objective(model, t, y0, D0 + noise, ["k1", "k2"])
    # Rate constants 1e5 times too fast: zero sensitivity, the optimiser cannot move.
    stuck = fit_global(obj, np.array([1.0e4, 1.0e4]), bounds=(0.0, np.inf), max_nfev=300)
    assert any("did not move from the initial values" in note for note in stuck.notes)
    # Evaluation budget exhausted.
    short = fit_global(obj, np.array([0.5, 0.5]), bounds=(0.0, np.inf), max_nfev=2)
    assert any("evaluation limit" in note for note in short.notes)
    good = fit_global(obj, np.array([0.03, 0.006]), bounds=(0.0, np.inf), max_nfev=300)
    assert not good.result.convergence_notes

    class _R:
        status, message = -1, "boom"

    assert any("failure" in n for n in convergence_notes(_R(), [1.0], [2.0], ["k"]))


def test_slow_fast_ambiguity_is_detected():
    t = np.linspace(0.0, 600.0, 60)
    y0 = {"A": 1e-3, "B": 0.0, "C": 0.0}
    model, _C, D0, noise = _case(CONSECUTIVE, {"k1": 0.05, "k2": 0.004}, y0, t, seed=1)
    obj = _objective(model, t, y0, D0 + noise, ["k1", "k2"])
    res = fit_global(obj, np.array([0.03, 0.006]), bounds=(0.0, np.inf), max_nfev=300)
    x = np.array([res.params["k1"], res.params["k2"]])
    pairs = exchange_equivalent_parameters(obj, x, ["k1", "k2"], ssq=res.ssq)
    assert [(p["a"], p["b"]) for p in pairs] == [("k1", "k2")]
    assert pairs[0]["ssq_ratio"] == pytest.approx(1.0, abs=1e-3)
    # the exchanged values really are an equally good fit
    swapped = fit_global(obj, x[::-1].copy(), bounds=(0.0, np.inf), max_nfev=300)
    assert swapped.ssq == pytest.approx(res.ssq, rel=1e-3)
    assert swapped.params["k1"] == pytest.approx(res.params["k2"], rel=1e-3)
    # bounds that express prior knowledge remove the alternative
    bounds = (np.array([0.01, 0.0]), np.array([1.0, 0.01]))
    assert exchange_equivalent_parameters(obj, x, ["k1", "k2"], ssq=res.ssq, bounds=bounds) == []
    # a fixed parameter is not exchanged
    assert exchange_equivalent_parameters(obj, x, ["k1", "k2"], ssq=res.ssq, fixed_mask=[False, True]) == []


def test_no_false_ambiguity_for_distinguishable_steps():
    t = np.linspace(0.0, 600.0, 60)
    y0 = {"A": 1e-3, "B": 2e-3, "C": 0.0, "D": 0.0}
    model, _C, D0, noise = _case(BIMOLECULAR, {"k1": 30.0, "k2": 0.01}, y0, t)
    obj = _objective(model, t, y0, D0 + noise, ["k1", "k2"])
    assert exchange_equivalent_parameters(obj, np.array([30.0, 0.01]), ["k1", "k2"]) == []


def test_random_starts_stay_inside_bounds_and_scale_with_the_parameter():
    rng = np.random.default_rng(1)
    x0 = np.array([0.03, 0.006])
    # GUI default table: min = 0, no max. The old additive N(0, 0.2) gave negative constants.
    open_bounds = (np.zeros(2), np.full(2, np.inf))
    draws = np.array([randomize_start(x0, open_bounds, rng) for _ in range(300)])
    assert np.all(draws > 0.0)
    assert np.median(draws[:, 1]) == pytest.approx(0.006, rel=0.5)  # centred on the start, in decades
    assert np.std(np.log10(draws[:, 0])) == pytest.approx(1.0, rel=0.2)
    # wide positive bounds: log-uniform (a linear draw on [1e-5, 1e3] is > 1 in 99.9 % of cases)
    wide = (np.full(2, 1e-5), np.full(2, 1e3))
    draws = np.array([randomize_start(x0, wide, rng) for _ in range(300)])
    assert np.all((draws >= 1e-5) & (draws <= 1e3))
    assert 0.3 < np.mean(draws < 1e-1) < 0.7
    # fixed parameters never move; log10-fitted ones move in decades and respect bounds
    out = randomize_start(np.array([-2.0, 0.5]), (np.array([-6.0, 0.0]), np.array([2.0, 1.0])), rng,
                          fixed_mask=[False, True], log_mask=[True, False])
    assert out[1] == 0.5 and -6.0 <= out[0] <= 2.0
    # reproducible
    a = randomize_start(x0, open_bounds, np.random.default_rng(7))
    b = randomize_start(x0, open_bounds, np.random.default_rng(7))
    assert np.array_equal(a, b)
