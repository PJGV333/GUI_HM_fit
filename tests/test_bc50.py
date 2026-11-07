"""BC50 / BC50^0 (hmfit_core.bc50) against the analytic expressions of the R09a SI
appendix (eqs A23-A30) and the limitation of note 18 (never a silent value)."""
import numpy as np
import pytest

from hmfit_core.bc50 import bc50_curve

X = np.array([0.0, 0.2, 0.5, 0.9])


def test_two_reagent_1_1_is_kd_over_1_minus_x():
    r = bc50_curve(["A", "B"], [[1, 1]], [3.0], "A", "B", x=X)
    np.testing.assert_allclose(r.bc50, 1e-3 / (1 - X), rtol=1e-9)
    assert r.warnings == []


def test_eq_A23_ABC_only():
    r = bc50_curve(["A", "B", "C"], [[1, 1, 1]], [4.0], "A", "B", "C", x=X, free_ancillary=0.1)
    np.testing.assert_allclose(r.bc50, 1.0 / ((1 - X) * 1e4 * 0.1), rtol=1e-9)


def test_eq_A24_AB_ABC():
    r = bc50_curve(["A", "B", "C"], [[1, 1, 0], [1, 1, 1]], [2.0, 3.0], "A", "B", "C", x=X)
    np.testing.assert_allclose(r.bc50, 1.0 / ((1 - X) * (1e2 + 1e3)), rtol=1e-9)


def test_eq_A25_AB_AC_BC_ABC():
    lb = [2.0, 1.0, 1.5, 3.0]
    b110, b101, b011, b111 = 10.0 ** np.array(lb)
    r = bc50_curve(["A", "B", "C"], [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], lb, "A", "B", "C", x=X)
    np.testing.assert_allclose(r.bc50, (1 + b011) * (1 + b101) / ((1 - X) * (b110 + b111)), rtol=1e-9)


def test_eq_A28_AB2C_is_indefinite_at_x0_and_warned():
    b121, C = 10.0 ** 6, 1.0
    xs = np.array([0.0, 0.1, 0.5, 0.8])
    r = bc50_curve(["A", "B", "C"], [[1, 2, 1]], [6.0], "A", "B", "C", x=xs)
    assert np.isnan(r.bc50[0])
    assert any("indefinite" in w for w in r.warnings) and any("note 18" in w for w in r.warnings)
    expected = b121 ** -0.5 / (2 * np.sqrt(xs[1:] * (1 - xs[1:])) * np.sqrt(C))
    np.testing.assert_allclose(r.bc50[1:], expected, rtol=1e-8)


def test_eq_A30_ABC_AB2C_depends_on_TB_and_is_warned():
    r = bc50_curve(["A", "B", "C"], [[1, 1, 1], [1, 2, 1]], [3.0, 5.0], "A", "B", "C", x=[0.0, 0.4])
    np.testing.assert_allclose(r.bc50[1], 1.0 / ((1 - 0.4) * (1e3 + 1e5 * r.totals["B"][1])), rtol=1e-9)
    assert np.isfinite(r.bc50[0])            # BC50^0 exists but ignores AB2C ...
    assert any("ignores them" in w for w in r.warnings)   # ... and says so


@pytest.mark.parametrize("cond", ["fixed_free", "electroneutral"])
def test_definition_holds_at_every_point(cond):
    """50 % of B bound, fraction x of A bound, and the condition on C."""
    xs = np.round(np.arange(0.02, 0.99, 0.08), 2)
    r = bc50_curve(["A", "B", "C"], [[1, 1, 0], [1, 1, 1], [0, 1, 1], [2, 1, 0], [2, 0, 0]],
                   [3.0, 6.0, 4.0, 4.5, 0.8], "A", "B", "C", x=xs, condition=cond, counterion="B",
                   species_names=["AB", "ABC", "BC", "A2B", "A2"])
    assert r.converged.all()
    S = np.vstack([np.eye(3), [[1, 1, 0], [1, 1, 1], [0, 1, 1], [2, 1, 0], [2, 0, 0]]])
    bound = (S[:, 0] > 0) & (S[:, 1] > 0)
    for i in range(xs.size):
        c = r.concentrations[i]
        assert abs((S[bound, 0] * c[bound]).sum() / r.totals["A"][i] - xs[i]) < 1e-9   # x_A
        assert abs((S[bound, 1] * c[bound]).sum() / r.totals["B"][i] - 0.5) < 1e-9     # 50 % of B
        assert r.bc50[i] == r.totals["A"][i]
    if cond == "electroneutral":
        np.testing.assert_allclose(r.totals["C"], r.totals["B"], rtol=1e-9)
    else:
        np.testing.assert_allclose(r.free["C"], 1.0)


def test_bad_arguments_are_rejected():
    with pytest.raises(ValueError):
        bc50_curve(["A", "B", "C"], [[1, 1, 1]], [3.0], "A", "B", "C", condition="electroneutral")
    with pytest.raises(ValueError):
        bc50_curve(["A", "B"], [[1, 1]], [3.0], "A", "B", x=[1.0])
