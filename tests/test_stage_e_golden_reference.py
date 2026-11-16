"""Independent bounded convex reference for the stage-E golden systems."""
import pytest

from validation.stage_e_golden_reference import diagnose


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_existing_newton_matches_independent_golden_reference(seed):
    result = diagnose(seed)
    scipy = result['scipy']
    newton = result['newton']
    assert scipy['success'], scipy['message']
    assert scipy['max_bound_violation'] < 1e-10
    assert newton['max_bound_violation'] < 1e-10
    # L-BFGS-B exits by relative objective reduction, so independently check KKT.
    assert scipy['projected_gradient_max'] < 1e-6
    assert newton['projected_gradient_max'] < 1e-9
    assert abs(scipy['inner_objective'] - newton['inner_objective']) < 1e-7
    assert result['prediction_difference_norm'] < 2e-4
