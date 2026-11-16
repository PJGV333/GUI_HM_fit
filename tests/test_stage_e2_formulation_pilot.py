"""Checks for the exploratory formulation pilot's objective accounting."""
import numpy as np
import pytest

from validation.stage_e2_formulation_pilot import _metrics


@pytest.mark.parametrize('mode', ['legacy', 'absolute'])
def test_pilot_metrics_use_fitted_coefficients_and_correct_outer_objective(mode):
    C = np.array([[1.], [2.]])
    A = np.array([[-1., .5, 2.]])
    Y = np.array([[.2, .4, .6], [.1, .3, .5]])
    W = np.array([1., .5, 2.])
    out = {'optimizer_result': {'success': True}, 'export_data': {
        'C': C.tolist(), 'abs_group_map': [[1.]], 'Y': Y.T.tolist(),
        'A_solver': A.T.tolist(), 'weights': W.tolist(), 'lower_bound': [-3.], 'k': [3.]}}
    result = _metrics(out, mode, .3, .2, np.zeros_like(A))
    r = C@A-Y
    L = np.array([[1.,-2.,1.],[-2.,4.,-2.],[1.,-2.,1.]])
    penalty = .3*np.sum(np.minimum(A,0.)**2) + .2*np.sum((A@L)*A)
    weighted_ss = np.sum((W*r)**2)
    expected_outer = np.sqrt((weighted_ss+(penalty if mode=='absolute' else 0))/Y.size)
    inner_data = weighted_ss if mode=='absolute' else np.sum(r**2)
    assert result['outer_objective_rms'] == pytest.approx(expected_outer)
    assert result['inner_objective'] == pytest.approx(.5*(inner_data+penalty))
    assert result['actual_residual_rms'] == pytest.approx(np.sqrt(np.mean(r**2)))
    assert result['coefficient_rmse_vs_raw_truth'] == pytest.approx(np.sqrt(np.mean(A**2)))
