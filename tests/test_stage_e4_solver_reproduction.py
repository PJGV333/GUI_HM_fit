"""Captured pre-change systems and their independent interior references."""
import json
from pathlib import Path

import numpy as np
import pytest

from validation.stage_e3_identifiability import bounded_reference
from validation.stage_e_golden_reference import refine_reference

PATH = Path(__file__).resolve().parents[1] / 'validation/results/stage_e4_failure_trace.json'


def systems():
    return json.loads(PATH.read_text())['records']


@pytest.mark.parametrize('index', [0, 1])
def test_historical_failure_system_has_independent_feasible_reference(index):
    row = systems()[index]
    s = row['system']
    C, Y, lower = (np.asarray(s[x], float) for x in ('C', 'Y', 'lower'))
    if s['mu'] == 0.:
        A, residual, pg = bounded_reference(C, Y, lower, s['alpha'])
        objective = .5 * float(residual @ residual)
    else:
        L = np.asarray(s['L'], float)
        A = refine_reference(C, Y, L, lower[:,None], np.linalg.pinv(C) @ Y,
                             mu=s['mu'], alpha=s['alpha'])
        grad = C.T@(C@A-Y) + s['mu']*np.minimum(A,0.) + s['alpha']*(A@L)
        pg = np.max(np.abs(np.where((A <= lower[:,None]+1e-9) & (grad > 0), 0., grad)))
        objective = .5*(np.sum((C@A-Y)**2) + s['mu']*np.sum(np.minimum(A,0.)**2)
                        + s['alpha']*np.sum((A@L)*A))
    assert np.min(A-lower[:,None]) >= -1e-9
    assert pg < 1e-9
    assert np.isfinite(objective)
    assert 'weighted Newton' in row['error']


@pytest.mark.parametrize('index', [0, 1])
def test_experimental_newton_never_hides_failure(index):
    from validation.stage_e4_newton_candidate import _solve_weighted_penalty_newton_verified
    from hmfit_core.processors.spectroscopy_processor import spectral_inner_optimality
    from hmfit_core.utils.nnls_utils import WeightedLinearSolveError
    s = systems()[index]['system']
    C, Y, W, lower = (np.asarray(s[x], float) for x in ('C', 'Y', 'W', 'lower'))
    L = np.asarray(s['L'], float) if s['L'] is not None else None
    try:
        A, info = _solve_weighted_penalty_newton_verified(C, Y, W, s['ridge'], s['mu'],
                                       lower, s['alpha'], L, s['max_iters'], return_info=True)
    except WeightedLinearSolveError as exc:
        assert index == 1, 'the unregularized captured case must converge in this pilot'
        assert 'line search failed' in str(exc)
        assert getattr(exc, 'diagnostics')['status'] != 'verified'
        return
    report = spectral_inner_optimality(C, Y, A, eps_solver_mode='soft_bound',
        weight_regularization_mode='absolute', mu=s['mu'], alpha_smooth=s['alpha'],
        smooth_matrix=L, lower_bound=lower, weights_row=W)
    assert report['status'] == 'verified', report
    assert info['iterations'] > 0 and info['stop_reason'] == 'kkt'
    assert info['rejected_steps'] >= 0
    if s['mu'] == 0:
        ref, residual, _ = bounded_reference(C, Y, lower, s['alpha'])
        reference_objective = .5 * float(residual @ residual)
    else:
        ref = refine_reference(C, Y, L, lower[:,None], np.linalg.pinv(C)@Y,
                               mu=s['mu'], alpha=s['alpha'])
        D = np.diff(np.eye(Y.shape[1]), n=2, axis=0)
        reference_objective = .5*(np.sum((C@ref-Y)**2) + s['mu']*np.sum(np.minimum(ref,0.)**2)
                                  + s['alpha']*np.sum((ref@D.T)**2))
    D = np.diff(np.eye(Y.shape[1]), n=2, axis=0)
    objective = .5*np.sum((C@A-Y)**2) + .5*s['mu']*np.sum(np.minimum(A,0.)**2)
    if s['alpha']:
        objective += .5*s['alpha']*np.sum((A@D.T)**2)
    assert objective == pytest.approx(reference_objective, rel=1e-8, abs=1e-11)
    np.testing.assert_allclose(C@A, C@ref, rtol=1e-5, atol=1e-7)


def test_smoothing_difference_squares_and_gradient_match_quadratic():
    rng = np.random.default_rng(74)
    A = rng.normal(size=(4, 31))
    direction = rng.normal(size=A.shape)
    D = np.diff(np.eye(A.shape[1]), n=2, axis=0)
    L = D.T@D
    np.testing.assert_allclose(np.sum((A@L)*A), np.sum((A@D.T)**2), rtol=1e-14)
    np.testing.assert_allclose(A@L, (A@D.T)@D, atol=1e-14)
    step=1e-5
    loss=lambda X: .5*np.sum((X@D.T)**2)
    finite=(loss(A+step*direction)-loss(A-step*direction))/(2*step)
    assert finite == pytest.approx(np.sum(((A@D.T)@D)*direction), rel=1e-8, abs=1e-8)


def test_decimal_gradient_reference_matches_exact_small_system():
    from validation.stage_e4_newton_candidate_probe import high_precision_gradient
    C=np.eye(2); Y=np.array([[1., 2., 3.], [2., 3., 4.]])
    A=np.array([[2., -1., 1.], [1., 2., 3.]])
    D=np.diff(np.eye(3),n=2,axis=0); L=D.T@D
    expected=(A-Y)+.25*np.minimum(A,0.)+.5*(A@L)
    np.testing.assert_allclose(high_precision_gradient(C,Y,A,L,.25,.5),expected,rtol=0,atol=0)


def test_experimental_newton_exhaustion_is_explicit_and_no_iteration_inflation_needed():
    from validation.stage_e4_newton_candidate import _solve_weighted_penalty_newton_verified
    from hmfit_core.utils.nnls_utils import WeightedLinearSolveError
    s=systems()[0]['system']
    C,Y,W,lb=(np.asarray(s[key],float) for key in ('C','Y','W','lower'))
    args=(C,Y,W,s['ridge'],s['mu'],lb,s['alpha'],None)
    with pytest.raises(WeightedLinearSolveError,match='did not converge'):
        _solve_weighted_penalty_newton_verified(*args,5)
    A,info=_solve_weighted_penalty_newton_verified(*args,60,return_info=True)
    B,info2=_solve_weighted_penalty_newton_verified(*args,300,return_info=True)
    np.testing.assert_array_equal(A,B)
    assert info['iterations']==info2['iterations'] < 60


def test_experimental_newton_rejects_unverifiable_data():
    from validation.stage_e4_newton_candidate import _solve_weighted_penalty_newton_verified
    from hmfit_core.utils.nnls_utils import WeightedLinearSolveError
    with pytest.raises(WeightedLinearSolveError,match='unverifiable'):
        _solve_weighted_penalty_newton_verified(np.eye(2),np.ones((2,1)),np.zeros((2,1)),
                                               0.,0.,None,0.,None,60)
