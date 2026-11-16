import numpy as np
import pytest
from scipy.optimize import nnls

from hmfit_core.processors import spectroscopy_processor as sp


def test_unconstrained_least_squares_is_verified_and_perturbation_is_not():
    C = np.array([[1., 0.], [0., 1.], [1., 1.]])
    Y = np.array([[1.], [2.], [3.]])
    A = np.linalg.lstsq(C, Y, rcond=None)[0]
    kw = dict(eps_solver_mode='soft_penalty', weight_regularization_mode='legacy', mu=0.)
    result = sp.spectral_inner_optimality(C, Y, A, **kw)
    assert result['status'] == 'verified'
    assert result['projected_gradient_normalized'] < 1e-8
    bad = sp.spectral_inner_optimality(C, Y, A+.1, **kw)
    assert bad['status'] == 'not_converged'


@pytest.mark.parametrize('mode', ['legacy', 'absolute'])
@pytest.mark.parametrize('weights', [np.array([2., .5, 3.]),
                                     np.array([[1., 2., 3.], [2., 1., 1.], [1., 3., 2.]])])
def test_hard_nnls_active_constraints_and_weight_semantics(mode, weights):
    C = np.array([[1., 0.], [0., 1.], [1., 1.]])
    Y = np.array([[-1., 1., -2.], [2., -1., 2.], [1., 0., 0.]])
    W, varying = sp._resolve_weights_matrix(weights, *Y.shape)
    # The hard solver ignores wavelength-only weights, in both modes.
    A = np.column_stack([nnls(C * (W[:, j, None] if varying else 1.),
                              Y[:, j] * (W[:, j] if varying else 1.))[0]
                         for j in range(Y.shape[1])])
    result = sp.spectral_inner_optimality(C, Y, A, eps_solver_mode='nnls_hard',
                                          weight_regularization_mode=mode, weights_row=weights)
    assert np.any(A == 0.)
    assert result['status'] == 'verified'
    assert result['complementarity_normalized'] < 1e-8
    A[0, 0] = -0.1
    assert sp.spectral_inner_optimality(C, Y, A, eps_solver_mode='nnls_hard',
                                        weight_regularization_mode=mode, weights_row=weights)['status'] == 'not_converged'


@pytest.mark.parametrize('mode', ['legacy', 'absolute'])
@pytest.mark.parametrize('weights', [np.array([2., .5, 3.]),
                                     np.array([[1., 2., 3.], [2., 1., 1.], [1., 3., 2.]])])
def test_soft_penalty_solution_with_effective_weights(mode, weights):
    C = np.array([[1., 0.], [0., 1.], [1., 1.]])
    Y = np.array([[-1., 1., -2.], [2., -1., 2.], [1., 0., 0.]])
    A, lb = sp._solve_absorptivities(C, Y, eps_solver_mode='soft_penalty', mu=.2,
                                     weights_row=weights, weight_regularization_mode=mode)
    result = sp.spectral_inner_optimality(C, Y, A, eps_solver_mode='soft_penalty', mu=.2,
                                          weights_row=weights, weight_regularization_mode=mode, lower_bound=lb)
    assert result['status'] == 'verified'


def test_smooth_and_bound_solution_against_independent_reference():
    from scipy.optimize import minimize
    C = np.array([[1., 0.], [0., 1.], [1., 1.]])
    Y = np.array([[-1., 1., -2.], [2., -1., 2.], [1., 0., 0.]])
    W = np.array([.5, 1., 2.])[None, :]
    L = sp._build_smoothness_laplacian(3)
    def fun(x):
        A = x.reshape(2, 3)
        R = (C@A-Y)*W
        return .5*(np.sum(R**2)+.2*np.sum(np.minimum(A, 0.)**2)+.3*np.sum((A@L)*A))
    ref = minimize(fun, np.zeros(6), bounds=[(-.5, None)]*6, method='L-BFGS-B',
                   options={'ftol':1e-15, 'gtol':1e-10, 'maxiter':10000})
    assert ref.success
    result = sp.spectral_inner_optimality(C, Y, ref.x.reshape(2, 3), eps_solver_mode='soft_bound',
                weight_regularization_mode='absolute', mu=.2, alpha_smooth=.3,
                smooth_matrix=L, lower_bound=np.array([-.5,-.5]), weights_row=W)
    assert result['status'] == 'verified', result


def test_insufficient_or_ill_conditioned_design_never_verified():
    C = np.array([[1., 1.], [2., 2.], [3., 3.]])
    Y = np.array([[1.], [2.], [3.]])
    kw = dict(eps_solver_mode='soft_penalty', weight_regularization_mode='legacy', mu=0.)
    result = sp.spectral_inner_optimality(C, Y, np.array([[.5], [.5]]), **kw)
    assert result['status'] == 'not_verified'
    assert 'rank-deficient' in result['reason']
    assert sp.spectral_inner_optimality(C, Y, np.zeros((2,1)), weights_row=np.zeros_like(Y),
                                         eps_solver_mode='soft_penalty', weight_regularization_mode='absolute')['status'] == 'not_verified'


def test_absolute_hard_channel_weights_match_the_unweighted_interior():
    C = np.eye(2)
    Y = np.array([[1., 2.], [3., 4.]])
    A = Y + np.array([[.5, .1], [.5, .1]])
    kw = dict(eps_solver_mode='nnls_hard', weight_regularization_mode='absolute')
    plain = sp.spectral_inner_optimality(C, Y, A, **kw)
    weighted = sp.spectral_inner_optimality(C, Y, A, weights_row=np.array([.01, 100.]), **kw)
    assert weighted['projected_gradient_normalized'] == pytest.approx(plain['projected_gradient_normalized'])


@pytest.mark.parametrize('seed,expected', [(1, 2.6887069411952402), (2, 2.67166713375592),
                                            (3, 2.6896840102331567)])
def test_golden_reports_inner_nonconvergence_without_changing_fit(tmp_path, seed, expected):
    from hmfit_core import run_spectroscopy
    from validation.phase31_comparison import golden
    path = tmp_path / 'golden.xlsx'
    golden._build_dataset(path, seed=seed, tail_noise=.0015 if seed == 3 else 0.)
    cfg = golden._base_config(path)
    cfg.update(render_graphs=False, baseline_mode='range', baseline_start=450., baseline_end=600.,
               weighting_mode='std', weighting_normalize=True, eps_solver_mode='soft_bound',
               delta_mode='relative', delta_rel=.005, weight_regularization_mode='legacy', mu=.01, alpha_smooth=.02)
    out = run_spectroscopy(cfg)
    assert out['success']
    assert out['export_data']['k'][0] == pytest.approx(expected, abs=1e-5)
    assert out['inner_optimality']['status'] == 'not_converged'
    assert out['export_data']['inner_optimality'] == out['inner_optimality']
    assert out['inner_optimality']['projected_gradient_normalized'] > .1
    assert any('inner' in warning.lower() for warning in out['warnings'])
    assert out['uncertainty_status'] == 'not_validated_for_constrained_smoothed_estimator'


def test_excel_exports_inner_and_uncertainty_status(tmp_path):
    from openpyxl import load_workbook
    from hmfit_core.exports import write_results_xlsx
    path = tmp_path / 'result.xlsx'
    diagnostic = {'status': 'not_converged', 'projected_gradient_normalized': .5,
                  'reason': 'first-order conditions not satisfied'}
    write_results_xlsx(path, statistics={'RMS': .1}, export_data={'inner_optimality': diagnostic,
                                         'uncertainty_status': 'not_validated_for_constrained_smoothed_estimator'})
    workbook = load_workbook(path, read_only=True)
    assert 'Inner_Optimality' in workbook.sheetnames
    values = list(workbook['Inner_Optimality'].values)
    assert any('not_converged' in row for row in values)
    assert any('not_validated_for_constrained_smoothed_estimator' in row for row in values)


def test_small_channel_cannot_be_hidden_by_large_channel():
    result = sp.spectral_inner_optimality(np.array([[1.]]), np.array([[1., 1e9]]),
                    np.array([[2., 1e9]]), eps_solver_mode='soft_penalty',
                    weight_regularization_mode='absolute')
    assert result['status'] == 'not_converged'


def test_small_species_cannot_be_hidden_by_large_species():
    result = sp.spectral_inner_optimality(np.eye(2), np.array([[1.], [1e9]]),
                    np.array([[2.], [1e9]]), eps_solver_mode='soft_penalty',
                    weight_regularization_mode='absolute')
    assert result['status'] == 'not_converged'


@pytest.mark.parametrize('mode', ['legacy', 'absolute'])
def test_coupled_soft_solution_uses_actual_inner_weights(mode):
    from hmfit_core.utils.nnls_utils import _solve_weighted_penalty_newton
    C = np.array([[1., 0.], [0., 1.], [1., 1.]])
    Y = np.array([[-1., 1., -2.], [2., -1., 2.], [1., 0., 0.]])
    W = np.broadcast_to(np.array([.5, 1., 2.]), Y.shape)
    effective = np.ones_like(W) if mode == 'legacy' else W
    L = sp._build_smoothness_laplacian(3)
    A = _solve_weighted_penalty_newton(C, Y, effective, 0., .2, None, .3, L, 300)
    result = sp.spectral_inner_optimality(C, Y, A, eps_solver_mode='soft_penalty',
                weight_regularization_mode=mode, mu=.2, alpha_smooth=.3, smooth_matrix=L, weights_row=W)
    assert result['status'] == 'verified', result


def test_diagnostic_failure_does_not_change_fit(tmp_path, monkeypatch):
    from hmfit_core import run_spectroscopy
    from validation.phase31_comparison import golden
    path = tmp_path / 'golden.xlsx'
    golden._build_dataset(path, seed=1)
    cfg = golden._base_config(path)
    cfg.update(render_graphs=False)
    original = run_spectroscopy(cfg)
    def fail(*args, **kwargs):
        raise RuntimeError('deliberate diagnostic failure')
    monkeypatch.setattr(sp, 'spectral_inner_optimality', fail)
    fallback = run_spectroscopy(cfg)
    assert fallback['success']
    assert fallback['inner_optimality']['status'] == 'not_verified'
    assert fallback['constants'] == original['constants']
    assert fallback['statistics'] == original['statistics']
    assert fallback['export_data']['A_solver'] == original['export_data']['A_solver']
