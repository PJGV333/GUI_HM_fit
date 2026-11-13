import numpy as np
import pytest

from hmfit_core.utils.explicit_lm import explicit_lm, centered_residual_jacobian
from hmfit_core.utils.profile_jacobian import spectral_profile_jacobian
from hmfit_core.processors.spectroscopy_processor import (
    _solve_spectral_model, _spectral_profile_objective, _spectral_profile_residual,
    _build_smoothness_laplacian,
)


@pytest.mark.parametrize('mode', ['legacy', 'absolute'])
@pytest.mark.parametrize('solver,delta,alpha', [
    ('soft_penalty', 'off', 0.0), ('nnls_hard', 'off', 0.0),
    ('soft_bound', 'off', 0.0), ('soft_penalty', 'relative', 0.0),
    ('soft_penalty', 'off', 0.2),
])
def test_general_profile_residual_matches_existing_objective(mode, solver, delta, alpha):
    C = np.array([[1., .2], [.2, 1.], [.8, .5], [.4, .9]])
    Y = np.array([[.5, .2, .3], [.4, .1, .6], [.8, .2, .5], [.3, .8, .2]])
    W = np.array([[1., .7, 1.], [.8, .9, .8], [1.2, 1., .5], [.9, 1.1, 1.]])
    L = _build_smoothness_laplacian(3)
    group = np.array([[1.], [1.]])
    def residual(x):
        fit = _solve_spectral_model(C * (1 + x[0]), Y, weights_row=W, mu=.2,
                                    eps_solver_mode=solver, delta_mode=delta,
                                    alpha_smooth=alpha, smooth_matrix=L,
                                    group_map=group, weight_regularization_mode=mode)
        r = _spectral_profile_residual(fit, Y, W, .2, alpha, L, mode)
        expected = _spectral_profile_objective(fit, Y, W, .2, alpha, L, mode)
        np.testing.assert_allclose(r @ r, expected ** 2, rtol=1e-12, atol=1e-14)
        return r
    r, j = centered_residual_jacobian(residual, np.array([.1]))
    assert np.all(np.isfinite(r)) and np.all(np.isfinite(j))


def test_lm_linear_and_active_bound():
    def f(x):
        return np.array([x[0] - 2, 2 * (x[1] + 3)]), np.diag([1., 2.])
    result = explicit_lm(f, [0., 0.], [(None, 1.), (None, None)])
    assert result.success
    np.testing.assert_allclose(result.x, [1., -3.], atol=1e-7)
    assert result.active_bounds.tolist() == [True, False]
    assert result.scaled_gradient < 1e-8


@pytest.mark.parametrize('mode', ['legacy', 'absolute'])
@pytest.mark.parametrize('solver', ['soft_penalty', 'nnls_hard'])
@pytest.mark.parametrize('negative', [False, True])
def test_profile_jacobian_matches_actual_residual(mode, solver, negative):
    C = np.array([[1., .2], [.2, 1.], [.8, .5], [.4, .9]])
    dC = np.array([[[.2, -.1], [.1, .2], [-.1, .1], [.1, -.1]]])
    Y = C @ np.array([[1., .7], [-.6 if negative else .6, .4]]) + .01
    W = np.array([[1., .7], [.8, .9], [1.2, 1.], [.9, 1.1]])
    mu = .2
    def residual(t):
        fit = _solve_spectral_model(C + t * dC[0], Y, weights_row=W, mu=mu,
                                    eps_solver_mode=solver, weight_regularization_mode=mode)
        a = fit['E_group']
        r = ((fit['yfit'] - Y) * W).ravel()
        if mode == 'absolute' and solver == 'soft_penalty':
            r = np.r_[r, np.sqrt(mu) * np.minimum(a, 0).ravel()]
        obj = _spectral_profile_objective(fit, Y, W, mu, 0., None, mode)
        np.testing.assert_allclose(r @ r / Y.size, obj ** 2, atol=1e-12)
        return r, a
    _, a = residual(0.)
    if negative:
        assert a[1, 0] < 0 if solver == 'soft_penalty' else a[1, 0] == 0
    else:
        assert np.all(a > 0)
    analytic = spectral_profile_jacobian(C, dC, Y, a, W, solver_mode=solver,
                                        mu=mu, regularization_mode=mode)
    h = 1e-5
    fd = (residual(h)[0] - residual(-h)[0]) / (2 * h)
    np.testing.assert_allclose(analytic[:, 0], fd, atol=2e-7, rtol=2e-5)
