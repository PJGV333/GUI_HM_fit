"""Independent convex reference for the three historical soft-bound cases."""
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from hmfit_core import run_spectroscopy
from hmfit_core.processors import spectroscopy_processor as sp
from hmfit_core.utils.nnls_utils import _solve_weighted_penalty_newton
from validation.phase31_comparison import golden
from validation.phase36_diagnosis import _quiet

ROOT = Path(__file__).resolve().parents[1]


def diagnose(seed):
    old = json.loads((ROOT / 'validation/results/phase31_comparison.json').read_text())['cases']
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'gold.xlsx'
        golden._build_dataset(path, seed=seed, tail_noise=.0015 if seed == 3 else 0.)
        cfg = golden._base_config(path)
        cfg.update(render_graphs=False, baseline_mode='range', baseline_start=450., baseline_end=600.,
                   baseline_auto_quantile=.2, weighting_mode='std', weighting_power=1.,
                   weighting_normalize=True, eps_solver_mode='soft_bound', mu=.01,
                   delta_mode='relative', delta_rel=.005, alpha_smooth=.02)
        result = _quiet(run_spectroscopy, cfg)
        assert result.get('success'), result.get('error')
        ed = result['export_data']
        Y = np.asarray(ed['Y'], float).T
        W = np.asarray(ed['weights'], float)[None, :]
        L = sp._build_smoothness_laplacian(Y.shape[1])
        solver = sp._build_equilibrium_solver('Newton-Raphson', ed['C_T'], ed['modelo'], [], 'Free')
        theta = np.asarray(old[f'golden/soft-bound-{seed}']['explicit_lm']['log_beta'], float)
        C = np.asarray(solver.concentraciones(theta)[0]) @ np.asarray(ed['abs_group_map'], float)
        initial = np.linalg.pinv(C) @ Y
        lower = -.005 * np.max(np.abs(initial), axis=1)[:, None]
        bounds = np.broadcast_to(lower, initial.shape)
        scale = 1000.

        def f(x):
            A = x.reshape(initial.shape) * scale
            R = C @ A - Y
            neg = np.minimum(A, 0.)
            obj = .5 * (np.sum(R*R) + .01*np.sum(neg*neg) + .02*np.sum((A@L)*A))
            grad = C.T @ R + .01*neg + .02*(A@L)
            return float(obj), (grad*scale).ravel()

        start = time.perf_counter()
        independent = minimize(f, np.maximum(initial, bounds).ravel()/scale, jac=True, method='L-BFGS-B',
                               bounds=list(zip((bounds/scale).ravel(), [None]*bounds.size)),
                               options={'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': 30000, 'maxls': 50, 'maxcor': 40})
        t_scipy = time.perf_counter()-start
        reference = independent.x.reshape(initial.shape)*scale
        start = time.perf_counter()
        production = sp._solve_spectral_model(C, Y, eps_solver_mode='soft_bound', mu=.01,
                                              delta_mode='relative', delta_rel=.005,
                                              alpha_smooth=.02, smooth_matrix=L,
                                              weights_row=W, max_iters=300)
        t_production = time.perf_counter()-start
        start = time.perf_counter()
        candidate = np.asarray(_solve_weighted_penalty_newton(C, Y, np.ones_like(Y), 0., .01,
                                                             lower[:, 0], .02, L, 300))
        t_candidate = time.perf_counter()-start
        def metrics(A):
            obj, g = f(A.ravel()/scale)
            grad = g.reshape(A.shape)/scale
            pg = np.where((A <= bounds + 1e-9) & (grad > 0), 0., grad)
            data_grad_scale = np.max(np.abs(C.T @ Y), initial=0.)
            model_grad_scale = np.max(np.abs(C.T @ (C @ A)), initial=0.)
            penalty_scale = max(np.max(np.abs(.01*np.minimum(A, 0.)), initial=0.),
                                np.max(np.abs(.02*(A@L)), initial=0.))
            scale_grad = max(data_grad_scale, model_grad_scale, penalty_scale, np.finfo(float).tiny)
            multiplier = np.where(A <= bounds + 1e-9, np.maximum(grad, 0.), 0.)
            return {'inner_objective': obj, 'projected_gradient_norm': float(np.linalg.norm(pg)),
                    'projected_gradient_normalized': float(np.max(np.abs(pg))/scale_grad),
                    'complementarity_max': float(np.max(np.abs(multiplier*(A-bounds)))),
                    'projected_gradient_max': float(np.max(np.abs(pg))),
                    'outer_ssq': float(np.sum((W*(C@A-Y))**2)),
                    'max_bound_violation': float(np.max(np.maximum(bounds-A, 0.)))}
        c_singular = np.linalg.svd(C, compute_uv=False)
        lipschitz = float(c_singular[0]**2 + .01 + .02*np.linalg.norm(L, 2))
        low_curvature = float(c_singular[-1]**2)
        return {'theta':theta.tolist(), 'shape':list(Y.shape), 'condition_C':float(np.linalg.cond(C)),
                'pgd_lipschitz_bound': lipschitz, 'data_min_curvature': low_curvature,
                'curvature_ratio_bound': lipschitz/low_curvature,
                'scipy':{**metrics(reference), 'success':bool(independent.success),
                         'message':str(independent.message), 'iterations':int(independent.nit), 'seconds':t_scipy},
                'production': {**metrics(production['E_group']), 'seconds':t_production},
                'newton':{**metrics(candidate), 'seconds':t_candidate},
                'coefficient_difference_norm':float(np.linalg.norm(candidate-reference)),
                'prediction_difference_norm':float(np.linalg.norm(C@(candidate-reference)))}


if __name__ == '__main__':
    print(json.dumps({str(i):diagnose(i) for i in (1,2,3)}, indent=2, allow_nan=False))
