"""Probe golden soft-bound inner convergence and residual smoothness without changing the estimator."""
import json
import tempfile
from pathlib import Path

import numpy as np
from hmfit_core import run_spectroscopy
from hmfit_core.processors import spectroscopy_processor as sp
from validation.phase31_comparison import golden
from validation.phase36_diagnosis import _quiet


def run():
    old = json.loads(Path('validation/results/phase31_comparison.json').read_text())['cases']
    result = {}
    with tempfile.TemporaryDirectory() as td:
        for seed in (1, 2, 3):
            path = Path(td) / 'gold.xlsx'
            golden._build_dataset(path, seed=seed, tail_noise=.0015 if seed == 3 else 0.)
            cfg = golden._base_config(path)
            cfg.update(render_graphs=False, baseline_mode='range', baseline_start=450., baseline_end=600.,
                       baseline_auto_quantile=.2, weighting_mode='std', weighting_power=1.,
                       weighting_normalize=True, eps_solver_mode='soft_bound', mu=.01,
                       delta_mode='relative', delta_rel=.005, alpha_smooth=.02)
            ed = _quiet(run_spectroscopy, cfg)['export_data']
            Y = np.asarray(ed['Y'], float).T
            weights = np.asarray(ed['weights'], float)[None, :]
            L = sp._build_smoothness_laplacian(Y.shape[1])
            solver = sp._build_equilibrium_solver('Newton-Raphson', ed['C_T'], ed['modelo'], [], 'Free')
            group = np.asarray(ed['abs_group_map'], float)
            theta = np.asarray(old[f'golden/soft-bound-{seed}']['explicit_lm']['log_beta'], float)
            hscale = max(abs(theta[0]), 1.)
            def evaluate(t, iterations=300):
                C = np.asarray(solver.concentraciones([t])[0], float)
                fit = sp._solve_spectral_model(C, Y, group_map=group, eps_solver_mode='soft_bound',
                    mu=.01, delta_mode='relative', delta_rel=.005, alpha_smooth=.02,
                    smooth_matrix=L, weights_row=weights, max_iters=iterations)
                r = sp._spectral_profile_residual(fit, Y, weights, .01, .02, L, 'legacy')
                A = fit['E_group']; lower = np.asarray(fit['lower_bound'])[:, None]
                grad = C.T @ (C @ A - Y) + .01 * np.minimum(A, 0.) + .02 * (A @ L)
                projected = np.where((A <= lower + 1e-10) & (grad > 0), 0., grad)
                return r, A, float(np.linalg.norm(projected)), float(np.max(np.abs(projected)))
            r0, A0, _, _ = evaluate(theta[0])
            progression = []
            for relative in np.logspace(-8, -2, 7):
                h = relative * hscale
                plus, _, _, _ = evaluate(theta[0] + h)
                minus, _, _, _ = evaluate(theta[0] - h)
                forward = (plus - r0)/h; backward = (r0 - minus)/h
                progression.append({'relative_step':float(relative), 'norm_r_plus':float(np.linalg.norm(plus)),
                    'norm_r_minus':float(np.linalg.norm(minus)),
                    'forward_backward_relative':float(np.linalg.norm(forward-backward)/max(np.linalg.norm((forward+backward)/2),1e-300))})
            jac = []
            for relative in (1e-8, 1e-5, 1e-2):
                h = relative * hscale
                plus = evaluate(theta[0]+h)[0]; minus = evaluate(theta[0]-h)[0]
                jac.append((plus-minus)/(2*h))
            convergence = []
            for iters in (300, 600, 3000):
                r, A, gn, gi = evaluate(theta[0], iters)
                convergence.append({'iterations':iters, 'projected_gradient_norm':gn,
                                    'projected_gradient_max':gi, 'delta_A_vs_300':float(np.linalg.norm(A-A0)),
                                    'outer_ssq':float(r@r*Y.size)})
            result[str(seed)] = {'theta':theta.tolist(), 'progression':progression, 'inner':convergence,
                'jacobian_relative_norms':[float(np.linalg.norm(jac[i]-jac[1])/max(np.linalg.norm(jac[1]),1e-300)) for i in range(3)]}
    return result


if __name__ == '__main__':
    out = run()
    Path('validation/results/phase36_golden_smoothness.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
