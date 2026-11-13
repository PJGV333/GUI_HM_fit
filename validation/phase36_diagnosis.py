"""Cross-start diagnostics at the pre-correction phase 3.1 solutions.

For NMR both optimizers are run with a fixed shared weight vector (the one
used by Powell in the last IRLS cycle). This isolates the inner optimizer from
IRLS weight updates. Run before modifying the LM algorithm.
"""
import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from hmfit_core import run_spectroscopy
from hmfit_core.api import run_nmr_fit
from hmfit_core.processors import nmr_processor as nmr
from hmfit_core.processors import spectroscopy_processor as sp
from hmfit_core.utils.explicit_lm import centered_residual_jacobian, explicit_lm, _gradient
from validation import literature_benchmark as lit
from validation.phase31_comparison import pc, golden, _nmr_phase1


def _quiet(f, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **kw)


def _diagnose(residual, starts, bounds, actual_jacobian=None):
    lower = np.asarray([b[0] if b[0] is not None else -np.inf for b in bounds])
    upper = np.asarray([b[1] if b[1] is not None else np.inf for b in bounds])
    def grad(x):
        r, j = centered_residual_jacobian(residual, x)
        return _gradient(r, j, np.asarray(x), lower, upper)[2]
    def record(x, success, reason, nit):
        r = residual(x)
        return {'x': np.asarray(x).tolist(), 'ssq': float(r @ r),
                'scaled_gradient': grad(x), 'success': bool(success),
                'reason': str(reason), 'iterations': int(nit)}
    out = {}
    for label, x0 in starts.items():
        x0 = np.asarray(x0, float)
        out[label + '_initial'] = record(x0, True, 'initial', 0)
        lm = explicit_lm(lambda x: centered_residual_jacobian(residual, x), x0, bounds)
        out['lm_from_' + label] = record(lm.x, lm.success, lm.reason, lm.nit)
        p = minimize(lambda x: float(np.dot(residual(x), residual(x))), x0,
                     method='powell', bounds=bounds)
        out['powell_from_' + label] = record(p.x, p.success, p.message, p.nit)
    # Independent centered differences at a different step reveal local numerical
    # nonsmoothness, including mode-specific active-set/inner-solver changes.
    x = np.asarray(starts['lm'], float)
    _, j1 = centered_residual_jacobian(residual, x)
    cols = []
    for i in range(x.size):
        h = 1e-5 * max(abs(x[i]), 1.)
        d = np.zeros_like(x); d[i] = h
        cols.append((residual(x + d) - residual(x - d)) / (2*h))
    j2 = np.column_stack(cols)
    actual, path = (actual_jacobian(x) if actual_jacobian else (j1, 'centered'))
    out['jacobian_check'] = {'method': path,
                             'max_abs': float(np.max(np.abs(actual-j2), initial=0)),
                             'relative_norm': float(np.linalg.norm(actual-j2) / max(np.linalg.norm(j2), 1e-300)),
                             'centered_step_relative_norm': float(np.linalg.norm(j1-j2) / max(np.linalg.norm(j2), 1e-300)),
                             'note': 'actual LM Jacobian versus centered differences at 1e-5 relative step'}
    return out


def _spectro(cfg, old):
    cfg = dict(cfg, render_graphs=False, optimizer='powell')
    out = _quiet(run_spectroscopy, cfg)
    ed = out['export_data']
    Y = np.asarray(ed['Y'], float).T
    weights = np.asarray(ed['weights'], float)
    if weights.ndim == 1:
        weights = weights[None, :]
    smooth = sp._build_smoothness_laplacian(Y.shape[1]) if cfg.get('alpha_smooth', 0) else None
    solver = sp._build_equilibrium_solver('Newton-Raphson', ed['C_T'], ed['modelo'],
                                         cfg.get('non_abs_species', []), cfg.get('model_settings', 'Free'))
    groups = np.asarray(ed['abs_group_map'], float)
    def residual(x):
        C = solver.concentraciones(np.asarray(x))[0]
        fit = sp._solve_spectral_model(C, Y, group_map=groups,
            eps_solver_mode=cfg.get('eps_solver_mode', 'soft_penalty'), mu=cfg.get('mu', .01),
            delta_mode=cfg.get('delta_mode', 'off'), delta_rel=cfg.get('delta_rel', .01),
            alpha_smooth=cfg.get('alpha_smooth', 0.), smooth_matrix=smooth, weights_row=weights,
            weight_regularization_mode=cfg.get('weight_regularization_mode', 'legacy'))
        return sp._spectral_profile_residual(fit, Y, weights, cfg.get('mu', .01),
            cfg.get('alpha_smooth', 0.), smooth, cfg.get('weight_regularization_mode', 'legacy')) * np.sqrt(Y.size)
    def actual_jacobian(x):
        from hmfit_core.utils.profile_jacobian import spectral_profile_jacobian
        C = np.asarray(solver.concentraciones(x)[0], float)
        fit = sp._solve_spectral_model(C, Y, group_map=groups,
            eps_solver_mode=cfg.get('eps_solver_mode', 'soft_penalty'), mu=cfg.get('mu', .01),
            delta_mode=cfg.get('delta_mode', 'off'), delta_rel=cfg.get('delta_rel', .01),
            alpha_smooth=cfg.get('alpha_smooth', 0.), smooth_matrix=smooth, weights_row=weights)
        if cfg.get('alpha_smooth', 0.) or cfg.get('delta_mode', 'off') != 'off' or np.any(np.abs(fit['E_group']) < 1e-12):
            return centered_residual_jacobian(residual, x)[1], 'centered'
        _, dc = centered_residual_jacobian(lambda theta: np.asarray(solver.concentraciones(theta)[0]).ravel(), x)
        dC = dc.T.reshape(len(x), *C.shape)
        jac = spectral_profile_jacobian(C, dC, Y, fit['E_group'], weights,
                                        solver_mode=cfg.get('eps_solver_mode','soft_penalty'), mu=cfg.get('mu',.01))
        return jac, 'analytic'
    return _diagnose(residual, {m: old[m]['log_beta'] for m in ('powell', 'lm')}, cfg['bounds'], actual_jacobian)


def _nmr(cfg, old):
    out = _quiet(run_nmr_fit, dict(cfg, optimizer='powell'), progress_cb=lambda _: None)
    ed = out['export_data']
    dq = np.asarray(ed['Chemical_Shifts'], float)
    ct = np.asarray(ed['C_T'], float)
    model = np.asarray(ed['modelo'], float).T
    nas = ed['non_absorbent_species']
    solver = nmr._build_nmr_solver('Newton-Raphson', ct, model, nas, cfg.get('model_settings', 'Free'))
    D, _ = nmr.build_D_cols(ct, ed['column_names'], ed['signal_names'], default_idx=0)
    mask = np.isfinite(dq) & np.isfinite(D)
    stoich = np.asarray(ed['stoichiometry_map'], float)
    if not stoich.size: stoich = None
    w = np.asarray(ed['weights_used_for_fit'], float)
    fixed = np.asarray(ed['fixed_mask'], bool)
    free = np.flatnonzero(~fixed)
    p0 = np.asarray(ed['k'], float)
    def residual(x):
        k = p0.copy(); k[free] = x
        C = solver.concentraciones(k)[0]
        pred = nmr.project_coeffs_block_onp_frac(dq, C, D, mask, stoichiometry=stoich,
                                                  rcond=1e-10, ridge=1e-8)
        return (dq - pred)[mask] * np.sqrt(np.broadcast_to(w[None, :], dq.shape)[mask])
    bounds = [(None, None)] * len(free)
    raw = cfg.get('bounds', [])
    for j, i in enumerate(free):
        if i < len(raw):
            b = raw[i]
            bounds[j] = (b.get('min'), b.get('max')) if isinstance(b, dict) else tuple(b)
    starts = {m: np.asarray(old[m]['log_beta'])[free] for m in ('powell', 'lm')}
    result = _diagnose(residual, starts, bounds)
    result['common_weights'] = w.tolist()
    result['weights_source'] = 'Powell last IRLS cycle'
    result['free_indices'] = free.tolist()
    return result


def main():
    old = json.loads(Path('validation/results/phase31_comparison.json').read_text())['cases']
    nmr.generate_figure_base64 = lambda *a, **k: ''
    nmr.generate_figure2_base64 = lambda *a, **k: ''
    results = {}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for key, pair in old.items():
            if 'unsupported' in pair: continue
            a, b = pair['powell'], pair['explicit_lm']
            if b['ssq'] <= a['ssq'] * (1 + 1e-8): continue
            starts = {'powell': a, 'lm': b}
            if key.startswith('literature/'):
                case = lit.load_case(key.split('/')[1]); conc, shifts = lit.numeric_table(case)
                cfg = lit.gui_config(case, str(root/'lit.xlsx'), conc, shifts)
                result = _nmr(cfg, starts)
            elif key == 'phase1/nmr-11':
                path = root/'nmr.xlsx'; model = [[1,0],[0,1],[1,1]]
                _nmr_phase1(path, model, [3.])
                cfg = {'file_path': str(path), 'nmr_sheet':'S', 'conc_sheet':'S',
                       'column_names':['H','G'], 'signal_names':['s1','s2'], 'receptor_label':'H',
                       'guest_label':'G', 'modelo':model, 'non_abs_species':[1],
                       'algorithm':'Newton-Raphson', 'model_settings':'Free', 'initial_k':[2.5],
                       'bounds':[{'min':0., 'max':8.}]}
                result = _nmr(cfg, starts)
            elif key == 'phase1/spectro-noiseless':
                path = root/'spec.xlsx'; model = [[1,0],[0,1],[1,1]]
                pc._spectro_dataset(str(path), [3.], model, noise=0.)
                result = _spectro(pc._spectro_cfg(path, model, [2.5], [(0.,6.)]), starts)
            elif key.startswith('golden/'):
                seed = int(key[-1]); path = root/'gold.xlsx'
                golden._build_dataset(path, seed=seed, tail_noise=.0015 if seed == 3 else 0.)
                cfg = golden._base_config(path)
                cfg.update(baseline_mode='range', baseline_start=450., baseline_end=600.,
                           baseline_auto_quantile=.2, weighting_mode='std', weighting_power=1.,
                           weighting_normalize=True, eps_solver_mode='soft_bound', mu=.01,
                           delta_mode='relative', delta_rel=.005, alpha_smooth=.02)
                result = _spectro(cfg, starts)
            else: raise ValueError(key)
            results[key] = result
            print('completed', key, flush=True)
    Path('validation/results/phase36_diagnosis_before.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': main()
