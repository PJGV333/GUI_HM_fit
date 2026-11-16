"""Pre-specified exploratory comparison of nested spectral objectives.

No hyperparameter tuning: one scale-based pilot is applied identically to all
cases. Results are not a confirmatory calibration. No production code is changed.
"""
import hashlib
import json
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from openpyxl.cell.cell import Cell

from hmfit_core import run_spectroscopy
from hmfit_core.processors import spectroscopy_processor as sp
from hmfit_core.solvers import NewtonRaphson
from hmfit_core.utils.nnls_utils import _solve_weighted_penalty_newton
from validation.phase31_comparison import golden, pc
from validation.phase36_diagnosis import _quiet


CASES = [('golden1', 'golden', 1, 'range', 'std', .0002),
         ('golden2', 'golden', 2, 'range', 'std', .0002),
         ('golden3', 'golden', 3, 'range', 'std', .0002),
         ('golden1_raw', 'golden', 1, 'off', 'none', .0002),
         ('golden1_baseline', 'golden', 1, 'range', 'none', .0002),
         ('golden1_weights', 'golden', 1, 'off', 'std', .0002),
         ('golden1_signal10', 'golden_signal10', 1, 'range', 'std', .0002),
         ('golden1_conc10', 'golden_conc10', 1, 'range', 'std', .0002),
         ('independent11', '11', 31, 'off', 'none', .0002),
         ('independent11_exact', '11', 31, 'off', 'none', 0.),
         ('independent12', '12', 32, 'off', 'none', .0002),
         ('independent12_exact', '12', 32, 'off', 'none', 0.)]
METHODS = ('legacy_300', 'legacy_exact', 'absolute_original', 'absolute_scaled',
           'absolute_unregularized', 'absolute_penalty_only', 'absolute_smooth_only')


def _metrics(out, mode, mu, alpha, true_eps):
    ed = out['export_data']
    C = np.asarray(ed['C'], float) @ np.asarray(ed['abs_group_map'], float)
    Y = np.asarray(ed['Y'], float).T
    A = np.asarray(ed['A_solver'], float).T
    W = np.broadcast_to(np.asarray(ed['weights'], float), Y.shape)
    L = sp._build_smoothness_laplacian(Y.shape[1])
    lb = np.asarray(ed['lower_bound'], float)[:, None]
    R = C@A-Y
    mult = W if mode == 'absolute' else np.ones_like(W)
    grad = C.T @ (mult**2*R) + mu*np.minimum(A, 0.) + alpha*(A@L)
    pg = np.where((A <= lb+1e-9) & (grad > 0), 0., grad)
    scale = max(float(np.max(np.abs(C.T@(mult**2*Y)))),
                float(np.max(np.abs(C.T@(mult**2*(C@A))))),
                float(np.max(np.abs(mu*np.minimum(A, 0.)))),
                float(np.max(np.abs(alpha*(A@L)))), np.finfo(float).tiny)
    data = float(np.sum((mult*R)**2))
    penalty = float(mu*np.sum(np.minimum(A, 0.)**2))
    smooth = float(alpha*np.sum((A@L)*A))
    return {'log_beta':ed['k'], 'optimizer_success':out['optimizer_result'].get('success'),
            'weighted_residual_rms':float(np.sqrt(np.mean((W*R)**2))),
            'actual_residual_rms':float(np.sqrt(np.mean(R**2))),
            'inner_objective':.5*(data+penalty+smooth),
            'outer_objective_rms':float(np.sqrt((np.sum((W*R)**2)+(penalty+smooth if mode=='absolute' else 0))/Y.size)),
            'kkt_normalized':float(np.max(np.abs(pg))/scale),
            'bound_violation':float(np.max(np.maximum(lb-A, 0.))),
            'spectral_roughness':float(np.sum((A@L)*A)),
            'coeff_norm':float(np.linalg.norm(A)),
            'coefficient_rmse_vs_raw_truth':float(np.sqrt(np.mean((A-true_eps)**2))),
            'bound_hits':int(np.count_nonzero((np.asarray(ed['k']) < 1e-5) | (np.asarray(ed['k']) > 6.-1e-5)))}


def run():
    records = []
    original = sp.solve_A_nnls_pgd
    def exact(C, Y, ridge=0., mu=.01, max_iters=300, lower_bound=None,
              alpha_smooth=0., smooth_matrix=None, weights_row=None):
        if weights_row is not None:
            return original(C, Y, ridge=ridge, mu=mu, max_iters=max_iters, lower_bound=lower_bound,
                            alpha_smooth=alpha_smooth, smooth_matrix=smooth_matrix, weights_row=weights_row)
        return _solve_weighted_penalty_newton(C, Y, np.ones_like(Y), ridge, mu, lower_bound,
                                              alpha_smooth, smooth_matrix, max_iters)
    try:
        with tempfile.TemporaryDirectory() as td:
            for case, kind, seed, baseline, weighting, noise in CASES:
                path = Path(td)/f'{case}.xlsx'
                if kind.startswith('golden'):
                    golden._build_dataset(path, seed=seed, tail_noise=.0015 if seed==3 else 0.)
                    cfg = golden._base_config(path)
                    truth = [3.]
                    if kind in ('golden_signal10', 'golden_conc10'):
                        workbook = load_workbook(path)
                        for row in workbook['Spectra'].iter_rows(min_row=2, min_col=2):
                            for cell in row:
                                assert isinstance(cell, Cell)
                                cell.value = float(str(cell.value)) * 10.
                        if kind == 'golden_conc10':
                            for row in workbook['Conc'].iter_rows(min_row=2):
                                for cell in row:
                                    assert isinstance(cell, Cell)
                                    cell.value = float(str(cell.value)) * 10.
                            truth = [2.]
                            cfg['initial_k'] = [2.]
                        workbook.save(path)
                else:
                    model = [[1,0],[0,1],[1,1]] if kind=='11' else [[1,0],[0,1],[1,1],[1,2]]
                    truth = [3.] if kind=='11' else [3.,2.5]
                    pc._spectro_dataset(path, truth, model, seed=seed, noise=noise)
                    cfg = pc._spectro_cfg(path, model, [2.8]*len(truth), [(0.,6.)]*len(truth))
                wl = np.linspace(240., 600., 31)
                if kind.startswith('golden'):
                    true_eps = np.vstack([amp*np.exp(-((wl-center)**2)/(2*width**2))
                                          for amp,center,width in [(1200.,265.,22.),(800.,320.,26.),(1600.,350.,30.)]])
                    if kind == 'golden_signal10':
                        true_eps *= 10.
                else:
                    true_eps = np.vstack([1200.*np.exp(-((wl-center)**2)/(2*25.**2))
                                          for center in np.linspace(265.,380.,len(cfg['modelo']))])
                model = np.asarray(cfg['modelo']).T
                totals = np.column_stack([np.full(10,1e-3),np.linspace(0.,2e-3,10)])
                if kind == 'golden_conc10':
                    totals *= 10.
                C = np.asarray(NewtonRaphson(totals, model, nas=[], model_sett='Free').concentraciones(np.asarray(cfg['initial_k']))[0])
                curvature = float(np.linalg.norm(C, 2)**2)
                lap = sp._build_smoothness_laplacian(31)
                pilot_mu = .01*curvature
                pilot_alpha = .01*curvature/float(np.linalg.norm(lap, 2))
                cfg.update(render_graphs=False, baseline_mode=baseline, baseline_start=450., baseline_end=600.,
                           weighting_mode=weighting, weighting_normalize=True, eps_solver_mode='soft_bound',
                           delta_mode='relative', delta_rel=.005)
                for method in METHODS:
                    mode = 'legacy' if method.startswith('legacy') else 'absolute'
                    mu, alpha = ((pilot_mu,pilot_alpha) if method=='absolute_scaled' else
                                 ((0.,0.) if method=='absolute_unregularized' else
                                  ((.01,0.) if method=='absolute_penalty_only' else
                                   ((0.,.02) if method=='absolute_smooth_only' else (.01,.02)))))
                    cfg.update(weight_regularization_mode=mode, mu=mu, alpha_smooth=alpha)
                    sp.solve_A_nnls_pgd = exact if method=='legacy_exact' else original
                    start = time.perf_counter()
                    row = {'case':case, 'kind':kind, 'seed':seed, 'truth':truth, 'method':method,
                           'mu':mu, 'alpha':alpha, 'baseline':baseline, 'weighting':weighting, 'noise':noise}
                    try:
                        out = _quiet(run_spectroscopy, cfg)
                        if not out.get('success'):
                            raise RuntimeError(out.get('error', 'fit unsuccessful'))
                        row.update(_metrics(out, mode, mu, alpha, true_eps))
                    except Exception as exc:
                        row['error'] = f'{type(exc).__name__}: {exc}'
                    row['seconds'] = time.perf_counter()-start
                    records.append(row)
    finally:
        sp.solve_A_nnls_pgd = original
    return records


if __name__ == '__main__':
    import scipy
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    provenance = {'revision':revision, 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'python':platform.python_version(), 'numpy':np.__version__, 'scipy':scipy.__version__,
                  'note':'exploratory pilot; working tree contains this validation script'}
    print(json.dumps({'provenance':provenance, 'records':run()}, indent=2, allow_nan=False))
