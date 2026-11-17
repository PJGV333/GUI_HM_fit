"""Capture the actual 1:2 production failure inputs, without altering the solver."""
import contextlib
import hashlib
import io
import json
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy

from hmfit_core import run_spectroscopy
from hmfit_core.processors import spectroscopy_processor as sp
from hmfit_core.utils.nnls_utils import WeightedLinearSolveError
from validation.phase31_comparison import pc
from validation.stage_e3_identifiability import bounded_reference
from validation.stage_e_golden_reference import refine_reference


def run_probe():
    model = [[1,0],[0,1],[1,1],[1,2]]
    records = []
    original = sp._solve_weighted_penalty_newton
    for method, mu, alpha in [('unregularized', 0., 0.), ('historical_penalties', .01, .02)]:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'data.xlsx'
            pc._spectro_dataset(path, [3.,2.5], model, seed=32, noise=2e-4)
            cfg = pc._spectro_cfg(path, model, [2.8,2.8], [(0.,6.),(0.,6.)])
            cfg.update(render_graphs=False, baseline_mode='off', weighting_mode='none',
                       eps_solver_mode='soft_bound', delta_mode='relative', delta_rel=.005,
                       weight_regularization_mode='absolute', mu=mu, alpha_smooth=alpha)
            row = {'method':method, 'mu':mu, 'alpha':alpha}
            def capture(C, Y, W, ridge, mu_inner, lower, alpha_inner, L, max_iters):
                try:
                    return original(C, Y, W, ridge, mu_inner, lower, alpha_inner, L, max_iters)
                except WeightedLinearSolveError as exc:
                    C, Y, W = (np.asarray(x, float) for x in (C,Y,W))
                    sv = np.linalg.svd(C, compute_uv=False)
                    row.update(error=str(exc), design_singular_values=sv.tolist(),
                               weighted_columns=int(Y.shape[1]), max_iters=int(max_iters),
                               weighted_is_unit=bool(np.array_equal(W, np.ones_like(W))))
                    try:
                        if not row['weighted_is_unit'] or ridge != 0 or lower is None:
                            raise ValueError('reference supports only unit weights, no ridge and a finite lower bound')
                        if mu_inner == 0:
                            A, residual, kkt = bounded_reference(C, Y, lower, alpha_inner)
                            objective = .5*float(residual@residual)
                        else:
                            lap = np.asarray(L, float) if L is not None else np.zeros((Y.shape[1],Y.shape[1]))
                            A = refine_reference(C, Y, lap, np.asarray(lower)[:,None], np.linalg.pinv(C)@Y,
                                                 mu=mu_inner, alpha=alpha_inner)
                            residual = C@A-Y
                            grad = C.T@residual + mu_inner*np.minimum(A,0.) + alpha_inner*(A@lap)
                            lb = np.asarray(lower)[:,None]
                            pg = np.where((A <= lb+1e-9)&(grad>0),0.,grad)
                            kkt = float(np.max(np.abs(pg)))
                            objective = .5*float(np.sum(residual**2)+mu_inner*np.sum(np.minimum(A,0.)**2)+alpha_inner*np.sum((A@lap)*A))
                        row.update(reference_kkt_max=kkt, reference_objective=objective,
                                   reference_min_slack=float(np.min(A-np.asarray(lower)[:,None])))
                        # Directional finite-difference check of the exact inner gradient.
                        lap = np.asarray(L, float) if L is not None else np.zeros((Y.shape[1],Y.shape[1]))
                        direction = np.random.default_rng(123).normal(size=A.shape)
                        direction /= np.linalg.norm(direction)
                        anchor = A + .2*direction
                        def objective_at(Z):
                            return .5*(np.sum((C@Z-Y)**2)+mu_inner*np.sum(np.minimum(Z,0.)**2)+alpha_inner*np.sum((Z@lap)*Z))
                        grad = C.T@(C@anchor-Y)+mu_inner*np.minimum(anchor,0.)+alpha_inner*(anchor@lap)
                        analytic = float(np.sum(grad*direction))
                        checks = []
                        for step in (1e-2, 1e-3, 1e-4):
                            fd=(objective_at(anchor+step*direction)-objective_at(anchor-step*direction))/(2*step)
                            checks.append({'step':step, 'finite_difference':float(fd),
                                           'absolute_difference':float(abs(fd-analytic))})
                        row.update(gradient_direction_analytic=analytic, gradient_checks=checks)
                    except Exception as ref_exc:
                        row['reference_error'] = type(ref_exc).__name__+': '+str(ref_exc)
                    raise
            sp._solve_weighted_penalty_newton = capture
            start = time.perf_counter()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    out = run_spectroscopy(cfg)
                row['fit_returned'] = bool(out.get('success'))
            except WeightedLinearSolveError as exc:
                row['fit_exception'] = str(exc)
            finally:
                sp._solve_weighted_penalty_newton = original
                row['seconds'] = time.perf_counter()-start
            records.append(row)
    return records


if __name__ == '__main__':
    provenance = {'revision':subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'python':platform.python_version(), 'numpy':np.__version__, 'scipy':scipy.__version__}
    print(json.dumps({'provenance':provenance, 'records':run_probe()}, indent=2, allow_nan=False))
