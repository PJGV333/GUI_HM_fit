"""Trace the historical 1:2 Newton failure without modifying its algorithm."""
import contextlib
import hashlib
import io
import json
import platform
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np
import scipy

from hmfit_core import run_spectroscopy
from hmfit_core.processors import spectroscopy_processor as sp
from hmfit_core.utils.nnls_utils import WeightedLinearSolveError
from validation.phase31_comparison import pc


def trace_failures():
    original = sp._solve_weighted_penalty_newton
    rows = []
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'case.xlsx'
        model = [[1,0],[0,1],[1,1],[1,2]]
        pc._spectro_dataset(path, [3.,2.5], model, seed=32, noise=2e-4)
        for label, mu, alpha in [('unregularized',0.,0.), ('historical',.01,.02)]:
            row = {'case':label, 'history':deque(maxlen=6), 'iterations':0, 'accepted_no_movement':0}
            cfg=pc._spectro_cfg(path, model, [2.8,2.8], [(0.,6.),(0.,6.)])
            cfg.update(render_graphs=False, baseline_mode='off', weighting_mode='none',
                       eps_solver_mode='soft_bound', delta_mode='relative', delta_rel=.005,
                       weight_regularization_mode='absolute', mu=mu, alpha_smooth=alpha)
            def capture(C,Y,W,ridge,mu_arg,lower,alpha_arg,L,max_iters):
                snapshots = deque(maxlen=6)
                calls = {'iterations':0,'accepted_no_movement':0}
                def tracer(frame,event,arg):
                    if event != 'line' or frame.f_code is not original.__code__:
                        return tracer
                    loc=frame.f_locals
                    if frame.f_lineno == 117:
                        calls['iterations']+=1
                    if frame.f_lineno == 137:
                        delta=np.asarray(loc['candidate'])-np.asarray(loc['x'])
                        if not np.any(delta):
                            calls['accepted_no_movement']+=1
                        snapshots.append({'iteration':calls['iterations'], 'step':float(loc['step']),
                            'step_norm':float(np.linalg.norm(delta)),
                            'objective':float(loc['value']),
                            'gradient_max':float(np.max(np.abs(loc['projected_grad']))),
                            'slope':float(loc['grad']@delta),
                            'active_count':int(np.sum(loc['active']))})
                    return tracer
                sys.settrace(tracer)
                try:
                    return original(C,Y,W,ridge,mu_arg,lower,alpha_arg,L,max_iters)
                except WeightedLinearSolveError as exc:
                    tb=exc.__traceback__
                    assert tb is not None
                    while tb.tb_next is not None:
                        tb=tb.tb_next
                    loc=tb.tb_frame.f_locals
                    x=np.asarray(loc['x']).reshape(np.asarray(C).shape[1],np.asarray(Y).shape[1])
                    data_grad=np.asarray(C).T @ (np.asarray(W)**2*(np.asarray(C)@x-np.asarray(Y)))
                    smooth_grad=float(alpha_arg)*(x@np.asarray(L)) if L is not None else np.zeros_like(x)
                    true_grad=data_grad + float(ridge)*x + float(mu_arg)*np.minimum(x,0.) + smooth_grad
                    direct_value=.5*(np.sum((np.asarray(W)*(np.asarray(C)@x-np.asarray(Y)))**2)
                                      +float(ridge)*np.sum(x*x)+float(mu_arg)*np.sum(np.minimum(x,0.)**2))
                    stable_value=direct_value
                    if L is not None:
                        d2=np.diff(np.eye(x.shape[1]),n=2,axis=0)
                        stable_value+=.5*float(alpha_arg)*np.sum((x@d2.T)**2)
                    row.update(error=str(exc),iterations=calls['iterations'],
                               accepted_no_movement=calls['accepted_no_movement'],history=list(snapshots),
                               failure_state={'gradient_normal_equations_max':float(np.max(np.abs(loc['projected_grad']))),
                                              'gradient_direct_max':float(np.max(np.abs(true_grad))),
                                              'gradient_difference_max':float(np.max(np.abs(true_grad.ravel()-loc['grad']))),
                                              'objective_quadratic':float(loc['value']),
                                              'objective_difference_squares':float(stable_value),
                                              'coefficient_abs_max':float(np.max(np.abs(x))),
                                              'lower_gap_min':float(np.min(loc['x']-loc['lower'])),
                                              'directional_derivative':float(loc['grad']@loc['direction']),
                                              'direction_norm':float(np.linalg.norm(loc['direction'])),
                                              'last_trial_step':float(loc['step'])},
                               system={'C':np.asarray(C).tolist(),'Y':np.asarray(Y).tolist(),
                                       'W':np.asarray(W).tolist(),'ridge':float(ridge),'mu':float(mu_arg),
                                       'lower':np.asarray(lower).tolist() if lower is not None else None,
                                       'alpha':float(alpha_arg),'L':np.asarray(L).tolist() if L is not None else None,
                                       'max_iters':int(max_iters)})
                    raise
                finally:
                    sys.settrace(None)
            sp._solve_weighted_penalty_newton=capture
            start=time.perf_counter()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    result=run_spectroscopy(cfg)
                row['returned_success']=bool(result.get('success'))
            except WeightedLinearSolveError:
                pass
            finally:
                sp._solve_weighted_penalty_newton=original
            row['seconds']=time.perf_counter()-start
            if isinstance(row['history'],deque):
                row['history']=list(row['history'])
            rows.append(row)
    return rows


if __name__ == '__main__':
    meta={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}
    print(json.dumps({'provenance':meta,'records':trace_failures()},indent=2,allow_nan=False))
