"""Run the E.4 research-only Newton candidate on the two captured systems."""
import hashlib
import json
import platform
import subprocess
import time
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import scipy

from hmfit_core.processors.spectroscopy_processor import spectral_inner_optimality
from hmfit_core.utils.nnls_utils import WeightedLinearSolveError
from validation.stage_e3_identifiability import bounded_reference
from validation.stage_e_golden_reference import refine_reference
from validation.stage_e4_newton_candidate import _solve_weighted_penalty_newton_verified

ROOT = Path(__file__).resolve().parents[1]


def high_precision_gradient(C,Y,A,L,mu,alpha):
    """Independent 60-digit evaluation of the captured unit-weight gradient."""
    with localcontext() as context:
        context.prec=60
        dc,dy,da,dl = ([[Decimal.from_float(float(v)) for v in row] for row in array]
                       for array in (C,Y,A,L))
        dmu,dalpha=Decimal.from_float(float(mu)),Decimal.from_float(float(alpha))
        pred=[[sum(dc[i][k]*da[k][j] for k in range(C.shape[1]))
               for j in range(Y.shape[1])] for i in range(C.shape[0])]
        grad=np.empty_like(A)
        for k in range(C.shape[1]):
            for j in range(Y.shape[1]):
                data=sum(dc[i][k]*(pred[i][j]-dy[i][j]) for i in range(C.shape[0]))
                penalty=dmu*min(da[k][j],Decimal(0))
                smooth=dalpha*sum(da[k][q]*dl[q][j] for q in range(Y.shape[1]))
                grad[k,j]=float(data+penalty+smooth)
    return grad


def run():
    records = json.loads((ROOT/'validation/results/stage_e4_failure_trace.json').read_text())['records']
    result=[]
    for row in records:
        p=row['system']
        C,Y,W,lb=(np.asarray(p[k],float) for k in ('C','Y','W','lower'))
        L=np.asarray(p['L'],float) if p['L'] is not None else None
        mu,alpha=p['mu'],p['alpha']
        D=np.diff(np.eye(Y.shape[1]),n=2,axis=0)
        if mu==0:
            reference,_,_=bounded_reference(C,Y,lb,alpha)
        else:
            reference=refine_reference(C,Y,L,lb[:,None],np.linalg.pinv(C)@Y,mu=mu,alpha=alpha)
        def obj(A):
            residual=W*(C@A-Y)
            return .5*(np.sum(residual**2)+p['ridge']*np.sum(A**2)+mu*np.sum(np.minimum(A,0.)**2)
                       +alpha*np.sum((A@D.T)**2))
        start=time.perf_counter()
        try:
            A,info=_solve_weighted_penalty_newton_verified(C,Y,W,p['ridge'],mu,lb,alpha,L,p['max_iters'],return_info=True)
            status='verified'
        except WeightedLinearSolveError as exc:
            if not hasattr(exc,'last_iterate'):
                raise
            A=getattr(exc,'last_iterate')
            info={'iterations':getattr(exc,'iterations'),'rejected_steps':getattr(exc,'rejected_steps'),'stop_reason':str(exc)}
            status='failed_explicitly'
        seconds=time.perf_counter()-start
        diag=spectral_inner_optimality(C,Y,A,eps_solver_mode='soft_bound',weight_regularization_mode='absolute',
                    mu=mu,alpha_smooth=alpha,smooth_matrix=L,lower_bound=lb,weights_row=W)
        audit=None
        if alpha and np.array_equal(W,np.ones_like(W)):
            precise=high_precision_gradient(C,Y,A,L,mu,alpha)
            terms=(C.T@(C@A),C.T@Y,mu*np.minimum(A,0.),alpha*(A@L))
            scale=np.maximum(np.maximum.reduce([np.abs(term) for term in terms]),np.finfo(float).tiny)
            naive=terms[0]-terms[1]+terms[2]+terms[3]
            stable=C.T@(C@A-Y)+mu*np.minimum(A,0.)+alpha*((A@D.T)@D)
            audit={'precision_digits':60,'max_normalized_gradient':float(np.max(np.abs(precise)/scale)),
                   'existing_gradient_max_abs_error':float(np.max(np.abs(naive-precise))),
                   'difference_squared_gradient_max_abs_error':float(np.max(np.abs(stable-precise)))}
        result.append({'case':row['case'],'before_error':row['error'],'before_iterations':row['iterations'],
                       'before_projected_gradient_absolute':row['failure_state']['gradient_normal_equations_max'],
                       'status':status,'info':info,'kkt':diag,'feasibility_min_slack':float(np.min(A-lb[:,None])),
                       'objective_candidate':float(obj(A)),'objective_reference':float(obj(reference)),
                       'prediction_difference_norm':float(np.linalg.norm(C@A-C@reference)),'decimal_audit':audit,'seconds':seconds})
    return result


if __name__=='__main__':
    files=['validation/stage_e4_newton_candidate.py','validation/stage_e4_newton_candidate_probe.py']
    provenance={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
                'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}
    print(json.dumps({'provenance':provenance,'records':run()},indent=2,allow_nan=False))
