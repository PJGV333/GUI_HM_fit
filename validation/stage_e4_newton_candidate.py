"""Validation-only projected-Newton experiment; not used in production."""
import numpy as np
from hmfit_core.utils.nnls_utils import WeightedLinearSolveError
from hmfit_core.processors.spectroscopy_processor import spectral_inner_optimality


def _solve_weighted_penalty_newton_verified(C, Y, W, ridge, mu, lower_bound,
                                             alpha_smooth, smooth_matrix, max_iters,
                                             *, return_info=False):
    C,Y,W = (np.asarray(v,dtype=float) for v in (C,Y,W))
    species, channels = C.shape[1], Y.shape[1]
    size = species*channels
    H = np.zeros((size,size))
    for j in range(channels):
        idx=np.arange(species)*channels+j
        H[np.ix_(idx,idx)]=C.T@((W[:,j,None]**2)*C)
    H.flat[::size+1]+=ridge
    L = np.asarray(smooth_matrix,float) if alpha_smooth and smooth_matrix is not None else None
    D = np.diff(np.eye(channels),n=2,axis=0) if channels>=3 else np.zeros((0,channels))
    canonical = L is not None and np.array_equal(L,D.T@D)
    if L is not None:
        H += alpha_smooth*np.kron(np.eye(species),L)
    lower=np.full(size,-np.inf)
    if lower_bound is not None:
        lb=np.asarray(lower_bound,float)
        if lb.ndim==1: lb=lb[:,None]
        lower=np.broadcast_to(lb,(species,channels)).ravel()
    finite=np.isfinite(lower)
    x=np.maximum(np.zeros(size),lower)
    rejected=0
    def calc(z):
        a=z.reshape(species,channels)
        residual=W*(C@a-Y)
        grad=C.T@(W*residual)+ridge*a+mu*np.minimum(a,0.)
        if L is not None:
            grad += alpha_smooth*((a@D.T)@D if canonical else a@L)
        return a,residual,grad.ravel()
    for iteration in range(int(max_iters)):
        A,residual,grad=calc(x)
        check=spectral_inner_optimality(C,Y,A,eps_solver_mode='soft_bound' if lower_bound is not None else 'soft_penalty',
                    weight_regularization_mode='absolute',mu=mu,alpha_smooth=alpha_smooth,
                    smooth_matrix=L,lower_bound=lower_bound,weights_row=W)
        if check['status']=='verified':
            info={'iterations':iteration+1,'rejected_steps':rejected,'stop_reason':'kkt'}
            return (A,info) if return_info else A
        if check['status']=='not_verified':
            raise WeightedLinearSolveError('experimental Newton unverifiable: '+check['reason'])
        threshold=np.zeros(size)
        threshold[finite]=1e-10*np.maximum(1.,np.abs(lower[finite]))
        active=finite & ((x-lower)<=threshold) & (grad>0)
        free=np.flatnonzero(~active)
        if free.size==0: raise WeightedLinearSolveError('experimental Newton no free direction')
        h=H[np.ix_(free,free)].copy()
        h.flat[::len(free)+1] += mu*(x[free]<0)
        scale=max(float(np.max(np.abs(np.diag(h)))),1e-20)
        direction=np.zeros(size)
        try: direction[free]=np.linalg.solve(h+np.eye(len(free))*scale*1e-12,-grad[free])
        except np.linalg.LinAlgError as exc: raise WeightedLinearSolveError('experimental Newton singular') from exc
        for k in range(50):
            candidate=np.maximum(x+(0.5**k)*direction,lower)
            delta=candidate-x
            slope=float(grad@delta)
            if not np.any(delta) or slope>=0:
                rejected+=1
                continue
            da=delta.reshape(species,channels)
            dr=W*(C@da)
            df=float(np.sum(residual*dr)+.5*np.sum(dr**2)+ridge*(np.sum(A*da)+.5*np.sum(da**2)))
            before=np.minimum(A,0.); after=np.minimum(A+da,0.)
            df+=.5*mu*float(np.sum((after-before)*(after+before)))
            if L is not None:
                if canonical:
                    b,d=A@D.T,da@D.T
                    df+=alpha_smooth*float(np.sum(b*d)+.5*np.sum(d*d))
                else:
                    df+=alpha_smooth*float(np.sum((A@L)*da)+.5*np.sum((da@L)*da))
            if df <= 1e-4*slope:
                x=candidate
                break
            rejected+=1
        else:
            error = WeightedLinearSolveError(f'experimental Newton line search failed: iteration={iteration+1}, rejected={rejected}, diagnostics={check}')
            setattr(error, 'last_iterate', A.copy())
            setattr(error, 'diagnostics', check)
            setattr(error, 'iterations', iteration+1)
            setattr(error, 'rejected_steps', rejected)
            raise error
    raise WeightedLinearSolveError('experimental Newton did not converge')
