"""Small explicit, bound-constrained Levenberg--Marquardt solver.

The callable returns the residual and its full Jacobian at a parameter vector.
No normal equations are used to compute a step.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class LMResult:
    x: np.ndarray
    fun: np.ndarray
    cost: float
    success: bool
    reason: str
    nit: int
    scaled_gradient: float
    active_bounds: np.ndarray


def _gradient(r, j, x, lower, upper):
    g = j.T @ r
    active = ((x <= lower) & (g > 0)) | ((x >= upper) & (g < 0))
    denom = np.linalg.norm(j, axis=0) * np.linalg.norm(r)
    scaled = np.divide(np.abs(g), denom, out=np.zeros_like(g), where=denom > 0)
    scaled[active] = 0
    return g, active, float(np.max(scaled, initial=0.0))


def explicit_lm(fun_jac, x0, bounds=None, *, ftol=1e-10, xtol=1e-10,
                gtol=1e-8, maxiter=200, damping=1e-3):
    """Minimize ||r||² using scaled damping and active-bound projection."""
    x = np.asarray(x0, dtype=float).ravel().copy()
    if bounds is None:
        bounds = [(None, None)] * x.size
    if len(bounds) != x.size:
        raise ValueError("bounds length must equal parameter count")
    lower = np.array([-np.inf if b[0] is None else b[0] for b in bounds], float)
    upper = np.array([np.inf if b[1] is None else b[1] for b in bounds], float)
    if np.any(lower > upper) or damping <= 0:
        raise ValueError("invalid bounds or damping")
    x = np.clip(x, lower, upper)

    def evaluate(v):
        r, j = fun_jac(v)
        r = np.asarray(r, float).ravel()
        j = np.asarray(j, float)
        if j.shape != (r.size, x.size) or not np.all(np.isfinite(r)) or not np.all(np.isfinite(j)):
            raise ValueError("invalid residual or Jacobian")
        return r, j

    r, j = evaluate(x)
    cost = float(r @ r / 2)
    reason = 'maxiter'
    success = False
    nit = 0
    for nit in range(1, maxiter + 1):
        _, active, scaled = _gradient(r, j, x, lower, upper)
        if scaled < gtol:
            reason, success = 'gtol', True
            break
        free = ~active & (lower < upper)
        if not np.any(free):
            reason, success = 'active_bounds', True
            break
        jf = j[:, free]
        diag = np.linalg.norm(jf, axis=0)
        diag = np.maximum(diag, np.finfo(float).tiny ** 0.5)
        aug = np.vstack((jf, np.sqrt(damping) * np.diag(diag)))
        rhs = np.concatenate((-r, np.zeros(diag.size)))
        step = np.zeros_like(x)
        step[free] = np.linalg.lstsq(aug, rhs, rcond=None)[0]
        trial = np.clip(x + step, lower, upper)
        step = trial - x
        if np.linalg.norm(step) / (np.linalg.norm(x) + xtol) < xtol:
            reason, success = 'xtol', True
            break
        rt, jt = evaluate(trial)
        trial_cost = float(rt @ rt / 2)
        if trial_cost < cost:
            relative = (cost - trial_cost) / max(cost, np.finfo(float).tiny)
            x, r, j, cost = trial, rt, jt, trial_cost
            damping = max(damping / 3, 1e-15)
            if relative < ftol:
                reason, success = 'ftol', True
                break
        else:
            damping = min(damping * 10, 1e30)
    _, _, scaled = _gradient(r, j, x, lower, upper)
    return LMResult(x, r, cost, success, reason, nit, scaled,
                    (x <= lower) | (x >= upper))
