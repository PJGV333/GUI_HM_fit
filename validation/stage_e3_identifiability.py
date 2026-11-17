"""Independent 1:2 equilibrium and profile references for E.3."""
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import brentq, lsq_linear, minimize_scalar


def equilibrium_12(totals, log_beta):
    """Mass-balance solution for H, G, HG and HG2, using a scalar bracket."""
    k1, k2 = 10.**np.asarray(log_beta, dtype=float)
    C = []
    for ht, gt in np.asarray(totals, dtype=float):
        def guest_balance(g):
            return g + ht*(k1*g + 2*k2*g*g)/(1+k1*g+k2*g*g) - gt
        g = 0. if gt == 0 else brentq(guest_balance, 0., gt, xtol=1e-15)
        h = ht/(1+k1*g+k2*g*g)
        C.append([h,g,k1*h*g,k2*h*g*g])
    return np.asarray(C)


def bounded_reference(C, Y, lower, alpha):
    """Independent BVLS reference for μ=0, including optional second differences."""
    C, Y = np.asarray(C, float), np.asarray(Y, float)
    species, channels = C.shape[1], Y.shape[1]
    design = np.kron(C, np.eye(channels))
    d2 = np.diff(np.eye(channels), n=2, axis=0)
    if alpha > 0:
        design = np.vstack((design, np.sqrt(alpha)*np.kron(np.eye(species), d2)))
    rhs = np.concatenate((Y.ravel(), np.zeros(design.shape[0]-Y.size)))
    lb = np.broadcast_to(np.asarray(lower)[:, None], (species, channels)).ravel()
    result = lsq_linear(design, rhs, bounds=(lb, np.inf), method='bvls', tol=1e-12, max_iter=1000)
    if not result.success:
        raise RuntimeError('independent BVLS failed: ' + result.message)
    residual = design@result.x-rhs
    gradient = design.T@residual
    pg = np.where((result.x <= lb+1e-9) & (gradient > 0), 0., gradient)
    return result.x.reshape(species, channels), residual, float(np.max(np.abs(pg), initial=0.))


def profile_objective(totals, Y, k, alpha):
    C = equilibrium_12(totals, k)
    # The studied relative-bound convention uses the unconstrained projection.
    lower = -.005 * np.max(np.abs(np.linalg.pinv(C) @ Y), axis=1)
    A, r, kkt = bounded_reference(C, Y, lower, alpha)
    return float(r @ r), r, kkt


def profile_coordinate(totals, Y, alpha, coordinate, fixed):
    """Reoptimize the other nonlinear constant; BVLS profiles all coefficients."""
    other = 1-coordinate
    def evaluate(value):
        k = np.empty(2)
        k[coordinate], k[other] = fixed, value
        f, _, kkt = profile_objective(totals, Y, k, alpha)
        return f, kkt
    grid = np.linspace(0., 6., 7)
    grid_results = [evaluate(value) for value in grid]
    index = int(np.argmin([r[0] for r in grid_results]))
    lo, hi = grid[max(0, index-1)], grid[min(len(grid)-1, index+1)]
    opt = minimize_scalar(lambda value: evaluate(value)[0], bounds=(lo, hi), method='bounded',
                          options={'xatol': .002, 'maxiter': 35})
    candidates = [(float(grid[index]), *grid_results[index]), (float(opt.x), *evaluate(opt.x))]
    value, objective, kkt = min(candidates, key=lambda item: item[1])
    return {'fixed':float(fixed), 'nuisance':value, 'objective':objective,
            'kkt_abs':kkt, 'outer_success':bool(opt.success)}


def sensitivity(totals, Y, alpha, k, step):
    columns = []
    for i in range(2):
        diff = np.zeros(2)
        diff[i] = step
        rp = profile_objective(totals, Y, k+diff, alpha)[1]
        rm = profile_objective(totals, Y, k-diff, alpha)[1]
        columns.append((rp-rm)/(2*step))
    J = np.column_stack(columns)
    singular = np.linalg.svd(J, compute_uv=False)
    norms = np.linalg.norm(J, axis=0)
    corr = float((J[:,0] @ J[:,1]) / (norms[0]*norms[1])) if np.all(norms > 0) else None
    return {'singular_values':singular.tolist(), 'column_norms':norms.tolist(),
            'column_correlation':corr, 'J':J}


def run_profiles():
    totals = np.column_stack((np.full(10, 1e-3), np.linspace(0., 2e-3, 10)))
    truth = np.array([3., 2.5])
    wl = np.linspace(240., 600., 31)
    eps = np.vstack([1200*np.exp(-((wl-center)**2)/(2*25.**2)) for center in np.linspace(265.,380.,4)])
    clean = equilibrium_12(totals, truth) @ eps
    noise = np.random.default_rng(32).normal(0., 2e-4, clean.shape)
    pilot_alpha = 7.541875008433269e-9
    output = []
    for label, Y in [('noiseless', clean), ('seed32_noise_2e-4', clean+noise)]:
        for alpha in [0., pilot_alpha]:
            start = time.perf_counter()
            item = {'case':label, 'alpha':alpha, 'mu':0., 'profiles':{}, 'errors':[]}
            try:
                base = profile_objective(totals, Y, truth, alpha)
                s1, s2 = (sensitivity(totals, Y, alpha, truth, step) for step in (1e-3, 5e-4))
                discrepancy = float(np.linalg.norm(s1['J']-s2['J']) / max(np.linalg.norm(s2['J']), np.finfo(float).tiny))
                s1.pop('J'); s2.pop('J')
                item.update(objective_at_truth=base[0], kkt_at_truth=base[2],
                            sensitivity=s2, sensitivity_step_relative_difference=discrepancy)
                for coordinate in range(2):
                    values = [0., 1., 2., 2.5, 3., 4., 6.]
                    item['profiles'][str(coordinate)] = [profile_coordinate(totals, Y, alpha, coordinate, x) for x in values]
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                item['errors'].append(type(exc).__name__ + ': ' + str(exc))
            item['seconds'] = time.perf_counter()-start
            output.append(item)
    C = equilibrium_12(totals, truth)
    return {'truth':truth.tolist(), 'noise_sd':.0002, 'seed':32,
            'design_singular_values_at_truth':np.linalg.svd(C, compute_uv=False).tolist(), 'runs':output}


if __name__ == '__main__':
    provenance = {'revision':subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'python':platform.python_version(), 'numpy':np.__version__, 'scipy':scipy.__version__}
    print(json.dumps({'provenance':provenance, 'profiles':run_profiles()}, indent=2, allow_nan=False))
