"""Independent reconstruction of the historical golden soft-bound PGD.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m
validation.stage_e_golden_reproduction. No historical artifact is overwritten.
The fixed-iteration loop below is a reference, not a production solver.
"""
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from hmfit_core import run_spectroscopy
from hmfit_core.processors import spectroscopy_processor as sp
from validation.phase31_comparison import golden
from validation.phase36_diagnosis import _quiet

ROOT = Path(__file__).resolve().parents[1]


def run():
    old = json.loads((ROOT / 'validation/results/phase31_comparison.json').read_text())['cases']
    output = {}
    with tempfile.TemporaryDirectory() as td:
        for seed in (1, 2, 3):
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
            step = 1. / (np.linalg.svd(C, compute_uv=False).max() ** 2 + .01
                         + .02 * np.linalg.svd(L, compute_uv=False).max() + 1e-12)
            A = initial.copy()
            ref = None
            entries = []
            started = time.perf_counter()
            for it in range(1, 3001):
                grad = C.T @ (C @ A - Y) + .01 * np.minimum(A, 0.) + .02 * (A @ L)
                A = np.maximum(A - step * grad, lower)
                if it in (300, 600, 3000):
                    if ref is None:
                        ref = A.copy()
                    grad = C.T @ (C @ A - Y) + .01 * np.minimum(A, 0.) + .02 * (A @ L)
                    pg = np.where((A <= lower + 1e-10) & (grad > 0), 0., grad)
                    r = C @ A - Y
                    inner = .5 * (np.sum(r*r) + .01 * np.sum(np.minimum(A, 0.)**2)
                                    + .02 * np.sum((A @ L) * A))
                    entries.append({'iterations': it, 'projected_gradient_norm': float(np.linalg.norm(pg)),
                                    'projected_gradient_max': float(np.max(np.abs(pg))),
                                    'delta_A_vs_300': float(np.linalg.norm(A-ref)),
                                    'outer_ssq': float(np.sum((W*r)**2)),
                                    'inner_objective': float(inner), 'elapsed_s': time.perf_counter()-started})
            output[str(seed)] = {'theta': theta.tolist(), 'shape': list(Y.shape), 'inner': entries}
    return output


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, allow_nan=False))
