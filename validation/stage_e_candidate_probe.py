"""Compare historical PGD and a validation-only exact-inner candidate.

The candidate deliberately keeps the historical outer objective and all weights;
this is not a production solver or a proposed change to the objective.
"""
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from hmfit_core import run_spectroscopy
from hmfit_core.processors import spectroscopy_processor as sp
from hmfit_core.utils.nnls_utils import _solve_weighted_penalty_newton
from validation.phase31_comparison import golden
from validation.phase36_diagnosis import _quiet


def run():
    original = sp.solve_A_nnls_pgd
    output = {}

    def candidate(C, Y, ridge=0., mu=.01, max_iters=300, lower_bound=None,
                  alpha_smooth=0., smooth_matrix=None, weights_row=None):
        if weights_row is not None:
            return original(C, Y, ridge=ridge, mu=mu, max_iters=max_iters,
                            lower_bound=lower_bound, alpha_smooth=alpha_smooth,
                            smooth_matrix=smooth_matrix, weights_row=weights_row)
        return _solve_weighted_penalty_newton(C, Y, np.ones_like(Y), ridge, mu, lower_bound,
                                              alpha_smooth, smooth_matrix, max_iters)

    try:
        with tempfile.TemporaryDirectory() as td:
            for seed in (1, 2, 3):
                path = Path(td) / 'gold.xlsx'
                golden._build_dataset(path, seed=seed, tail_noise=.0015 if seed == 3 else 0.)
                cfg = golden._base_config(path)
                cfg.update(render_graphs=False, baseline_mode='range', baseline_start=450., baseline_end=600.,
                           baseline_auto_quantile=.2, weighting_mode='std', weighting_power=1.,
                           weighting_normalize=True, eps_solver_mode='soft_bound', mu=.01,
                           delta_mode='relative', delta_rel=.005, alpha_smooth=.02)
                records = {}
                for label, impl in [('historical', original), ('candidate', candidate)]:
                    sp.solve_A_nnls_pgd = impl
                    start = time.perf_counter()
                    result = _quiet(run_spectroscopy, cfg)
                    elapsed = time.perf_counter() - start
                    if not result.get('success'):
                        raise RuntimeError(f'{seed} {label}: {result}')
                    ed = result['export_data']
                    records[label] = {'log_beta':ed['k'], 'se_log_beta':ed['SE_log10K'],
                                      'rms':result['statistics']['RMS'],
                                      'covfit':dict(ed['stats_table'])['covfit'],
                                      'seconds':elapsed}
                output[str(seed)] = records
    finally:
        sp.solve_A_nnls_pgd = original
    return output


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, allow_nan=False))
