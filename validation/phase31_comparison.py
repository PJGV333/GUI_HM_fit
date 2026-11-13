"""Reproducible five-run comparison of Powell and explicit LM (phase 3.1).

Run with QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m validation.phase31_comparison --output validation/results/phase31.json
"""
import argparse
import contextlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hmfit_core import run_spectroscopy
from hmfit_core.api import run_nmr_fit
from hmfit_core.processors import nmr_processor
from hmfit_core.solvers import NewtonRaphson
from validation import literature_benchmark as lit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
import phase1_cases as pc
import test_spectro_golden_regression as golden


def _nmr_phase1(path, model, truth, noise=0.002, seed=4, n=12):
    h = np.full(n, 1e-3)
    g = np.linspace(0, 5e-3, n)
    solver = NewtonRaphson(np.column_stack([h, g]), np.asarray(model).T, nas=[1], model_sett='Free')
    _, co = solver.concentraciones(np.asarray(truth, dtype=float))
    species = [i for i, row in enumerate(model) if row[0] > 0]
    frac = co[:, species] * np.asarray([model[i][0] for i in species]) / h[:, None]
    rng = np.random.default_rng(seed)
    s1 = frac @ np.linspace(7., 7.6, len(species)) + rng.normal(0, noise, n)
    s2 = frac @ np.linspace(8., 7.5, len(species)) + rng.normal(0, noise, n)
    pd.DataFrame({'H': h, 'G': g, 's1': s1, 's2': s2}).to_excel(path, index=False, sheet_name='S')


def _measure(technique, cfg, repeats):
    runs = []
    for method in ('powell', 'explicit_lm'):
        for _ in range(repeats):
            opts = dict(cfg, optimizer=method)
            started = time.perf_counter()
            with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
                out = (run_spectroscopy(opts) if technique == 'UV-Vis'
                       else run_nmr_fit(opts, progress_cb=lambda _: None))
            elapsed = time.perf_counter() - started
            if not out.get('success'):
                runs.append({'method': method, 'elapsed': elapsed, 'success': False,
                             'error': str(out.get('error'))})
                continue
            ed = out['export_data']
            diag = out.get('optimizer_result') or {}
            if technique == 'UV-Vis':
                # The profiled objective is RMS normalized by data-point count.
                ssq = float(diag['objective_rms'] ** 2 * np.asarray(ed['Y']).size)
                weighted_ssq = ssq
            else:
                ssq = ed['unweighted_ssq']
                weighted_ssq = ed['weighted_ssq_fit']
            runs.append({'method': method, 'elapsed': elapsed, 'success': bool(diag.get('success', False)),
                         'reason': str(diag.get('message', '')), 'iterations': diag.get('nit', diag.get('iterations')),
                         'log_beta': ed['k'], 'se': ed['SE_log10K'], 'ssq': ssq,
                         'weighted_ssq': weighted_ssq, 'rms': dict(ed['stats_table']).get('RMS')})
    result = {}
    for method in ('powell', 'explicit_lm'):
        subset = [r for r in runs if r['method'] == method]
        result[method] = dict(subset[0], median_s=statistics.median(r['elapsed'] for r in subset),
                              repeats=repeats)
    return result


def run(repeats):
    results = {}
    # Exclude plot rendering equally for both optimizers; time includes parsing,
    # speciation, fitting, error calculation and export.
    nmr_processor.generate_figure_base64 = lambda *args, **kwargs: ''
    nmr_processor.generate_figure2_base64 = lambda *args, **kwargs: ''
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for cid in lit.case_ids():
            case = lit.load_case(cid)
            ok, why = lit.supported(case)
            if not ok:
                results['literature/' + cid] = {'unsupported': why}
                continue
            conc, shifts = lit.numeric_table(case)
            cfg = lit.gui_config(case, str(root / 'lit.xlsx'), conc, shifts)
            result = _measure('NMR', cfg, repeats)
            pw = result['powell']
            if pw.get('log_beta') is not None:
                mock = {'rms': pw['rms'], 'names': [s['name'] for s in case['species']],
                        'log_beta': pw['log_beta'], 'se': pw['se'], 'fixed': cfg['fixed_mask']}
                pw['published_reference_within_tolerance'] = lit.compare(case, mock)['levels']['L2']['pass']
            results['literature/' + cid] = result
            print('completed', cid, flush=True)
        m11 = [[1, 0], [0, 1], [1, 1]]
        m12 = [[1, 0], [0, 1], [1, 1], [1, 2]]
        specs = [('noisy', [3.], m11, [2.5], [(0., 6.)], 2e-4, 2e-3),
                 ('noiseless', [3.], m11, [2.5], [(0., 6.)], 0., 2e-3),
                 ('fixed', [4., 7.], m12, [3.5, 7.], [(0., 8.), (7., 7.)], 2e-4, 4e-3)]
        for name, truth, model, start, bounds, noise, gmax in specs:
            path = root / 'spectro.xlsx'
            pc._spectro_dataset(str(path), truth, model, noise=noise, g_max=gmax)
            cfg = pc._spectro_cfg(path, model, start, bounds)
            cfg['render_graphs'] = False
            results['phase1/spectro-' + name] = _measure('UV-Vis', cfg, repeats)
        for name, truth, model, start, fixed in [('11', [3.], m11, [2.5], [False]),
                                                  ('12-fixed', [4., 7.], m12, [3.5, 7.], [False, True])]:
            path = root / 'nmr.xlsx'
            _nmr_phase1(path, model, truth, n=15 if name == '12-fixed' else 12)
            cfg = {'file_path': str(path), 'nmr_sheet': 'S', 'conc_sheet': 'S',
                   'column_names': ['H', 'G'], 'signal_names': ['s1', 's2'],
                   'receptor_label': 'H', 'guest_label': 'G', 'modelo': model,
                   'non_abs_species': [1], 'algorithm': 'Newton-Raphson', 'model_settings': 'Free',
                   'initial_k': start, 'bounds': [{'min': 0., 'max': 8.} for _ in start], 'k_fixed': fixed}
            results['phase1/nmr-' + name] = _measure('NMR', cfg, repeats)
        for seed, tail in ((1, 0.), (2, 0.), (3, .0015)):
            path = root / 'golden.xlsx'
            golden._build_dataset(path, seed=seed, tail_noise=tail)
            cfg = golden._base_config(path)
            cfg.update(render_graphs=False, baseline_mode='range', baseline_start=450., baseline_end=600.,
                       baseline_auto_quantile=.2, weighting_mode='std', weighting_power=1.,
                       weighting_normalize=True, eps_solver_mode='soft_bound', mu=.01,
                       delta_mode='relative', delta_rel=.005, alpha_smooth=.02)
            results[f'golden/soft-bound-{seed}'] = _measure('UV-Vis', cfg, repeats)
    return {'repeats': repeats, 'timing': 'wall clock, plot rendering disabled equally', 'cases': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    output = run(args.repeats)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
