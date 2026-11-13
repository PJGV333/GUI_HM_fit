"""cProfile literature bank and one full-refit bootstrap replica per technique."""
import argparse
import contextlib
import cProfile
import io
import json
import os
import pstats
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import hmfit_core.potentiometry as pot
from hmfit_core import run_spectroscopy
from hmfit_core.api import run_nmr_fit
from hmfit_core.acid_base import (NMRAcidBaseDataset, SpectroscopicAcidBaseDataset,
                                  distribution_fractions_from_pH, fit_nmr_acid_base,
                                  fit_spectroscopy_acid_base, make_simple_acid_base_system, pka_to_log_beta)
from hmfit_core.utils.errors import bootstrap_full_refit
from validation import literature_benchmark as lit
from validation.phase31_comparison import _nmr_phase1, pc


def profile(fn):
    p = cProfile.Profile()
    with contextlib.redirect_stdout(io.StringIO()):
        result = p.runcall(fn)
    stats = pstats.Stats(p)
    rows = []
    for (path, line, function), (primitive, calls, self_time, cumulative, _) in stats.stats.items():
        if '/GUI_HM_fit/' in path and '/.venv/' not in path:
            rows.append({'function': f'{Path(path).name}:{line}:{function}', 'calls': calls,
                         'self_s': self_time, 'cumulative_s': cumulative})
    rows.sort(key=lambda row: row['cumulative_s'], reverse=True)
    return {'profiled_total_s': stats.total_tt, 'top_project_functions': rows[:12],
            'result': result}


def paired(fn):
    """Profile identical workload with scalar and vector potentiometry paths."""
    current = pot.simulate_pH_titration
    try:
        pot.simulate_pH_titration = pot._simulate_pH_titration_reference
        before = profile(fn)
        pot.simulate_pH_titration = current
        after = profile(fn)
    finally:
        pot.simulate_pH_titration = current
    return {'before': before, 'after': after}


def _boot(make_data, refit, initial):
    out = bootstrap_full_refit(initial, make_data, refit, 1, seed=20261113)
    return {'n_success': int(out['n_success']), 'n_fail': int(out['n_fail']),
            'samples': np.asarray(out['samples']).tolist()}


def run():
    from hmfit.kinetics.data.dataset import KineticsDataset
    from hmfit.kinetics.fit.objective import GlobalKineticsObjective
    from hmfit.kinetics.fit.optimizer import fit_global
    from hmfit.kinetics.mechanism_editor.parser import parse_mechanism
    from hmfit.kinetics.model.kinetics_model import KineticsModel

    results = {}
    def literature():
        cases = lit.run_all()['cases']
        return {'case_count': len(cases), 'estimates': {
            key: value.get('fit', {}).get('log_beta') for key, value in cases.items()
        }}
    results['literature_bank'] = paired(literature)
    system = make_simple_acid_base_system(analytical_concentration=.01, pka=[7., 4.], base_charge=-2)
    volumes = np.linspace(0, 35., 60)
    exp = pot.PotentiometryExperiment(initial_volume=10., titrant_volumes=volumes,
                                     analyte_concentration=.01, titrant_concentration=.01)
    truth = pot._simulate_pH_titration_reference(exp, system)
    noise = np.random.default_rng(17).normal(0, .002, truth.size)
    make_pot = lambda rng, wild: truth + noise * rng.choice([-1., 1.], truth.shape)
    def refit_pot(data, theta, **kwargs):
        fit = pot.fit_potentiometry(replace(exp, measured_pH=data), system, initial_pka=theta)
        return fit.theta_hat, fit.success, {}
    current = pot.simulate_pH_titration
    try:
        for label, implementation in [('before', pot._simulate_pH_titration_reference), ('after', current)]:
            pot.simulate_pH_titration = implementation
            results['bootstrap_potentiometry_' + label] = profile(lambda: _boot(make_pot, refit_pot, [6.5, 4.5]))
    finally:
        pot.simulate_pH_titration = current

    ph = np.linspace(2., 10., 25)
    base = distribution_fractions_from_pH(ph, pka_to_log_beta([7., 4.])) @ np.array([7.9, 7.45, 7.])
    residual = np.random.default_rng(2).normal(0, .003, base.size)
    make_ab = lambda rng, wild: base + residual * rng.choice([-1., 1.], base.shape)
    def refit_ab(data, theta, **kwargs):
        fit = fit_nmr_acid_base(NMRAcidBaseDataset(pH=ph, shifts=data[:, None], nuclei_labels=['H1']), system, initial_pka=theta)
        return fit.theta_hat, fit.success, {}
    results['bootstrap_acid_base_nmr'] = paired(lambda: _boot(make_ab, refit_ab, [6.5, 4.5]))
    fractions = distribution_fractions_from_pH(ph, pka_to_log_beta([7., 4.]))
    signals = fractions @ np.array([[.1, .3, .7], [.6, .8, .4], [1., .5, .2]])
    signal_noise = np.random.default_rng(3).normal(0, .002, signals.shape)
    make_ab_spec = lambda rng, wild: signals + signal_noise * rng.choice([-1., 1.], signals.shape)
    def refit_ab_spec(data, theta, **kwargs):
        dataset = SpectroscopicAcidBaseDataset(pH=ph, signal=data)
        fit = fit_spectroscopy_acid_base(dataset, system, initial_pka=theta)
        return fit.theta_hat, fit.success, {}
    results['bootstrap_acid_base_spectroscopy'] = paired(lambda: _boot(make_ab_spec, refit_ab_spec, [6.5, 4.5]))

    model = KineticsModel(parse_mechanism(pc.FIRST_ORDER))
    t = np.linspace(0, 4, 40)
    y0 = {'A': 1., 'B': 0., 'C': 0.}
    C = model.solve_concentrations(t, y0, {'k1': 1.2, 'k2': .4})
    D = C @ np.random.default_rng(1).random((3, 12))
    kres = np.random.default_rng(3).normal(0, .01, D.shape)
    make_kin = lambda rng, wild: D + kres * rng.choice([-1., 1.], D.shape)
    def refit_kin(data, theta, **kwargs):
        obj = GlobalKineticsObjective(model, [KineticsDataset(t=t, y0=y0, D=data)], param_names=['k1', 'k2'])
        fit = fit_global(obj, theta, bounds=(0., np.inf))
        return np.array([fit.params['k1'], fit.params['k2']]), fit.result.success, {}
    results['bootstrap_kinetics'] = paired(lambda: _boot(make_kin, refit_kin, [1., .5]))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model11 = [[1, 0], [0, 1], [1, 1]]
        truepath, noisypath = root / 'true.xlsx', root / 'noisy.xlsx'
        pc._spectro_dataset(str(truepath), [3.], model11, noise=0.)
        pc._spectro_dataset(str(noisypath), [3.], model11)
        true_s = pd.read_excel(truepath, sheet_name='Spectra', index_col=0)
        noisy_s = pd.read_excel(noisypath, sheet_name='Spectra', index_col=0)
        conc = pd.read_excel(truepath, sheet_name='Conc')
        def make_spec(rng, wild):
            return true_s + (noisy_s - true_s) * rng.choice([-1., 1.], true_s.shape)
        def refit_spec(data, theta, **kwargs):
            path = root / 'specstar.xlsx'
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                data.to_excel(writer, sheet_name='Spectra')
                conc.to_excel(writer, sheet_name='Conc', index=False)
            cfg = pc._spectro_cfg(path, model11, theta.tolist(), [(0., 6.)])
            cfg['render_graphs'] = False
            out = run_spectroscopy(cfg)
            return out['export_data']['k'], out['optimizer_result']['success'], {}
        results['bootstrap_spectroscopy'] = paired(lambda: _boot(make_spec, refit_spec, [2.5]))

        _nmr_phase1(truepath, model11, [3.], noise=0.)
        _nmr_phase1(noisypath, model11, [3.])
        true_n = pd.read_excel(truepath, sheet_name='S')
        noisy_n = pd.read_excel(noisypath, sheet_name='S')
        def make_nmr(rng, wild):
            data = true_n.copy()
            for name in ['s1', 's2']:
                data[name] = true_n[name] + (noisy_n[name] - true_n[name]) * rng.choice([-1., 1.], len(data))
            return data
        def refit_nmr(data, theta, **kwargs):
            path = root / 'nmrstar.xlsx'
            data.to_excel(path, sheet_name='S', index=False)
            cfg = {'file_path': str(path), 'nmr_sheet': 'S', 'conc_sheet': 'S', 'column_names': ['H', 'G'],
                   'signal_names': ['s1', 's2'], 'receptor_label': 'H', 'guest_label': 'G', 'modelo': model11,
                   'non_abs_species': [1], 'algorithm': 'Newton-Raphson', 'optimizer': 'powell',
                   'model_settings': 'Free', 'initial_k': theta.tolist(), 'bounds': [{'min': 0., 'max': 8.}]}
            out = run_nmr_fit(cfg, progress_cb=lambda _: None)
            return out['export_data']['k'], out['optimizer_result']['success'], {}
        results['bootstrap_nmr'] = paired(lambda: _boot(make_nmr, refit_nmr, [2.5]))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    results = run()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
