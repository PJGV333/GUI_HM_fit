"""Compare scalar and vectorized potentiometry full-refit replicas."""
import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import hmfit_core.potentiometry as pot
from hmfit_core.acid_base import make_simple_acid_base_system


def run():
    system = make_simple_acid_base_system(analytical_concentration=.01, pka=[7., 4.], base_charge=-2)
    exp = pot.PotentiometryExperiment(initial_volume=10., titrant_volumes=np.linspace(0, 35., 60),
                                     analyte_concentration=.01, titrant_concentration=.01)
    true_ph = pot._simulate_pH_titration_reference(exp, system)
    rng = np.random.default_rng(20261113)
    observed = true_ph + rng.normal(0, .002, len(true_ph))
    exp = replace(exp, measured_pH=observed)
    # Fixed bootstrap residual draw; five timings of the same full-refit replica.
    residual_draw = rng.choice(observed - true_ph, len(true_ph), replace=True)
    replica = replace(exp, measured_pH=true_ph + residual_draw)
    original = pot.simulate_pH_titration
    rows = {}
    try:
        for name, implementation in [('scalar_brent', pot._simulate_pH_titration_reference),
                                     ('vectorized', original)]:
            pot.simulate_pH_titration = implementation
            times = []
            for _ in range(5):
                start = time.perf_counter()
                fit = pot.fit_potentiometry(replica, system, initial_pka=[6.5, 4.5])
                times.append(time.perf_counter() - start)
            rows[name] = {'times_s': times, 'median_s': statistics.median(times),
                          'success': bool(fit.success), 'theta': fit.theta_hat.tolist(),
                          'se': np.sqrt(np.maximum(np.diag(fit.covariance_matrix), 0)).tolist(),
                          'ssq': float(fit.residuals @ fit.residuals)}
    finally:
        pot.simulate_pH_titration = original
    rows['speedup'] = rows['scalar_brent']['median_s'] / rows['vectorized']['median_s']
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
