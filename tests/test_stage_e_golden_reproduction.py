"""Historical golden PGD reproduction, independent of the production loop."""
import json
from pathlib import Path

import numpy as np

from validation.stage_e_golden_reproduction import run


def test_historical_golden_inner_reproduction():
    expected = json.loads((Path(__file__).resolve().parents[1] /
                           'validation/results/phase36_golden_smoothness.json').read_text())
    actual = run()
    for seed in ('1', '2', '3'):
        assert [r['iterations'] for r in actual[seed]['inner']] == [300, 600, 3000]
        for observed, ref in zip(actual[seed]['inner'], expected[seed]['inner']):
            for field in ('projected_gradient_norm', 'projected_gradient_max',
                          'delta_A_vs_300', 'outer_ssq'):
                np.testing.assert_allclose(observed[field], ref[field], rtol=2e-6, atol=1e-8,
                                           err_msg=f'{seed}: {field}')
