"""Captured pre-change systems and their independent interior references."""
import json
from pathlib import Path

import numpy as np
import pytest

from validation.stage_e3_identifiability import bounded_reference
from validation.stage_e_golden_reference import refine_reference

PATH = Path(__file__).resolve().parents[1] / 'validation/results/stage_e4_failure_trace.json'


def systems():
    return json.loads(PATH.read_text())['records']


@pytest.mark.parametrize('index', [0, 1])
def test_historical_failure_system_has_independent_feasible_reference(index):
    row = systems()[index]
    s = row['system']
    C, Y, lower = (np.asarray(s[x], float) for x in ('C', 'Y', 'lower'))
    if s['mu'] == 0.:
        A, residual, pg = bounded_reference(C, Y, lower, s['alpha'])
        objective = .5 * float(residual @ residual)
    else:
        L = np.asarray(s['L'], float)
        A = refine_reference(C, Y, L, lower[:,None], np.linalg.pinv(C) @ Y,
                             mu=s['mu'], alpha=s['alpha'])
        grad = C.T@(C@A-Y) + s['mu']*np.minimum(A,0.) + s['alpha']*(A@L)
        pg = np.max(np.abs(np.where((A <= lower[:,None]+1e-9) & (grad > 0), 0., grad)))
        objective = .5*(np.sum((C@A-Y)**2) + s['mu']*np.sum(np.minimum(A,0.)**2)
                        + s['alpha']*np.sum((A@L)*A))
    assert np.min(A-lower[:,None]) >= -1e-9
    assert pg < 1e-9
    assert np.isfinite(objective)
    assert 'weighted Newton' in row['error']
