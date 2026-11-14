"""Independent conditional covariance checks on affected literature fits."""
import numpy as np
import pytest

from hmfit_core.api import run_nmr_fit
from hmfit_core.processors import nmr_processor
from hmfit_core.solvers import NewtonRaphson
from validation import literature_benchmark as lb


CASES = ['r09a-p2-058', 'r09a-p2-106', 'r09a-r1-3sp', 'r09a-r1-5sp', 'r09b-r-qx']


@pytest.mark.parametrize('cid', CASES)
def test_conditional_covariance_against_numerical_joint_design(cid, tmp_path, monkeypatch):
    case = lb.load_case(cid)
    conc, shifts = lb.numeric_table(case)
    cfg = lb.gui_config(case, str(tmp_path / 'case.xlsx'), conc, shifts)
    cfg['weights_mode'] = 'irls'
    captured = []
    original = nmr_processor.compute_errors_nmr_varpro

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append((kwargs, result))
        return result

    monkeypatch.setattr(nmr_processor, 'compute_errors_nmr_varpro', capture)
    monkeypatch.setattr(nmr_processor, 'generate_figure_base64', lambda *a, **k: '')
    monkeypatch.setattr(nmr_processor, 'generate_figure2_base64', lambda *a, **k: '')
    out = run_nmr_fit(cfg, progress_cb=lambda _: None)
    assert out['success'], out
    ed = out['export_data']
    kwargs, metrics = captured[-1]
    k = np.asarray(ed['k'])
    fixed = np.asarray(ed['fixed_mask'], dtype=bool)
    free = np.flatnonzero(~fixed)
    np.testing.assert_allclose(k, [case['_baseline']['log_beta'][s['name']] for s in case['species']], atol=1e-4, rtol=0)
    weights = np.asarray(ed['weights_used_for_fit'])
    np.testing.assert_array_equal(kwargs['weights'], weights)
    model = np.asarray(ed['modelo']).T
    solver = NewtonRaphson(conc, model, [], 'Free')
    stoich = np.asarray(ed['stoichiometry_map'])
    coef = metrics['coef']

    def design(theta, j):
        C = np.asarray(solver.concentraciones(theta)[0])
        den = C @ stoich[:, j]
        return (C * stoich[:, j]) / den[:, None]

    jac_blocks, residuals = [], []
    n_linear = 0
    for j in range(shifts.shape[1]):
        C = np.asarray(solver.concentraciones(k)[0])
        den = C @ stoich[:, j]
        valid = np.isfinite(shifts[:, j]) & np.isfinite(den) & (np.abs(den) > 1e-12)
        X = design(k, j)[valid]
        n_linear += np.linalg.matrix_rank(X, tol=1e-10 * np.linalg.svd(X, compute_uv=False)[0])
        beta = coef[:, j]
        raw = X @ beta - shifts[valid, j]
        residuals.append(np.sqrt(weights[j]) * raw)
        D = []
        for idx in free:
            kp, km = k.copy(), k.copy()
            kp[idx] += 1e-5
            km[idx] -= 1e-5
            D.append(((design(kp, j) - design(km, j))[valid] @ beta) / 2e-5)
        D = np.column_stack(D)
        # Eliminate the nuisance linear coefficients from the joint design.
        D -= X @ np.linalg.lstsq(X, D, rcond=1e-10)[0]
        jac_blocks.append(np.sqrt(weights[j]) * D)
    J = np.vstack(jac_blocks)
    r = np.concatenate(residuals)
    dof = len(r) - n_linear - len(free)
    assert metrics['dof'] == dof > 0
    assert metrics['nobs'] == len(r)
    np.testing.assert_allclose(metrics['covfit'], r @ r / dof, rtol=1e-7)
    cov = (r @ r / dof) * np.linalg.inv(J.T @ J)
    np.testing.assert_allclose(metrics['Cov_log10K_free'], cov, rtol=2e-3, atol=1e-9)
