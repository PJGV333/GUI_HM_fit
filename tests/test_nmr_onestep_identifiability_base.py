"""NMR one-step bootstrap: identifiability is decided with the Jacobian of the *base fit*
(metrics at the observed data), never with the Jacobian of the last bootstrap replicate."""
from __future__ import annotations

import numpy as np

import hmfit_core.utils.errors as E


def _run(monkeypatch, J_base, J_rep):
    calls = {"n": 0}
    n_obs = J_base.shape[1]
    rng = np.random.default_rng(0)

    def fake_metrics(k_hat, res, dq, *args, **kwargs):
        calls["n"] += 1
        J = J_base if calls["n"] == 1 else J_rep
        return {"J": J, "r": rng.normal(0.0, 1e-3, n_obs), "n_linear": 0}

    monkeypatch.setattr(E, "compute_errors_nmr_varpro", fake_metrics)
    dq = np.ones((n_obs // 2, 2))
    out = E.bootstrap_one_step_nmr(np.array([3.0, 2.0]), None, dq, dq + 1e-3, None, None, [], B=7, seed=1)
    assert calls["n"] == 1 + 7
    return out["samples"]


def _J(identifiable_second: bool, n_obs=20):
    rng = np.random.default_rng(5)
    J = rng.normal(size=(2, n_obs))
    if not identifiable_second:
        J[1] = 0.0
    return J


def test_last_replicate_rank_loss_does_not_flag_parameter(monkeypatch):
    samples = _run(monkeypatch, J_base=_J(True), J_rep=_J(False))
    assert np.all(np.isfinite(samples[:, 1]))


def test_base_fit_rank_loss_flags_parameter_even_if_replicates_are_fine(monkeypatch):
    samples = _run(monkeypatch, J_base=_J(False), J_rep=_J(True))
    assert np.all(np.isnan(samples[:, 1])) and np.all(np.isfinite(samples[:, 0]))
