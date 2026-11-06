"""One-step bootstrap optimisation: the speciation at k_hat is solved once per analysis and
the result is bit-for-bit the same as recomputing it for every replicate."""
from __future__ import annotations

import numpy as np

import hmfit_core.utils.errors as E
from hmfit_core.solvers import NewtonRaphson


class _Counting:
    def __init__(self, solver):
        self.solver, self.calls = solver, 0

    def concentraciones(self, k):
        self.calls += 1
        return self.solver.concentraciones(k)


def _case():
    rng = np.random.default_rng(3)
    m, model, nas, k = 10, [[1, 0], [0, 1], [1, 1]], [], np.array([3.0])
    c = np.column_stack([np.full(m, 1e-3), np.linspace(0.0, 4e-3, m)])
    M = np.asarray(model, dtype=float).T
    solver = NewtonRaphson(c, M, nas=nas, model_sett="Free")
    C, _ = solver.concentraciones(k)
    E_true = np.abs(rng.normal(size=(C.shape[1], 12))) * 1e3
    Y = (C @ E_true).T + rng.normal(0.0, 1e-3, (12, m))
    return solver, M, nas, k, Y


def _run(monkeypatch, cached):
    solver, M, nas, k, Y = _case()
    counting = _Counting(solver)
    if not cached:
        monkeypatch.setattr(E, "_FixedPointConcentrations", lambda res: res)
    out = E.bootstrap_one_step_spectro(k, counting, Y, M, nas, B=25, seed=7)
    return np.asarray(out["samples"]), counting.calls


def test_cached_speciation_is_bitwise_identical(monkeypatch):
    with monkeypatch.context() as mp:
        uncached, calls_uncached = _run(mp, cached=False)
    cached, calls_cached = _run(monkeypatch, cached=True)
    assert np.array_equal(cached, uncached)
    assert calls_uncached == 1 + 25 and calls_cached == 1


def test_wrapper_returns_copies_and_tracks_k():
    solver, *_ = _case()
    counting = _Counting(solver)
    w = E._FixedPointConcentrations(counting)
    a = w.concentraciones(np.array([3.0]))
    a[0][:] = -1.0  # caller mutation must not reach the cache
    b = w.concentraciones(np.array([3.0]))
    assert counting.calls == 1 and np.all(b[0] >= 0)
    w.concentraciones(np.array([3.1]))
    assert counting.calls == 2
    assert E._FixedPointConcentrations(solver).nas is solver.nas  # attributes delegate to the solver


def test_jax_type_resolved_once(monkeypatch):
    E._JAX_ARRAY_TYPE = E._UNRESOLVED
    first = E._jax_array_type()
    import builtins

    real_import = builtins.__import__

    def no_more_imports(name, *a, **k):
        assert not name.startswith("jax"), "jax import attempted again"
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_more_imports)
    assert E._jax_array_type() is first
    assert np.array_equal(E._as_onp([1.0, 2.0]), np.array([1.0, 2.0]))
