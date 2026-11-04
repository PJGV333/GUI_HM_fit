"""Stage H: sparse-vs-dense equivalence and benchmark for the smoothed
weighted penalized spectral solve (validation layer; production untouched).

Replicates the semismooth-Newton solve in
hmfit_core/utils/nnls_utils._solve_weighted_penalty_newton using
scipy.sparse (block data Hessian + alpha*kron(I_s, L)) and compares, per
case: coefficient max-abs difference, objective difference, projected
first-order (KKT) residual for both paths, and wall time.

Run: QT_QPA_PLATFORM=offscreen .venv/bin/python -m validation.sparse_weighted_probe
"""
from __future__ import annotations

import json
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from hmfit_core.processors.spectroscopy_processor import _build_smoothness_laplacian
from hmfit_core.utils.nnls_utils import _solve_weighted_penalty_newton


def _sparse_h0(C, W2, alpha, s, nw):
    rows, cols, vals = [], [], []
    # data term: s x s block at each wavelength j (layout species-major)
    for j in range(nw):
        Bj = C.T @ (W2[:, j][:, None] * C)
        idx = np.arange(s) * nw + j
        for a in range(s):
            for b in range(s):
                rows.append(idx[a]); cols.append(idx[b]); vals.append(Bj[a, b])
    H = sp.csr_matrix((vals, (rows, cols)), shape=(s * nw, s * nw))
    if alpha > 0:
        L = _build_smoothness_laplacian(nw)
        Ls = sp.csr_matrix(L)
        H = H + float(alpha) * sp.kron(sp.identity(s, format="csr"), Ls, format="csr")
    return H


def _solve_sparse(C, Y, W, alpha, mu, lower_bound, max_iters=300, tol=1e-10):
    s, nw = C.shape[1], Y.shape[1]
    W2 = W * W
    size = s * nw
    H0 = _sparse_h0(C, W2, alpha, s, nw)
    b = (C.T @ (W2 * Y)).ravel()
    lower = np.full(size, -np.inf)
    if lower_bound is not None:
        lb = np.asarray(lower_bound, dtype=float)
        if lb.ndim == 1 and lb.size == s:
            lb = lb[:, None]
        lower = np.broadcast_to(lb, (s, nw)).ravel()
    x = np.maximum(np.zeros(size), lower)

    def objective(z):
        A = z.reshape(s, nw)
        return (0.5 * np.sum((W * (C @ A - Y)) ** 2)
                + 0.5 * float(mu) * np.dot(np.minimum(z, 0), np.minimum(z, 0))
                + (0.5 * float(alpha) * np.sum((A @ np.asarray(_build_smoothness_laplacian(nw))) * A)
                   if alpha > 0 else 0.0))

    Hs = H0  # sparse matvec for grad (no dense materialization)
    for _ in range(int(max_iters)):
        grad = Hs.dot(x) - b + float(mu) * np.minimum(x, 0)
        active = (x <= lower + 1e-12) & (grad > 0)
        pg = np.where(active, 0.0, grad)
        if np.max(np.abs(pg), initial=0.0) <= tol * max(1.0, np.max(np.abs(b), initial=0.0)):
            return x.reshape(s, nw), objective(x), float(np.max(np.abs(pg)))
        free = np.flatnonzero(~active)
        Hf = H0[np.ix_(free, free)].tocsc()
        d = Hf.diagonal()
        Hf = (Hf + sp.diags(float(mu) * (x[free] < 0) + 1e-20 * np.ones(len(free))))
        try:
            lu = spla.splu(Hf)
            step_dir_free = lu.solve(-grad[free])
        except Exception:
            step_dir_free = np.linalg.solve(Hf.toarray() + np.eye(len(free)) * 1e-12 * max(1.0, np.abs(d).max()), -grad[free])
        direction = np.zeros(size)
        direction[free] = step_dir_free
        value = objective(x)
        gdot = np.dot(grad, direction)
        st = 1.0
        for _ in range(50):
            cand = np.maximum(x + st * direction, lower)
            if objective(cand) <= value + 1e-4 * st * gdot:
                x = cand
                break
            st *= 0.5
        else:
            raise RuntimeError("sparse line search failed")
    raise RuntimeError("sparse Newton did not converge")


def run_case(s, nw, alpha, mu, seed=0, per_point=True):
    rng = np.random.default_rng(seed)
    C = rng.uniform(1e-4, 2e-3, size=(max(4 * s, 12), s))
    Y = rng.uniform(-1e-4, 3e-3, size=(C.shape[0], nw))
    W = (rng.uniform(0.5, 1.5, size=(C.shape[0], nw)) if per_point
         else np.ones((C.shape[0], nw)))
    t0 = time.perf_counter()
    try:
        Ad = _solve_weighted_penalty_newton(C, Y, W, 0.0, mu, None, alpha,
                                            _build_smoothness_laplacian(nw), 300)
        dense_ok, td = True, time.perf_counter() - t0
    except Exception as exc:
        Ad, dense_ok, td = None, False, time.perf_counter() - t0
        dense_err = str(exc)
    t1 = time.perf_counter()
    As, f_s, kkt_s = _solve_sparse(C, Y, W, alpha, mu, None, 300)
    ts = time.perf_counter() - t1
    out = {"s": s, "nw": nw, "alpha": alpha, "mu": mu, "size": s * nw,
           "sparse_s": round(ts, 3), "dense_ok": dense_ok,
           "dense_s": round(td, 3), "kkt_sparse": kkt_s, "obj_sparse": f_s}
    # shared objective + KKT evaluation for both paths
    def _obj(z):
        A = z.reshape(s, nw)
        return (0.5 * np.sum((W * (C @ A - Y)) ** 2)
                + 0.5 * float(mu) * np.dot(np.minimum(z, 0), np.minimum(z, 0))
                + (0.5 * float(alpha) * np.sum((A @ _build_smoothness_laplacian(nw)) * A)
                   if alpha > 0 else 0.0))
    H0 = _sparse_h0(C, W * W, alpha, s, nw)
    b = (C.T @ ((W * W) * Y)).ravel()

    def _kkt(z):
        g = H0.dot(z) - b + float(mu) * np.minimum(z, 0)
        return float(np.max(np.abs(g)))     # lower = -inf here: no bound activity
    out["kkt_shared_sparse"] = _kkt(As.ravel())
    out["obj_shared_sparse"] = _obj(As.ravel())
    if dense_ok:
        out["max_abs_diff"] = float(np.max(np.abs(Ad - As)))
        out["kkt_shared_dense"] = _kkt(Ad.ravel())
        out["obj_shared_dense"] = _obj(Ad.ravel())
        out["obj_abs_gap"] = abs(out["obj_shared_sparse"] - out["obj_shared_dense"])
    else:
        out["max_abs_diff"] = None
        out["dense_error"] = dense_err
    return out


def main() -> None:
    cases = []
    for (s, nw, alpha, mu) in [
        (3, 12, 0.0, 1e-2), (3, 12, 1e-6, 1e-2), (3, 30, 1e-6, 1e-2),
        (3, 30, 1e-5, 0.0), (3, 12, 1e-6, 5e-2),          # mu strongly active
        (2, 40, 1e-7, 1e-3),
        (3, 200, 1e-6, 1e-2), (3, 800, 1e-6, 1e-2), (3, 1400, 1e-6, 1e-2),
    ]:
        cases.append(run_case(s, nw, alpha, mu, seed=1))
    print(json.dumps(cases, indent=1))


if __name__ == "__main__":
    main()
