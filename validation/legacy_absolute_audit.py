"""Stage H: legacy-vs-absolute weight/regularization audit.

Documents and numerically verifies the conventions of the two
weight_regularization_mode paths in
hmfit_core.processors.spectroscopy_processor._solve_absorptivities:

- "legacy": historical PGD estimator; mu/alpha are *relative* regularizers
  that are never rescaled by the wavelength/point weights, and the hard-NNLS
  + smoothing incompatibility is preserved.
- "absolute": semismooth Newton on the explicit objective
  1/2 sum w^2 (CA-Y)^2 + 1/2 mu ||min(A,0)||^2 + 1/2 alpha <A, L A>,
  with mu/alpha in absolute units.

Predicted and tested equivalence: legacy at uniform weight w equals absolute
with mu_eff = mu / w^2 (multiplying the data term by w^2 rescales the whole
legacy objective). Cases:

1. W == 1: legacy == absolute exactly (same objective).
2. W == w uniform (w=3): legacy(mu) == absolute(mu/w^2).
3. per-wavelength weights: the w^2-rescaling argument holds per column only
   if smoothing couples columns; uncoupled columns each satisfy their own
   column-weight identity, coupled smoothing breaks it. Measured.
4. per-point (rows) weights: legacy passes them into the residual term only;
   absolute uses them in the full objective. Difference reported.
5. nnls_hard + smoothing: legacy runs (PGD), absolute raises
   NotImplementedError - documented API asymmetry.

Run: QT_QPA_PLATFORM=offscreen .venv/bin/python -m validation.legacy_absolute_audit
"""
from __future__ import annotations

import json

import numpy as np

from hmfit_core.processors.spectroscopy_processor import (
    _build_smoothness_laplacian,
    _solve_absorptivities,
)


def _problem(seed=2, s=3, nw=24):
    rng = np.random.default_rng(seed)
    C = rng.uniform(1e-4, 2e-3, size=(14, s))
    Y = rng.uniform(-1e-4, 3e-3, size=(C.shape[0], nw))
    return C, Y


def _weights_matrix(W, C, Y):
    # _solve_absorptivities expects weights_row either per-wavelength (nw,)
    # or a flattened (n_rows*nw,) per-point matrix packed like Y.T row-major.
    return W


def run_audit():
    C, Y = _problem()
    nw = Y.shape[1]
    mu = 1e-2
    out = []

    def solve(mode, W=None, alpha=0.0, mu_=mu, mode_norm="soft_penalty"):
        flat = None
        if W is not None and W.ndim == 2:
            flat = np.asarray(W, dtype=float)  # (m, nw) accepted directly
        A, _ = _solve_absorptivities(
            C, Y.copy(), eps_solver_mode=mode_norm, mu=mu_, alpha_smooth=alpha,
            smooth_matrix=_build_smoothness_laplacian(nw) if alpha > 0 else None,
            weights_row=flat if flat is not None else (W if W is not None else None),
            weight_regularization_mode=mode)
        return np.asarray(A, dtype=float)

    # case 1: unit weights (None) -> same objective
    a1 = solve("legacy")
    b1 = solve("absolute")
    out.append({"case": "unit_weights", "max_abs_diff": float(np.max(np.abs(a1 - b1)))})

    # case 2: uniform per-point weight w -> legacy(mu) == absolute(mu*w^2).
    # legacy IGNORES uniform weight scales (argmin of the unscaled objective);
    # absolute scales the data term by w^2, so matching it needs mu_eff = mu*w^2
    # (divide absolute's objective by w^2 to compare with legacy's).
    w = 3.0
    W2 = np.full((C.shape[0], nw), w)
    a2 = solve("legacy", W=W2)
    b2 = solve("absolute", W=W2, mu_=mu * w ** 2)
    out.append({"case": "uniform_w3_mu_times_w2", "max_abs_diff": float(np.max(np.abs(a2 - b2)))})

    # case 2b: same mu both modes -> quantifies the naive-mismatch magnitude
    b2b = solve("absolute", W=W2, mu_=mu)
    out.append({"case": "uniform_w3_same_mu", "max_abs_diff": float(np.max(np.abs(a2 - b2b)))})

    # case 3: per-wavelength weights (weights_row=(nw,)); legacy vs absolute
    wl = np.linspace(0.5, 2.0, nw)
    def solve_lam(mode, mu_):
        A, _ = _solve_absorptivities(
            C, Y.copy(), eps_solver_mode="soft_penalty", mu=mu_,
            weights_row=wl, weight_regularization_mode=mode)
        return np.asarray(A, dtype=float)
    a3 = solve_lam("legacy", mu)
    b3 = solve_lam("absolute", mu)  # absolute: same call; identity check
    out.append({"case": "per_lambda_same_mu", "max_abs_diff": float(np.max(np.abs(a3 - b3)))})

    # case 4: per-point weights, same mu both modes
    Wp = np.random.default_rng(3).uniform(0.5, 1.5, size=(C.shape[0], nw))
    a4 = solve("legacy", W=Wp)
    b4 = solve("absolute", W=Wp)
    out.append({"case": "per_point_same_mu", "max_abs_diff": float(np.max(np.abs(a4 - b4)))})

    # case 5: nnls_hard + smoothing asymmetry
    legacy_nnls_ok = True
    try:
        solve("legacy", alpha=1e-6, mode_norm="nnls_hard")
    except NotImplementedError:
        legacy_nnls_ok = False
    abs_nnls_raises = False
    try:
        solve("absolute", alpha=1e-6, mode_norm="nnls_hard")
    except NotImplementedError:
        abs_nnls_raises = True
    out.append({"case": "nnls_hard_smoothing",
                "legacy_runs": legacy_nnls_ok, "absolute_raises_NotImplemented": abs_nnls_raises})

    return out


def main() -> None:
    print(json.dumps(run_audit(), indent=1))


if __name__ == "__main__":
    main()
