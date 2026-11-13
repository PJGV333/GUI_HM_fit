"""Small, experimental implicit sensitivities for chemical equilibria."""
from __future__ import annotations

import numpy as np


def association_11_logk_sensitivity(total, free, log10_k):
    """Differentiate H + G ⇌ HG at a supplied, balanced equilibrium.

    ``total`` and ``free`` are length-two vectors in the same concentration
    units, ordered H then G; log10_k is log10(K), where K has inverse
    concentration units. Return d[H, G, HG]/dlog10(K) and diagnostics.
    The totals and all other physical quantities are held fixed. This is an
    experimental validation utility, not a solver or a production fit path.
    """
    total = np.asarray(total, dtype=float)
    free = np.asarray(free, dtype=float)
    if total.shape != (2,) or free.shape != (2,):
        raise ValueError("total and free concentrations must be length-two vectors")
    if not np.all(np.isfinite(total)) or not np.all(np.isfinite(free)):
        raise ValueError("concentrations must be finite")
    if np.any(total < 0) or np.any(free < 0):
        raise ValueError("concentrations must be nonnegative")
    log10_k = float(log10_k)
    if not np.isfinite(log10_k):
        raise ValueError("log10_k must be finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        K = np.power(10.0, log10_k)
        bound = K * free[0] * free[1]
    if not np.isfinite(K) or K <= 0 or not np.isfinite(bound):
        raise ValueError("equilibrium constant or concentration is outside numerical range")
    balances = free + bound - total
    scale = np.maximum(total, np.finfo(float).tiny)
    balance_error = float(np.max(np.abs(balances) / scale))
    if not np.isfinite(balance_error) or balance_error > 1e-9:
        raise ValueError("supplied concentrations do not satisfy the 1:1 mass balance")

    h, g = free
    jac = np.array([[1 + K*g, K*h], [K*g, 1 + K*h]], dtype=float)
    condition = float(np.linalg.cond(jac))
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError("ill-conditioned equilibrium sensitivity system")
    d_free = np.linalg.solve(jac, -np.log(10.0) * bound * np.ones(2))
    # Differentiate H_total = H_free + HG; avoids cancellation in the
    # equivalent derivative of K*H_free*G_free for tight binding.
    d_bound = -d_free[0]
    derivative = np.array([d_free[0], d_free[1], d_bound])
    if not np.all(np.isfinite(derivative)):
        raise ValueError("nonfinite equilibrium sensitivity")
    return derivative, {"condition_number": condition, "relative_balance_error": balance_error}
