"""Experimental direct sensitivities for a first-order kinetic model."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def solve_first_order_sensitivity(times, initial, rate, *, initial_sensitivity=None,
                                  method="BDF", rtol=1e-8, atol=1e-12):
    """Integrate A -> B and d[A,B]/dk for physical rate k, using SciPy.

    ``initial`` gives concentrations [A, B] at times[0]. ``rate`` has inverse
    time units. ``initial_sensitivity`` gives their derivatives with respect to
    k and defaults to zero. Results are arrays of shape (n_times, 2), followed
    by solver diagnostics. This experimental model is not used by fitting.
    """
    t = np.asarray(times, dtype=float)
    y0 = np.asarray(initial, dtype=float)
    s0 = np.zeros(2) if initial_sensitivity is None else np.asarray(initial_sensitivity, dtype=float)
    if t.ndim != 1 or t.size < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("time grid must have at least two finite, strictly increasing entries")
    if y0.shape != (2,) or not np.all(np.isfinite(y0)) or np.any(y0 < 0):
        raise ValueError("initial concentrations must be finite nonnegative [A, B]")
    if s0.shape != (2,) or not np.all(np.isfinite(s0)):
        raise ValueError("initial_sensitivity must be a finite length-two vector")
    rate = float(rate)
    if not np.isfinite(rate) or rate < 0:
        raise ValueError("rate must be finite and nonnegative")
    rtol, atol = float(rtol), float(atol)
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol <= 0 or atol <= 0:
        raise ValueError("integration tolerances must be finite and positive")

    def rhs(_t, state):
        a, _b, sa, _sb = state
        flux = rate * a
        dflux = a + rate * sa
        return [-flux, flux, -dflux, dflux]

    sol = solve_ivp(rhs, (float(t[0]), float(t[-1])), np.concatenate((y0, s0)),
                    t_eval=t, method=method, rtol=rtol, atol=atol)
    if not sol.success or sol.y.shape != (4, t.size) or not np.all(np.isfinite(sol.y)):
        raise RuntimeError(f"sensitivity integration failed: {sol.message}")
    return sol.y[:2].T, sol.y[2:].T, {
        "nfev": int(sol.nfev), "njev": int(sol.njev), "method": method,
        "rtol": rtol, "atol": atol,
    }
