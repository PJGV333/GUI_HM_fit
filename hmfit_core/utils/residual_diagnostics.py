# SPDX-License-Identifier: GPL-3.0-or-later
"""
Residual-structure diagnostics shared by the kinetics and acid-base modules.

Every uncertainty estimate HM fit reports (analytic covariance and all the bootstraps) assumes
that the errors of consecutive points are independent. A kinetic trace or a titration curve
is an ordered series: when the model leaves systematic structure unexplained (wrong
mechanism, concentration error, drift, electrode calibration) or the noise itself is
correlated, consecutive residuals share their sign and all those estimates become too small,
typically by a factor of two or three, without any other symptom. The lag-1 autocorrelation
of the residuals along the ordering variable is the simplest quantitative symptom.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

# Below this many ordered points the statistic is too noisy to be interpreted.
MIN_POINTS = 8
# r1 must exceed both the absolute level and z standard errors (1/sqrt(n) under independence).
WARN_R1 = 0.30
STRONG_R1 = 0.50
Z_THRESHOLD = 2.0


def _as_blocks(residuals: Any) -> tuple[list[np.ndarray], bool]:
    """Residual blocks and whether the input was a list of NumPy arrays (one per dataset)."""
    is_list = (
        isinstance(residuals, (list, tuple))
        and len(residuals) > 0
        and all(isinstance(block, np.ndarray) for block in residuals)
    )
    blocks = [np.asarray(b, dtype=float) for b in residuals] if is_list else [np.asarray(residuals, dtype=float)]
    out = []
    for block in blocks:
        if block.ndim == 1:
            block = block.reshape(-1, 1)
        if block.ndim != 2:
            raise ValueError("Residual blocks must be 1-D or 2-D (rows = ordered points).")
        out.append(block)
    return out, is_list


def lag1_autocorrelation(
    residuals: Any,
    *,
    order: Sequence[float] | Iterable[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """
    Pooled lag-1 autocorrelation of residuals along their ordering variable.

    residuals : (n,) or (n, channels) array-like, or a list of NumPy arrays (one per dataset).
                Rows are ordered points (time, titrant volume, pH); non-finite entries are
                ignored pairwise.
    order     : optional ordering variable (one vector, or one per block); rows are sorted by
                it. Without it the rows are assumed to be already ordered.

    Returns {"r1", "n_points", "n_pairs", "z", "level", "inflation"} where

    * r1 = sum_j sum_i r[i, j] r[i+1, j] / sum_j sum_i r[i, j]^2 (no centring: a non-zero mean
      residual is itself lack of fit);
    * z = r1 * sqrt(n_points) with n_points the largest number of ordered points of a block.
      Channels of one spectrum are treated as fully dependent, so the test is conservative
      (it never gains significance from the number of wavelengths);
    * level is "insufficient" (fewer than MIN_POINTS points), "ok", "warn" or "strong";
    * inflation = sqrt((1 + r1) / (1 - r1)) for r1 > 0, the factor by which the standard error
      of a mean of AR(1) errors is underestimated. It is indicative only and, because the fit
      absorbs part of the slow structure, usually a lower bound.
    """
    blocks, is_list = _as_blocks(residuals)
    orders: list[np.ndarray | None]
    if order is None:
        orders = [None] * len(blocks)
    elif not is_list:
        orders = [np.asarray(order, dtype=float).reshape(-1)]
    else:
        orders = [None if o is None else np.asarray(o, dtype=float).reshape(-1) for o in order]  # type: ignore[union-attr]
        if len(orders) != len(blocks):
            raise ValueError("order must provide one vector per residual block.")

    num = 0.0
    den = 0.0
    n_pairs = 0
    n_points = 0
    for block, ordering in zip(blocks, orders):
        if ordering is not None:
            if ordering.shape[0] != block.shape[0]:
                raise ValueError("order length does not match the number of residual rows.")
            block = block[np.argsort(ordering, kind="stable"), :]
        finite = np.isfinite(block)
        rows_used = int(np.count_nonzero(np.any(finite, axis=1)))
        n_points = max(n_points, rows_used)
        clean = np.where(finite, block, 0.0)
        pair_ok = finite[:-1, :] & finite[1:, :]
        num += float(np.sum(clean[:-1, :] * clean[1:, :] * pair_ok))
        den += float(np.sum(clean * clean))
        n_pairs += int(np.count_nonzero(pair_ok))

    if n_points < MIN_POINTS or n_pairs == 0 or not np.isfinite(den) or den <= 0.0:
        return {"r1": float("nan"), "n_points": int(n_points), "n_pairs": int(n_pairs),
                "z": float("nan"), "level": "insufficient", "inflation": float("nan")}
    r1 = float(np.clip(num / den, -1.0, 1.0))
    z = r1 * float(np.sqrt(n_points))
    level = "ok"
    if r1 >= WARN_R1 and z >= Z_THRESHOLD:
        level = "strong" if r1 >= STRONG_R1 else "warn"
    inflation = float(np.sqrt((1.0 + r1) / (1.0 - r1))) if 0.0 < r1 < 1.0 else float("nan")
    return {"r1": r1, "n_points": int(n_points), "n_pairs": int(n_pairs), "z": float(z),
            "level": level, "inflation": inflation}


def serial_correlation_note(diag: dict[str, Any], axis_label: str = "the measurement order") -> str:
    """One readable warning for a ``lag1_autocorrelation`` result ("" when nothing to say)."""
    level = str((diag or {}).get("level") or "")
    if level not in {"warn", "strong"}:
        return ""
    r1 = float(diag["r1"])
    inflation = float(diag.get("inflation", float("nan")))
    factor = f" (indicatively by a factor of {inflation:.1f} or more)" if np.isfinite(inflation) else ""
    strength = "strongly " if level == "strong" else ""
    return (
        f"Residuals are {strength}correlated along {axis_label} (lag-1 autocorrelation "
        f"r1 = {r1:.2f} over {int(diag['n_points'])} points; about 0 is expected when the model is "
        "adequate and the noise is independent). The model leaves systematic structure "
        "unexplained or the noise is correlated. The analytic and bootstrap uncertainties assume "
        f"independent errors and are too small in this situation{factor}: inspect the residual "
        "plot before reporting the standard errors."
    )


__all__ = ["lag1_autocorrelation", "serial_correlation_note", "MIN_POINTS", "WARN_R1", "STRONG_R1", "Z_THRESHOLD"]
