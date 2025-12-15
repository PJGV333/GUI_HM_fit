"""
Helpers for the "Non-cooperative" / statistical 1:N (or N:1) model.

The model is valid only for 2 components and a pure sequential series:
  - 1:N: (1,1), (1,2), ..., (1,N)
  - N:1: (1,1), (2,1), ..., (N,1)

This module infers N and the stage index j for each complex column so the
formation constants (beta) can be assigned robustly even when complex columns
are not ordered as HG, HG2, ...
"""

from __future__ import annotations

import numpy as onp
from typing import Iterable, Tuple


def _as_int_stoich(x: float) -> int:
    xi = int(round(float(x)))
    if abs(float(x) - xi) > 1e-9:
        raise ValueError("Non-cooperative requiere estequiometrías enteras (1..N).")
    return xi


def infer_noncoop_series(modelo: onp.ndarray) -> Tuple[int, onp.ndarray, onp.ndarray]:
    """
    Infer N and the stage index j for each complex column.

    Args:
        modelo: (n_comp, nspec) stoichiometry matrix.

    Returns:
        N: max stage in the 1:N or N:1 series (also equals n_complex).
        j_per_complex_col: (n_complex,) array with j in [1..N] for each complex column.
        complex_cols: (n_complex,) array with column indices in the original model.
    """
    M = onp.asarray(modelo, dtype=float)
    if M.ndim != 2:
        raise ValueError("modelo debe ser 2D.")
    n_comp, nspec = M.shape
    if n_comp != 2:
        raise ValueError("Non-cooperative requiere exactamente 2 componentes (Host/Guest).")

    complex_cols = onp.arange(n_comp, nspec, dtype=int)
    n_complex = int(nspec - n_comp)
    if n_complex <= 0:
        raise ValueError("Non-cooperative requiere al menos un complejo (1:N o N:1).")

    a = onp.array([_as_int_stoich(M[0, col]) for col in complex_cols], dtype=int)
    b = onp.array([_as_int_stoich(M[1, col]) for col in complex_cols], dtype=int)

    if onp.all(a == 1):
        stages = b
        orientation = "1:N"
    elif onp.all(b == 1):
        stages = a
        orientation = "N:1"
    else:
        raise ValueError(
            "Non-cooperative requiere serie secuencial 1:N o N:1 con 2 componentes "
            "(p.ej. (1,1),(1,2),... o (1,1),(2,1),...)."
        )

    if onp.any(stages <= 0):
        raise ValueError("Non-cooperative requiere coeficientes positivos en los complejos.")

    N = int(onp.max(stages))
    expected = set(range(1, N + 1))
    got = set(int(x) for x in stages.tolist())
    if got != expected or n_complex != N:
        raise ValueError(
            f"Non-cooperative ({orientation}) requiere complejos exactamente para j=1..N; "
            f"se esperaban {sorted(expected)}, llegaron {sorted(got)} (n_complex={n_complex})."
        )

    return N, stages, complex_cols


def noncoop_derived_from_logK1(modelo: onp.ndarray, logK1: float) -> dict:
    """
    Compute stepwise macroscopic log10(K_j) and cumulative log10(beta_j) for Non-cooperative models.

    Args:
        modelo: (2, nspec) stoichiometry matrix.
        logK1: log10 of the first macroscopic step constant K1.

    Returns:
        dict with:
          - N: number of stages
          - logK_by_j: (N,) log10(K_j) for j=1..N
          - K_by_j: (N,) K_j values (linear)
    """
    N, _, _ = infer_noncoop_series(modelo)
    js = onp.arange(1, N + 1, dtype=float)
    logK_by_j = float(logK1) + onp.log10((N - js + 1.0) / (js * float(N)))
    return {
        "N": int(N),
        "logK_by_j": onp.asarray(logK_by_j, dtype=float),
        "K_by_j": onp.power(10.0, logK_by_j),
    }
