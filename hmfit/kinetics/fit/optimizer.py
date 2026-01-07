"""Global optimization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from .objective import GlobalKineticsObjective


@dataclass(frozen=True)
class GlobalFitResult:
    params: dict[str, float]
    result: OptimizeResult
    ssq: float
    nfev: int
    njev: int | None


def fit_global(
    objective: GlobalKineticsObjective,
    params0: Mapping[str, float] | Sequence[float] | np.ndarray,
    *,
    bounds: tuple[float, float] | tuple[Sequence[float], Sequence[float]] | Mapping[
        str, tuple[float, float]
    ]
    | None = None,
    method: str = "trf",
    loss: str = "linear",
    **kwargs: object,
) -> GlobalFitResult:
    x0, param_names = _coerce_params0(objective, params0)
    bounds = _coerce_bounds(bounds, param_names)

    result = least_squares(
        objective.residuals,
        x0=x0,
        bounds=bounds,
        method=method,
        loss=loss,
        **kwargs,
    )

    params = {name: float(value) for name, value in zip(param_names, result.x, strict=True)}
    return GlobalFitResult(
        params=params,
        result=result,
        ssq=float(2.0 * result.cost),
        nfev=result.nfev,
        njev=result.njev,
    )


def _coerce_params0(
    objective: GlobalKineticsObjective,
    params0: Mapping[str, float] | Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    if isinstance(params0, Mapping):
        if objective.param_names is None:
            objective.param_names = list(params0.keys())
        param_names = list(objective.param_names)
        x0 = np.array([params0[name] for name in param_names], dtype=float)
        return x0, param_names

    if objective.param_names is None:
        raise ValueError("Objective param_names required when params0 is not a dict.")

    x0 = np.asarray(params0, dtype=float).reshape(-1)
    param_names = list(objective.param_names)
    if x0.shape[0] != len(param_names):
        raise ValueError("params0 length does not match param_names.")
    return x0, param_names


def _coerce_bounds(
    bounds: (
        tuple[float, float]
        | tuple[Sequence[float], Sequence[float]]
        | Mapping[str, tuple[float, float]]
        | None
    ),
    param_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
    if bounds is None:
        return (-np.inf, np.inf)

    if isinstance(bounds, Mapping):
        lower = []
        upper = []
        for name in param_names:
            if name not in bounds:
                raise KeyError(f"Missing bounds for '{name}'.")
            lo, hi = bounds[name]
            lower.append(lo)
            upper.append(hi)
        return (np.array(lower, dtype=float), np.array(upper, dtype=float))

    return bounds
