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
    errors: np.ndarray | None = None
    covariance: np.ndarray | None = None
    correlations: np.ndarray | None = None
    condition_number: float | None = None


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

    params = objective._coerce_params(result.x)

    errors = None
    covariance = None
    correlations = None
    condition_number = None
    if result.jac is not None:
        J = np.asarray(result.jac, dtype=float)
        if J.size:
            n_data = result.fun.size if result.fun is not None else J.shape[0]
            m = J.shape[1]
            dof = max(n_data - m, 1)
            sigma2 = 2.0 * result.cost / dof
            jt_j = J.T @ J
            try:
                condition_number = float(np.linalg.cond(jt_j))
            except np.linalg.LinAlgError:
                condition_number = None
            covariance = sigma2 * np.linalg.pinv(jt_j)
            if objective.log_params:
                scale = np.ones(m, dtype=float)
                for idx, name in enumerate(param_names):
                    if name in objective.log_params:
                        scale[idx] = np.log(10.0) * float(params[name])
                covariance = (scale[:, None] * covariance) * scale[None, :]
            errors = np.sqrt(np.diag(covariance))
            denom = np.outer(errors, errors)
            correlations = np.divide(
                covariance,
                denom,
                out=np.zeros_like(covariance),
                where=denom != 0,
            )
    return GlobalFitResult(
        params=params,
        result=result,
        ssq=float(2.0 * result.cost),
        nfev=result.nfev,
        njev=result.njev,
        errors=errors,
        covariance=covariance,
        correlations=correlations,
        condition_number=condition_number,
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
