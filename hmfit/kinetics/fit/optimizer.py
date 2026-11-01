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
    fixed_mask: np.ndarray | None = None
    nonidentifiable: np.ndarray | None = None
    notes: tuple[str, ...] = ()


def resolve_fixed_parameters(
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    fixed_mask: Sequence[bool] | None = None,
    param_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate bounds and decide which parameters are fixed.

    Rules
    -----
    * lower < upper            -> free parameter.
    * lower == upper (finite)  -> fixed at that common value.
    * lower > upper            -> ValueError (never silently treated as fixed).
    * NaN in bounds or x0      -> ValueError.
    * fixed_mask[i] True       -> fixed at x0[i] exactly (bounds are not used to move it);
      if lower == upper as well, both must agree with x0 within round-off.
    * fixed_mask must be None/empty (= nothing fixed) or have one entry per parameter.

    Returns (x_full0, fixed) where x_full0 holds the values used for fixed parameters.
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    n = x0.size
    names = list(param_names) if param_names is not None else [f"p{i+1}" for i in range(n)]
    if lower.size != n or upper.size != n:
        raise ValueError("Bounds must have one entry per parameter.")
    if np.any(np.isnan(x0)):
        bad = [names[i] for i in np.where(np.isnan(x0))[0]]
        raise ValueError("Initial value is NaN for parameter(s): " + ", ".join(bad))
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        bad = [names[i] for i in np.where(np.isnan(lower) | np.isnan(upper))[0]]
        raise ValueError("Bounds contain NaN for parameter(s): " + ", ".join(bad))
    inverted = lower > upper
    if np.any(inverted):
        bad = [f"{names[i]} (lower={lower[i]:g} > upper={upper[i]:g})" for i in np.where(inverted)[0]]
        raise ValueError("Invalid bounds: " + "; ".join(bad))

    mask = np.zeros(n, dtype=bool)
    if fixed_mask is not None:
        fm = np.asarray(list(fixed_mask), dtype=bool).reshape(-1)
        if fm.size not in (0, n):
            raise ValueError(
                f"fixed_mask has {fm.size} entries but there are {n} parameters."
            )
        if fm.size == n:
            mask = fm.copy()

    equal_bounds = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
    x_full0 = x0.copy()
    # Equal finite bounds define the value; if also mask-fixed, x0 must agree with it.
    for i in np.where(equal_bounds)[0]:
        if mask[i] and not np.isclose(x0[i], lower[i], rtol=1e-12, atol=1e-12):
            raise ValueError(
                f"Parameter {names[i]} is fixed at {x0[i]:g} but its bounds force {lower[i]:g}."
            )
        x_full0[i] = lower[i]
    fixed = mask | equal_bounds
    return x_full0, fixed


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
    fixed_mask: Sequence[bool] | None = None,
    **kwargs: object,
) -> GlobalFitResult:
    """
    Global least-squares fit.

    Fixed parameters (``fixed_mask`` or finite ``lower == upper``) are removed from the
    optimisation vector (SciPy rejects lb == ub). They keep their defined value exactly and
    get zero variance *by definition* (``result.fixed_mask`` marks them as not estimated).
    Inverted bounds (lower > upper), NaN bounds and NaN initial values raise ValueError.
    """
    x0, param_names = _coerce_params0(objective, params0)
    lower, upper = _bounds_arrays(_coerce_bounds(bounds, param_names), x0.size)
    x_full0, fixed = resolve_fixed_parameters(
        x0, lower, upper, fixed_mask=fixed_mask, param_names=param_names
    )
    free = np.where(~fixed)[0]

    def _expand(x_free: np.ndarray) -> np.ndarray:
        x = x_full0.copy()
        x[free] = x_free
        return x

    if free.size == 0:
        fun = np.asarray(objective.residuals(x_full0), dtype=float)
        result = OptimizeResult(
            x=x_full0, fun=fun, cost=0.5 * float(fun @ fun), jac=np.zeros((fun.size, x0.size)),
            nfev=1, njev=0, success=True, status=1, message="All parameters fixed.",
        )
    else:
        x_start = x_full0[free]
        outside = (x_start < lower[free]) | (x_start > upper[free])
        if np.any(outside):
            names = [param_names[i] for i in free[outside]]
            raise ValueError(
                "Initial value outside its bounds for parameter(s): " + ", ".join(names)
            )
        result = least_squares(
            lambda xf: objective.residuals(_expand(xf)),
            x0=x_start,
            bounds=(lower[free], upper[free]),
            method=method,
            loss=loss,
            **kwargs,
        )
        jac_free = None if result.jac is None else np.asarray(result.jac, dtype=float)
        result.x = _expand(np.asarray(result.x, dtype=float))
        if jac_free is not None:
            jac_full = np.zeros((jac_free.shape[0], x0.size), dtype=float)
            jac_full[:, free] = jac_free
            result.jac = jac_full
        if loss != "linear" and result.fun is not None:
            # report the plain SSE, not the robust-loss cost
            result.cost = 0.5 * float(np.sum(np.asarray(result.fun) ** 2))
    result.fixed_mask = fixed

    params = objective._coerce_params(result.x)

    errors = None
    covariance = None
    correlations = None
    condition_number = None
    nonid_full = np.zeros(x0.size, dtype=bool)
    notes: list[str] = []
    if result.jac is not None:
        from hmfit_core.utils.errors import linearized_covariance

        J = np.asarray(result.jac, dtype=float)
        if J.size:
            n_data = result.fun.size if result.fun is not None else J.shape[0]
            m = J.shape[1]
            n_linear = _count_linear_parameters(objective, result.x)
            Jf = J[:, free]
            lin = linearized_covariance(Jf, 2.0 * result.cost, n_data, n_linear)
            dof = lin["dof"]
            sigma2 = lin["s2"]
            notes.extend(lin["notes"])
            # Fixed parameters: exactly zero (co)variance by definition, not estimated.
            covariance = np.zeros((m, m), dtype=float)
            if free.size:
                condition_number = _scaled_condition_number(Jf)
                covariance[np.ix_(free, free)] = lin["cov"]
                nonid_full[free] = lin["nonidentifiable"]
            if objective.log_params:
                scale = np.ones(m, dtype=float)
                for idx, name in enumerate(param_names):
                    if name in objective.log_params:
                        scale[idx] = np.log(10.0) * float(params[name])
                covariance = (scale[:, None] * covariance) * scale[None, :]
            diag = np.diag(covariance)
            errors = np.where(np.isfinite(diag), np.sqrt(np.clip(diag, 0.0, np.inf)), np.nan)
            denom = np.outer(errors, errors)
            with np.errstate(invalid="ignore"):
                correlations = np.divide(
                    covariance,
                    denom,
                    out=np.zeros_like(covariance),
                    where=(denom != 0) & np.isfinite(denom),
                )
            correlations[nonid_full, :] = np.nan
            correlations[:, nonid_full] = np.nan
            np.fill_diagonal(correlations, np.where(fixed | nonid_full, np.nan, 1.0))
            result.dof = dof
            result.n_linear = n_linear
            result.sigma2 = sigma2
    if np.any(fixed):
        notes.append(
            "Fixed (not estimated) parameters: "
            + ", ".join(param_names[i] for i in np.where(fixed)[0])
            + " — reported uncertainty is zero by definition."
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
        fixed_mask=fixed.copy(),
        nonidentifiable=nonid_full,
        notes=tuple(notes),
    )


def _bounds_arrays(bounds, n: int) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = bounds
    lower = np.broadcast_to(np.asarray(lo, dtype=float), (n,)).astype(float).copy()
    upper = np.broadcast_to(np.asarray(hi, dtype=float), (n,)).astype(float).copy()
    return lower, upper


def _scaled_condition_number(J: np.ndarray) -> float | None:
    """Condition number of J with unit-norm columns (independent of parameter units)."""
    try:
        norms = np.linalg.norm(J, axis=0)
        norms = np.where(norms > 0, norms, 1.0)
        sv = np.linalg.svd(J / norms[None, :], compute_uv=False)
        if sv.size == 0 or sv[-1] <= 0:
            return float("inf")
        return float((sv[0] / sv[-1]) ** 2)  # condition of the scaled J^T J
    except np.linalg.LinAlgError:
        return None


def _count_linear_parameters(objective: GlobalKineticsObjective, x: np.ndarray) -> int:
    """Linear observation coefficients (A matrices) estimated by variable projection."""
    total = 0
    try:
        for dataset in objective.datasets:
            C, A, _ = objective.predict_dataset(x, dataset)
            rank = int(np.linalg.matrix_rank(np.asarray(C, dtype=float)))
            total += rank * int(np.asarray(A).shape[1])
    except Exception:
        return 0
    return total


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
