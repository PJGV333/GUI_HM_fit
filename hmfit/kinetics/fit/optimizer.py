"""Global optimization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from ..observation.linear_matrix import effective_rank
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


def residual_scale(objective: GlobalKineticsObjective) -> float:
    """
    Characteristic magnitude of the (weighted) observed signal, used only to make the
    optimiser's termination tests independent of the units of the data.

    SciPy's ``gtol`` is an *absolute* tolerance on the gradient J^T r, which scales with the
    square of the signal. With signals of the order of 1e-3 (concentrations in M, weak
    absorbances) the default ``gtol = 1e-8`` could be met at the initial values or before the
    minimum, so the same experiment gave different "converged" parameters in different
    units. Dividing the residuals by this constant leaves the minimum, the parameters and
    (after scaling back) SSQ, Jacobian and covariance unchanged.
    """
    from .objective import _dataset_weights

    total = 0.0
    count = 0
    for dataset in objective.datasets:
        D = getattr(dataset, "D", None)
        if D is None:
            continue
        D = np.asarray(D, dtype=float)
        try:
            weights = _dataset_weights(dataset)
        except Exception:
            weights = None
        if weights is not None:
            D = D * weights
        finite = np.isfinite(D)
        total += float(np.sum(np.square(D[finite])))
        count += int(np.count_nonzero(finite))
    if count == 0:
        return 1.0
    scale = float(np.sqrt(total / count))
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0
    return scale


def scaled_least_squares(residual_fn, x0, *, scale: float = 1.0, jac=None, **kwargs) -> OptimizeResult:
    """
    ``scipy.optimize.least_squares`` on ``residual_fn(x) / scale``.

    The returned ``fun``, ``jac``, ``cost``, ``grad`` and ``optimality`` are expressed in the
    original units of ``residual_fn`` (as if no scaling had been applied); ``residual_scale``
    is stored on the result. ``jac`` may be a callable returning the Jacobian of
    ``residual_fn`` (it is scaled consistently) or any value accepted by SciPy.
    """
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    inv = 1.0 / scale

    def fun(x):
        return np.asarray(residual_fn(x), dtype=float) * inv

    if callable(jac):
        user_jac = jac

        def jac_scaled(x):
            return np.asarray(user_jac(x), dtype=float) * inv

        kwargs["jac"] = jac_scaled
    elif jac is not None:
        kwargs["jac"] = jac
    result = least_squares(fun, x0, **kwargs)
    if result.fun is not None:
        result.fun = np.asarray(result.fun, dtype=float) * scale
    if result.jac is not None:
        result.jac = np.asarray(result.jac, dtype=float) * scale
    result.cost = float(result.cost) * scale * scale
    if getattr(result, "grad", None) is not None:
        result.grad = np.asarray(result.grad, dtype=float) * scale * scale
    result.optimality_scaled = float(getattr(result, "optimality", np.nan))
    result.residual_scale = scale
    return result


def convergence_notes(result: OptimizeResult, x_start, x_end, param_names: Sequence[str]) -> list[str]:
    """
    Plain-language warnings about how the optimiser stopped (never raises).

    * status 0  -> the evaluation budget ran out: the values are not a minimum.
    * status < 0 -> the optimiser failed.
    * the parameters did not move from their initial values -> the residuals are insensitive
      to them there (typically a rate constant far from the time scale of the data); the
      "result" is the initial guess, not a fit.
    """
    notes: list[str] = []
    try:
        status = int(getattr(result, "status", 1))
    except Exception:
        status = 1
    message = str(getattr(result, "message", "") or "").strip()
    if status == 0:
        notes.append(
            "The optimiser stopped at the evaluation limit before converging: the reported "
            "parameters are NOT a minimum. Improve the initial estimates (or use Multi-start) "
            "and fit again."
            + (f" [{message}]" if message else "")
        )
    elif status < 0:
        notes.append("The optimiser reported a failure" + (f": {message}" if message else "."))
    try:
        a = np.asarray(x_start, dtype=float).reshape(-1)
        b = np.asarray(x_end, dtype=float).reshape(-1)
        if a.size and a.shape == b.shape:
            unmoved = np.isclose(a, b, rtol=1e-9, atol=0.0)
            if np.all(unmoved):
                names = ", ".join(str(n) for n in param_names) if param_names else "all parameters"
                notes.append(
                    f"The optimiser did not move from the initial values ({names}): the data are "
                    "insensitive to the parameters at that starting point, so these numbers are "
                    "the initial guess, not a fit. Start from rate constants of the order of "
                    "1/(time span of the data), tick Log10 for constants of unknown magnitude, "
                    "or use Multi-start."
                )
    except Exception:
        pass
    return notes


def exchange_equivalent_parameters(
    objective: GlobalKineticsObjective,
    x_opt: np.ndarray,
    param_names: Sequence[str],
    *,
    fixed_mask: Sequence[bool] | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ssq: float | None = None,
    rtol: float = 1e-3,
    min_relative_difference: float = 0.01,
    max_free: int = 8,
) -> list[dict[str, object]]:
    """
    Pairs of free parameters whose *values can be exchanged* without changing the fit.

    Consecutive first-order steps (A -> B -> C with free species profiles) have two exact
    solutions, (k1, k2) and (k2, k1): the data cannot tell which step is the fast one, and the
    optimiser returns whichever is closer to the starting point. One residual evaluation per
    pair is enough to detect it: if SSQ at the exchanged values equals SSQ at the optimum
    (within ``rtol``) the exchange is an equivalent solution.

    ``x_opt`` is the optimisation vector (log10 for ``objective.log_params``); the exchange is
    done on the physical values. Pairs whose exchanged values violate ``bounds`` are skipped:
    bounds express prior knowledge that already selects one solution.
    Returns a list of {"a", "b", "value_a", "value_b", "ssq_ratio"}; never raises.
    """
    names = [str(n) for n in param_names]
    x_opt = np.asarray(x_opt, dtype=float).reshape(-1)
    n = x_opt.size
    if n != len(names) or n < 2:
        return []
    fixed = np.zeros(n, dtype=bool)
    if fixed_mask is not None and len(fixed_mask) == n:
        fixed = np.asarray(list(fixed_mask), dtype=bool)
    free = [i for i in range(n) if not fixed[i]]
    if len(free) < 2 or len(free) > int(max_free):
        return []
    log_params = set(getattr(objective, "log_params", set()) or set())
    is_log = np.array([name in log_params for name in names], dtype=bool)
    physical = np.where(is_log, 10.0 ** x_opt, x_opt)
    lower = upper = None
    if bounds is not None:
        lower, upper = _bounds_arrays(bounds, n)
    try:
        if ssq is None:
            r0 = np.asarray(objective.residuals(x_opt), dtype=float)
            ssq = float(r0 @ r0)
    except Exception:
        return []
    if not np.isfinite(ssq) or ssq <= 0.0:
        return []
    found: list[dict[str, object]] = []
    for pos, i in enumerate(free):
        for j in free[pos + 1:]:
            vi, vj = float(physical[i]), float(physical[j])
            scale = max(abs(vi), abs(vj))
            if scale == 0.0 or abs(vi - vj) <= float(min_relative_difference) * scale:
                continue
            if (is_log[i] and vj <= 0.0) or (is_log[j] and vi <= 0.0):
                continue
            x_swap = x_opt.copy()
            x_swap[i] = np.log10(vj) if is_log[i] else vj
            x_swap[j] = np.log10(vi) if is_log[j] else vi
            if lower is not None and (
                x_swap[i] < lower[i] or x_swap[i] > upper[i] or x_swap[j] < lower[j] or x_swap[j] > upper[j]
            ):
                continue
            try:
                with np.errstate(all="ignore"):
                    r = np.asarray(objective.residuals(x_swap), dtype=float)
                ssq_swap = float(r @ r)
            except Exception:
                continue
            if np.isfinite(ssq_swap) and abs(ssq_swap - ssq) <= float(rtol) * ssq:
                found.append({"a": names[i], "b": names[j], "value_a": vi, "value_b": vj,
                              "ssq_ratio": ssq_swap / ssq})
    return found


def exchange_equivalence_notes(pairs: Sequence[Mapping[str, object]]) -> list[str]:
    """Readable warning for every pair reported by ``exchange_equivalent_parameters``."""
    notes = []
    for pair in pairs:
        notes.append(
            f"Two equivalent solutions: exchanging {pair['a']} = {float(pair['value_a']):.6g} and "
            f"{pair['b']} = {float(pair['value_b']):.6g} gives the same fit (SSQ ratio "
            f"{float(pair['ssq_ratio']):.6f}). The data alone cannot tell which of the two steps is "
            "the fast one (slow-fast ambiguity); the solutions differ only in the fitted profiles "
            "(A) of the intermediates. Decide with chemical knowledge, known spectra, NNLS or "
            "bounds, and report which solution was chosen."
        )
    return notes


def randomize_start(
    x0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
    fixed_mask: Sequence[bool] | None = None,
    log_mask: Sequence[bool] | None = None,
) -> np.ndarray:
    """
    Random starting point for multi-start, always inside the bounds.

    * both bounds finite: uniform between them, or log-uniform when the parameter is positive
      and spans more than one decade (a uniform draw on [1e-5, 1e3] practically never tries a
      small rate constant);
    * otherwise: one decade (1 sigma) around the initial value, multiplicative for positive
      parameters and additive in log10 space for log-fitted ones, then clipped to the bounds.
      The previous additive N(0, 0.2) step ignored the magnitude of the parameter: with the
      default table (min = 0, no max) it produced negative rate constants and the fit aborted.
    * fixed parameters are never moved.
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    n = x0.size
    lower, upper = _bounds_arrays(bounds, n)
    fixed = np.zeros(n, dtype=bool)
    if fixed_mask is not None and len(fixed_mask) == n:
        fixed = np.asarray(list(fixed_mask), dtype=bool)
    in_log = np.zeros(n, dtype=bool)
    if log_mask is not None and len(log_mask) == n:
        in_log = np.asarray(list(log_mask), dtype=bool)
    x = x0.copy()
    for idx in range(n):
        if fixed[idx]:
            continue
        lo, hi, start = float(lower[idx]), float(upper[idx]), float(x0[idx])
        if np.isfinite(lo) and np.isfinite(hi):
            if hi <= lo:
                continue
            if not in_log[idx] and lo > 0.0 and hi / lo > 10.0:
                value = 10.0 ** rng.uniform(np.log10(lo), np.log10(hi))
            else:
                value = rng.uniform(lo, hi)
        elif in_log[idx]:
            value = start + rng.normal(0.0, 1.0)
        elif start > 0.0 and lo >= 0.0:
            value = start * 10.0 ** rng.normal(0.0, 1.0)
        else:
            value = start + rng.normal(0.0, max(abs(start), 1.0))
        x[idx] = min(max(value, lo), hi)
    return x


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
    from .rate_parameters import _shared_rate_order_error

    order_error = _shared_rate_order_error(objective.model)
    if order_error:
        raise ValueError(order_error)
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
        result = scaled_least_squares(
            lambda xf: objective.residuals(_expand(xf)),
            x_start,
            scale=residual_scale(objective),
            bounds=(lower[free], upper[free]),
            method=method,
            loss=loss,
            **kwargs,
        )
        result.x_start = _expand(np.asarray(x_start, dtype=float))
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
    if free.size:
        x_begin = getattr(result, "x_start", None)
        if x_begin is not None:
            conv = convergence_notes(
                result, np.asarray(x_begin)[free], np.asarray(result.x)[free],
                [param_names[i] for i in free],
            )
            result.convergence_notes = tuple(conv)
            notes.extend(conv)
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
            rank = effective_rank(np.asarray(C, dtype=float))
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
