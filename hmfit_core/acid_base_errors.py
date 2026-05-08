# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence
import re
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from hmfit_core.acid_base import (
    AcidBaseSystem,
    NMRAcidBaseDataset,
    SpectroscopicAcidBaseDataset,
    log_beta_to_pka,
    pka_to_log_beta,
    nmr_acid_base_residuals,
    spectroscopy_acid_base_residuals,
)
from hmfit_core.acid_base_model_utils import acid_base_system_from_model
from hmfit_core.potentiometry import PotentiometryExperiment, potentiometry_residuals
from hmfit_core.utils.errors import (
    bootstrap_full_refit,
    bootstrap_linearized_wild,
    build_identifiability_report,
    percent_metrics_from_log10K,
    wild_multipliers,
)

_PKA_RE = re.compile(r"^pka(?:[_\-\s]*|\[)?(\d+)\]?$", re.IGNORECASE)
_LOGBETA_RE = re.compile(
    r"^(?:log[_\-\s]*beta|logbeta|beta)(?:[_\-\s]*|\[)?(\d+)\]?$",
    re.IGNORECASE,
)


def _as_float_vector(values: Sequence[float] | np.ndarray | None, *, name: str) -> np.ndarray:
    arr = np.asarray([] if values is None else values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _normalize_fixed_mask(fixed_mask: Sequence[bool] | np.ndarray | None, size: int) -> np.ndarray:
    if fixed_mask is None:
        return np.zeros(size, dtype=bool)
    mask = np.asarray(fixed_mask, dtype=bool).reshape(-1)
    if mask.size != size:
        raise ValueError("fixed_mask length does not match parameter vector.")
    return mask


def _normalize_bounds(bounds: Any, size: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(size, -np.inf, dtype=float)
    upper = np.full(size, np.inf, dtype=float)
    if bounds is None:
        return lower, upper
    if isinstance(bounds, Mapping):
        bounds = bounds.get("pairs") or bounds.get("bounds") or []
    if isinstance(bounds, tuple) and len(bounds) == 2:
        left, right = bounds
        left_arr = np.asarray(left, dtype=float).reshape(-1)
        right_arr = np.asarray(right, dtype=float).reshape(-1)
        if left_arr.size == 1:
            left_arr = np.full(size, float(left_arr[0]), dtype=float)
        if right_arr.size == 1:
            right_arr = np.full(size, float(right_arr[0]), dtype=float)
        if left_arr.size == size:
            lower[:] = left_arr
        if right_arr.size == size:
            upper[:] = right_arr
        return lower, upper

    rows = list(bounds or [])
    for idx in range(min(size, len(rows))):
        row = rows[idx]
        if isinstance(row, Mapping):
            lo = row.get("min")
            hi = row.get("max")
        else:
            seq = list(row or [])
            lo = seq[0] if len(seq) >= 1 else None
            hi = seq[1] if len(seq) >= 2 else None
        if lo not in (None, ""):
            lower[idx] = float(lo)
        if hi not in (None, ""):
            upper[idx] = float(hi)
    return lower, upper


def _match_param_index(pattern: re.Pattern[str], name: str) -> int | None:
    match = pattern.match(str(name).strip())
    if not match:
        return None
    idx = int(match.group(1)) - 1
    return idx if idx >= 0 else None


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    if cov.size == 0:
        return np.empty((0, 0), dtype=float)
    diag = np.clip(np.diag(cov), 0.0, np.inf)
    scale = np.sqrt(diag)
    denom = np.outer(scale, scale)
    corr = np.zeros_like(cov, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.divide(cov, denom, where=denom > 0.0)
    corr = np.nan_to_num(corr, nan=0.0)
    corr = np.clip(corr, -1.0, 1.0)
    for idx in range(corr.shape[0]):
        corr[idx, idx] = 1.0
    return corr


def _lower_triangular_ones(n: int) -> np.ndarray:
    return np.tril(np.ones((n, n), dtype=float))


def _first_difference_matrix(n: int) -> np.ndarray:
    out = np.zeros((n, n), dtype=float)
    if n <= 0:
        return out
    out[0, 0] = 1.0
    for idx in range(1, n):
        out[idx, idx] = 1.0
        out[idx, idx - 1] = -1.0
    return out


def propagate_pka_to_log_beta(
    pka_values: Sequence[float] | np.ndarray,
    covariance_pka: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pka = np.asarray(pka_values, dtype=float).reshape(-1)
    cov = np.asarray(covariance_pka, dtype=float)
    if cov.shape != (pka.size, pka.size):
        raise ValueError("covariance_pka shape does not match pKa vector.")
    transform = _lower_triangular_ones(pka.size)
    log_beta = transform @ pka
    cov_log_beta = transform @ cov @ transform.T
    return log_beta, cov_log_beta, transform


def propagate_log_beta_to_pka(
    log_beta_values: Sequence[float] | np.ndarray,
    covariance_log_beta: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_beta = np.asarray(log_beta_values, dtype=float).reshape(-1)
    cov = np.asarray(covariance_log_beta, dtype=float)
    if cov.shape != (log_beta.size, log_beta.size):
        raise ValueError("covariance_log_beta shape does not match log_beta vector.")
    transform = _first_difference_matrix(log_beta.size)
    pka = transform @ log_beta
    cov_pka = transform @ cov @ transform.T
    return pka, cov_pka, transform


def _acid_base_constant_info(
    parameter_names: Sequence[str],
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    names = [str(name) for name in parameter_names]
    ctx = dict(context or {})
    explicit_indices = ctx.get("constant_parameter_indices")
    if explicit_indices is not None:
        indices = [int(idx) for idx in explicit_indices]
    else:
        pka_hits: list[tuple[int, int]] = []
        lb_hits: list[tuple[int, int]] = []
        for param_pos, name in enumerate(names):
            pka_idx = _match_param_index(_PKA_RE, name)
            if pka_idx is not None:
                pka_hits.append((param_pos, pka_idx))
                continue
            lb_idx = _match_param_index(_LOGBETA_RE, name)
            if lb_idx is not None:
                lb_hits.append((param_pos, lb_idx))
        primary_space = str(ctx.get("primary_constant_space") or "").strip().lower()
        hits = pka_hits
        if primary_space == "log_beta":
            hits = lb_hits
        elif not hits and lb_hits:
            hits = lb_hits
            primary_space = "log_beta"
        else:
            primary_space = "pka"
        hits = sorted(hits, key=lambda item: item[1])
        indices = [param_pos for param_pos, _ in hits]
    if not indices:
        count = int(ctx.get("n_acid_base_constants") or 0)
        indices = list(range(min(count, len(names))))
    primary_space = str(ctx.get("primary_constant_space") or "pka").strip().lower()
    if primary_space not in {"pka", "log_beta"}:
        primary_space = "pka"
    labels = [names[idx] for idx in indices]
    return {
        "indices": indices,
        "labels": labels,
        "primary_space": primary_space,
    }


def _finite_difference_jacobian(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    theta_hat: np.ndarray,
    *,
    fixed_mask: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    rel_step: float = 1e-6,
    abs_step: float = 1e-8,
) -> np.ndarray:
    theta = np.asarray(theta_hat, dtype=float).reshape(-1)
    base = np.asarray(residual_fn(theta), dtype=float).reshape(-1)
    nobs = base.size
    npar = theta.size
    lower, upper = bounds
    jac = np.zeros((nobs, npar), dtype=float)
    for idx in range(npar):
        if fixed_mask[idx]:
            continue
        step = max(abs(theta[idx]) * rel_step, abs_step)
        step_plus = min(step, max(upper[idx] - theta[idx], 0.0)) if np.isfinite(upper[idx]) else step
        step_minus = min(step, max(theta[idx] - lower[idx], 0.0)) if np.isfinite(lower[idx]) else step
        if step_plus > 0.0 and step_minus > 0.0:
            plus = theta.copy()
            minus = theta.copy()
            plus[idx] += step_plus
            minus[idx] -= step_minus
            r_plus = np.asarray(residual_fn(plus), dtype=float).reshape(-1)
            r_minus = np.asarray(residual_fn(minus), dtype=float).reshape(-1)
            jac[:, idx] = (r_plus - r_minus) / (step_plus + step_minus)
        elif step_plus > 0.0:
            plus = theta.copy()
            plus[idx] += step_plus
            r_plus = np.asarray(residual_fn(plus), dtype=float).reshape(-1)
            jac[:, idx] = (r_plus - base) / step_plus
        elif step_minus > 0.0:
            minus = theta.copy()
            minus[idx] -= step_minus
            r_minus = np.asarray(residual_fn(minus), dtype=float).reshape(-1)
            jac[:, idx] = (base - r_minus) / step_minus
        else:
            jac[:, idx] = 0.0
    return jac


def _parameter_percent_error(
    name: str,
    estimate: float,
    se: float,
) -> float | None:
    if not np.isfinite(estimate) or not np.isfinite(se):
        return None
    if _match_param_index(_LOGBETA_RE, name) is not None:
        return float(100.0 * np.log(10.0) * se)
    return None


def _linear_metrics_from_log_beta(values: np.ndarray, se_values: np.ndarray) -> dict[str, np.ndarray]:
    metrics = percent_metrics_from_log10K(values, se_values)
    return {
        "linear_value": np.asarray(metrics["K"], dtype=float),
        "se_linear": np.asarray(metrics["SE_K"], dtype=float),
        "percent_error": np.asarray(metrics["perc_linear"], dtype=float),
    }


def _linear_metrics_from_pka(values: np.ndarray, se_values: np.ndarray) -> dict[str, np.ndarray]:
    ka = np.power(10.0, -values)
    rel = np.log(10.0) * se_values
    return {
        "linear_value": ka,
        "se_linear": ka * rel,
        "percent_error": 100.0 * rel,
    }


def _all_fixed_result(parameter_names: Sequence[str], theta_hat: np.ndarray, residuals: np.ndarray) -> dict[str, Any]:
    p = theta_hat.size
    warnings.warn("All parameters are fixed; returning zero uncertainty estimates.", UserWarning)
    stability_diag = {
        "status": "warn",
        "cond_jjt": 1.0,
        "diag_summary": "All parameters are fixed; uncertainty estimates are zero.",
        "diag_full": "All parameters are fixed; covariance and correlation are not informative.",
        "max_abs_corr": 0.0,
        "rank_eff": 0,
        "p_free": 0,
        "high_corr_pairs": [],
        "stability_indicator": {
            "status": "warn",
            "label": "All fixed",
            "icon": "⚠️",
            "cond": 1.0,
            "max_abs_corr": 0.0,
            "rank_eff": 0,
            "p_free": 0,
            "reasons": ["all fixed"],
        },
    }
    return {
        "theta_hat": theta_hat,
        "param_names": list(parameter_names),
        "residuals": residuals,
        "jacobian": np.zeros((residuals.size, p), dtype=float),
        "covariance_matrix": np.zeros((p, p), dtype=float),
        "correlation_matrix": np.eye(p, dtype=float),
        "se": np.zeros(p, dtype=float),
        "nobs": int(residuals.size),
        "p_free": 0,
        "dof": int(residuals.size),
        "sse": float(np.sum(residuals**2)),
        "s2": float(np.sum(residuals**2) / max(int(residuals.size), 1)),
        "rms": float(np.sqrt(np.mean(residuals**2))) if residuals.size else 0.0,
        "stability_diag": stability_diag,
        "samples": None,
        "bootstrap_summary": None,
    }


def compute_errors_acid_base(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    theta_hat: Sequence[float] | np.ndarray,
    parameter_names: Sequence[str],
    fixed_mask: Sequence[bool] | np.ndarray | None = None,
    bounds: Any = None,
    method: str = "analytic",
    options: Mapping[str, Any] | None = None,
    progress_cb=None,
    cancel_cb=None,
) -> dict[str, Any]:
    opts = dict(options or {})
    theta = _as_float_vector(theta_hat, name="theta_hat")
    param_names = [str(name) for name in parameter_names]
    if len(param_names) != theta.size:
        raise ValueError("parameter_names length does not match theta_hat.")
    fixed = _normalize_fixed_mask(fixed_mask, theta.size)
    bounds_norm = _normalize_bounds(bounds, theta.size)
    residuals = np.asarray(residual_fn(theta), dtype=float).reshape(-1)
    if residuals.size == 0:
        raise ValueError("Residual function returned an empty vector.")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("Residual function returned non-finite values.")

    free_idx = np.where(~fixed)[0]
    if free_idx.size == 0:
        result = _all_fixed_result(param_names, theta, residuals)
    else:
        jacobian = opts.get("jacobian")
        if jacobian is not None:
            J_full = np.asarray(jacobian, dtype=float)
            if J_full.ndim != 2 or J_full.shape[0] != residuals.size or J_full.shape[1] != theta.size:
                J_full = None
        else:
            J_full = None
        if J_full is None:
            J_full = _finite_difference_jacobian(
                residual_fn,
                theta,
                fixed_mask=fixed,
                bounds=bounds_norm,
                rel_step=float(opts.get("fd_rel_step", 1e-6) or 1e-6),
                abs_step=float(opts.get("fd_abs_step", 1e-8) or 1e-8),
            )
        J_free = np.asarray(J_full[:, free_idx], dtype=float)
        jtj = J_free.T @ J_free
        nobs = int(residuals.size)
        p_free = int(free_idx.size)
        dof = nobs - p_free
        sse = float(np.sum(residuals**2))
        s2 = sse / float(max(dof, 1))
        cov_free = s2 * np.linalg.pinv(jtj, rcond=float(opts.get("rcond", 1e-12) or 1e-12))
        cov_full = np.zeros((theta.size, theta.size), dtype=float)
        cov_full[np.ix_(free_idx, free_idx)] = cov_free
        se = np.sqrt(np.clip(np.diag(cov_full), 0.0, np.inf))
        corr = _cov_to_corr(cov_full)
        stability_diag = build_identifiability_report(jtj, cov_free, param_names=[param_names[i] for i in free_idx])
        result = {
            "theta_hat": theta,
            "param_names": param_names,
            "residuals": residuals,
            "jacobian": J_full,
            "covariance_matrix": cov_full,
            "correlation_matrix": corr,
            "se": se,
            "nobs": nobs,
            "p_free": p_free,
            "dof": dof,
            "sse": sse,
            "s2": float(s2),
            "rms": float(np.sqrt(np.mean(residuals**2))),
            "stability_diag": stability_diag,
            "samples": None,
            "bootstrap_summary": None,
        }

    if method == "analytic":
        return result

    if method == "bootstrap_onestep":
        raise NotImplementedError("Bootstrap (one-step LM) is not available for acid-base fits yet.")

    free_idx = np.where(~fixed)[0]
    if method == "bootstrap_linear":
        if free_idx.size == 0:
            samples = np.tile(theta, (int(opts.get("B", 0) or 0), 1))
        else:
            boot = bootstrap_linearized_wild(
                theta[free_idx],
                np.asarray(result["jacobian"], dtype=float)[:, free_idx].T,
                result["residuals"],
                int(opts.get("B", 500) or 500),
                seed=opts.get("seed"),
                wild=str(opts.get("wild") or "rademacher"),
                rcond=float(opts.get("rcond", 1e-12) or 1e-12),
                ridge=0.0,
            )
            samples = np.tile(theta, (boot["samples"].shape[0], 1))
            samples[:, free_idx] = np.asarray(boot["samples"], dtype=float)
        result["samples"] = samples
        result["bootstrap_summary"] = None
        return result

    if method != "bootstrap_full_refit_audit":
        raise ValueError(f"Unsupported acid-base error method: {method}")

    make_data_star_fn = opts.get("make_data_star_fn")
    refit_from_data = opts.get("refit_from_data")
    if not callable(make_data_star_fn):
        raise ValueError("Missing make_data_star_fn for acid-base bootstrap refit.")
    if not callable(refit_from_data):
        raise ValueError("Missing refit_from_data for acid-base bootstrap refit.")
    boot_summary = bootstrap_full_refit(
        theta,
        make_data_star_fn,
        refit_from_data,
        int(opts.get("B", 50) or 50),
        seed=opts.get("seed"),
        wild=str(opts.get("wild") or "rademacher"),
        max_iter=int(opts.get("max_iter", 30) or 30),
        tol=float(opts.get("tol", 1e-8) or 1e-8),
        fail_policy=str(opts.get("fail_policy") or "skip"),
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )
    result["samples"] = np.asarray(boot_summary.get("samples"), dtype=float)
    result["bootstrap_summary"] = boot_summary
    return result


def _stability_classification(diag: Mapping[str, Any]) -> str:
    status = str(diag.get("status") or "").strip().lower()
    if status == "excellent":
        return "stable"
    if status == "warn":
        return "warning"
    if status == "critical":
        return "ill-conditioned"
    return status or "unknown"


def _acid_base_notes(
    context: Mapping[str, Any],
    *,
    pka_values: np.ndarray,
    corr: np.ndarray,
    parameter_names: list[str],
    stability_diag: Mapping[str, Any],
    nobs: int,
    p_free: int,
) -> list[str]:
    notes: list[str] = []
    dataset = dict(context.get("dataset") or {})
    analysis_kind = str(context.get("analysis_kind") or "").strip().lower()

    pH_axis = None
    if dataset.get("pH") is not None:
        pH_axis = np.asarray(dataset.get("pH"), dtype=float).reshape(-1)
    elif dataset.get("pH_calculated") is not None:
        pH_axis = np.asarray(dataset.get("pH_calculated"), dtype=float).reshape(-1)
    elif dataset.get("pH_observed") is not None:
        pH_axis = np.asarray(dataset.get("pH_observed"), dtype=float).reshape(-1)
    if pH_axis is not None and pH_axis.size:
        pH_axis = pH_axis[np.isfinite(pH_axis)]
    if pH_axis is not None and pH_axis.size:
        pH_min = float(np.min(pH_axis))
        pH_max = float(np.max(pH_axis))
        for idx, value in enumerate(pka_values):
            if value < pH_min - 1.0 or value > pH_max + 1.0:
                notes.append(
                    f"pKa{idx + 1} lies outside the sampled transition region ({pH_min:.2f} to {pH_max:.2f} pH)."
                )

    if p_free > 0 and p_free >= max(2, nobs // 8):
        notes.append("Too many protonation steps or local parameters are being fitted relative to the amount of data.")

    high_pairs = list(stability_diag.get("high_corr_pairs") or [])
    for left, right, value in high_pairs:
        left_name = str(left).lower()
        right_name = str(right).lower()
        if left_name.startswith("pka") and right_name.startswith("pka"):
            notes.append(f"{left} and {right} are highly correlated (r={float(value):.3f}).")
        if {"electrode_e0", "electrode_slope"} & {left_name, right_name}:
            notes.append(f"Electrode parameters and acid-base constants are strongly correlated ({left}, {right}, r={float(value):.3f}).")
        if "pkw" in {left_name, right_name}:
            notes.append(f"pKw is strongly correlated with another fitted constant ({left}, {right}, r={float(value):.3f}).")

    if analysis_kind == "potentiometry" and dataset.get("pH_calculated") is not None:
        calc_pH = np.asarray(dataset.get("pH_calculated"), dtype=float).reshape(-1)
        calc_pH = calc_pH[np.isfinite(calc_pH)]
        if calc_pH.size >= 3:
            curvature = np.max(np.abs(np.diff(calc_pH, n=2)))
            if curvature < 0.02:
                notes.append("Potentiometric data show limited curvature, which weakens pKa identifiability.")

    if analysis_kind == "potentiometry" and pka_values.size and np.any(pka_values >= 10.0):
        lowered = [str(name).lower() for name in parameter_names]
        if "pkw" in lowered:
            pkw_idx = lowered.index("pkw")
            for idx, value in enumerate(pka_values):
                if value >= 10.0:
                    corr_val = abs(float(corr[idx, pkw_idx])) if corr.size else 0.0
                    if corr_val >= 0.95:
                        notes.append(
                            f"High-pH pKa{idx + 1} is strongly correlated with pKw (|r|={corr_val:.3f})."
                        )
    return notes


def _summarize_constant_table(
    labels: list[str],
    estimate: np.ndarray,
    se: np.ndarray,
    ci_2p5: np.ndarray,
    ci_97p5: np.ndarray,
    ci_16: np.ndarray | None,
    ci_84: np.ndarray | None,
) -> pd.DataFrame:
    data: dict[str, Any] = {
        "Parameter": labels,
        "Estimate": estimate,
        "SE": se,
        "CI 2.5": ci_2p5,
        "CI 97.5": ci_97p5,
    }
    if ci_16 is not None and ci_84 is not None:
        data["CI 16"] = ci_16
        data["CI 84"] = ci_84
    return pd.DataFrame(data)


def _build_acid_base_output(
    raw: dict[str, Any],
    context: Mapping[str, Any],
    method: str,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    theta_hat = np.asarray(raw["theta_hat"], dtype=float).reshape(-1)
    param_names = [str(name) for name in raw["param_names"]]
    fixed_mask = _normalize_fixed_mask(context.get("fixed_mask"), theta_hat.size)
    se = np.asarray(raw["se"], dtype=float).reshape(-1)
    cov = np.asarray(raw["covariance_matrix"], dtype=float)
    corr = np.asarray(raw["correlation_matrix"], dtype=float)
    bootstrap_samples = raw.get("samples")
    include_16_84 = bool(options.get("include_16_84", False))
    z95 = 1.96
    z68 = 1.0

    if method == "analytic" or bootstrap_samples is None or np.asarray(bootstrap_samples).size == 0:
        ci_2p5 = theta_hat - z95 * se
        ci_97p5 = theta_hat + z95 * se
        ci_16 = theta_hat - z68 * se if include_16_84 else None
        ci_84 = theta_hat + z68 * se if include_16_84 else None
        summary_median = theta_hat.copy()
        sample_corr = corr
        sample_count = None
        n_success = None
        n_fail = None
    else:
        samples = np.asarray(bootstrap_samples, dtype=float)
        summary_median = np.median(samples, axis=0)
        ci_2p5 = np.percentile(samples, 2.5, axis=0)
        ci_97p5 = np.percentile(samples, 97.5, axis=0)
        ci_16 = np.percentile(samples, 16.0, axis=0) if include_16_84 else None
        ci_84 = np.percentile(samples, 84.0, axis=0) if include_16_84 else None
        se = np.std(samples, axis=0, ddof=1 if samples.shape[0] > 1 else 0)
        sample_corr = np.corrcoef(samples, rowvar=False) if samples.shape[0] >= 2 else corr
        sample_corr = np.nan_to_num(sample_corr, nan=0.0)
        if sample_corr.ndim == 2:
            for idx in range(sample_corr.shape[0]):
                sample_corr[idx, idx] = 1.0
        sample_count = int(samples.shape[0])
        boot_summary = dict(raw.get("bootstrap_summary") or {})
        n_success = boot_summary.get("n_success")
        n_fail = boot_summary.get("n_fail")

    const_info = _acid_base_constant_info(param_names, context)
    const_idx = const_info["indices"]
    const_labels = const_info["labels"]
    primary_space = const_info["primary_space"]
    constant_estimate = theta_hat[const_idx] if const_idx else np.asarray([], dtype=float)
    constant_cov = cov[np.ix_(const_idx, const_idx)] if const_idx else np.empty((0, 0), dtype=float)

    if primary_space == "log_beta":
        log_beta_estimate = constant_estimate
        cov_log_beta = constant_cov
        pka_estimate, cov_pka, _ = propagate_log_beta_to_pka(log_beta_estimate, cov_log_beta) if const_idx else (np.asarray([]), np.empty((0, 0)), np.empty((0, 0)))
    else:
        pka_estimate = constant_estimate
        cov_pka = constant_cov
        log_beta_estimate, cov_log_beta, _ = propagate_pka_to_log_beta(pka_estimate, cov_pka) if const_idx else (np.asarray([]), np.empty((0, 0)), np.empty((0, 0)))

    se_pka = np.sqrt(np.clip(np.diag(cov_pka), 0.0, np.inf)) if cov_pka.size else np.asarray([], dtype=float)
    se_log_beta = np.sqrt(np.clip(np.diag(cov_log_beta), 0.0, np.inf)) if cov_log_beta.size else np.asarray([], dtype=float)

    if bootstrap_samples is not None and np.asarray(bootstrap_samples).size and const_idx:
        samples = np.asarray(bootstrap_samples, dtype=float)
        samples_constants = samples[:, const_idx]
        if primary_space == "log_beta":
            samples_log_beta = samples_constants
            samples_pka = np.asarray([log_beta_to_pka(row.tolist()) for row in samples_log_beta], dtype=float)
        else:
            samples_pka = samples_constants
            samples_log_beta = np.asarray([pka_to_log_beta(row.tolist()) for row in samples_pka], dtype=float)
        pka_ci_2p5 = np.percentile(samples_pka, 2.5, axis=0)
        pka_ci_97p5 = np.percentile(samples_pka, 97.5, axis=0)
        pka_ci_16 = np.percentile(samples_pka, 16.0, axis=0) if include_16_84 else None
        pka_ci_84 = np.percentile(samples_pka, 84.0, axis=0) if include_16_84 else None
        log_beta_ci_2p5 = np.percentile(samples_log_beta, 2.5, axis=0)
        log_beta_ci_97p5 = np.percentile(samples_log_beta, 97.5, axis=0)
        log_beta_ci_16 = np.percentile(samples_log_beta, 16.0, axis=0) if include_16_84 else None
        log_beta_ci_84 = np.percentile(samples_log_beta, 84.0, axis=0) if include_16_84 else None
        se_pka = np.std(samples_pka, axis=0, ddof=1 if samples_pka.shape[0] > 1 else 0)
        se_log_beta = np.std(samples_log_beta, axis=0, ddof=1 if samples_log_beta.shape[0] > 1 else 0)
    else:
        pka_ci_2p5 = pka_estimate - z95 * se_pka
        pka_ci_97p5 = pka_estimate + z95 * se_pka
        pka_ci_16 = pka_estimate - z68 * se_pka if include_16_84 else None
        pka_ci_84 = pka_estimate + z68 * se_pka if include_16_84 else None
        log_beta_ci_2p5 = log_beta_estimate - z95 * se_log_beta
        log_beta_ci_97p5 = log_beta_estimate + z95 * se_log_beta
        log_beta_ci_16 = log_beta_estimate - z68 * se_log_beta if include_16_84 else None
        log_beta_ci_84 = log_beta_estimate + z68 * se_log_beta if include_16_84 else None

    linear_beta = _linear_metrics_from_log_beta(log_beta_estimate, se_log_beta) if log_beta_estimate.size else {
        "linear_value": np.asarray([], dtype=float),
        "se_linear": np.asarray([], dtype=float),
        "percent_error": np.asarray([], dtype=float),
    }
    linear_ka = _linear_metrics_from_pka(pka_estimate, se_pka) if pka_estimate.size else {
        "linear_value": np.asarray([], dtype=float),
        "se_linear": np.asarray([], dtype=float),
        "percent_error": np.asarray([], dtype=float),
    }

    fixed_labels = ["Yes" if bool(value) else "" for value in fixed_mask]
    primary_rows: list[list[Any]] = []
    for idx, name in enumerate(param_names):
        primary_rows.append(
            [
                name,
                theta_hat[idx],
                se[idx],
                ci_2p5[idx],
                ci_97p5[idx],
                (ci_16[idx] if ci_16 is not None else ""),
                (ci_84[idx] if ci_84 is not None else ""),
                (_parameter_percent_error(name, theta_hat[idx], se[idx]) if _parameter_percent_error(name, theta_hat[idx], se[idx]) is not None else ""),
                fixed_labels[idx],
            ]
        )

    pka_labels = [f"pKa{i + 1}" for i in range(pka_estimate.size)]
    log_beta_labels = [f"log_beta{i + 1}" for i in range(log_beta_estimate.size)]
    pka_df = _summarize_constant_table(
        pka_labels,
        pka_estimate,
        se_pka,
        pka_ci_2p5,
        pka_ci_97p5,
        pka_ci_16,
        pka_ci_84,
    )
    log_beta_df = _summarize_constant_table(
        log_beta_labels,
        log_beta_estimate,
        se_log_beta,
        log_beta_ci_2p5,
        log_beta_ci_97p5,
        log_beta_ci_16,
        log_beta_ci_84,
    )

    linear_rows: list[dict[str, Any]] = []
    for idx in range(log_beta_estimate.size):
        linear_rows.append(
            {
                "Constant": f"beta{idx + 1}",
                "log value": log_beta_estimate[idx],
                "value in linear scale": linear_beta["linear_value"][idx],
                "SE(log)": se_log_beta[idx],
                "SE(linear)": linear_beta["se_linear"][idx],
                "% error": linear_beta["percent_error"][idx],
            }
        )
    for idx in range(pka_estimate.size):
        linear_rows.append(
            {
                "Constant": f"Ka{idx + 1}",
                "log value": -pka_estimate[idx],
                "value in linear scale": linear_ka["linear_value"][idx],
                "SE(log)": se_pka[idx],
                "SE(linear)": linear_ka["se_linear"][idx],
                "% error": linear_ka["percent_error"][idx],
            }
        )
    linear_df = pd.DataFrame(linear_rows)

    stability_diag = dict(raw.get("stability_diag") or {})
    notes = _acid_base_notes(
        context,
        pka_values=pka_estimate,
        corr=np.asarray(sample_corr, dtype=float),
        parameter_names=param_names,
        stability_diag=stability_diag,
        nobs=int(raw.get("nobs") or 0),
        p_free=int(raw.get("p_free") or 0),
    )

    method_labels = {
        "analytic": "Analytical covariance",
        "bootstrap_linear": "Bootstrap (linearized + wild)",
        "bootstrap_full_refit_audit": "Bootstrap (full refit, audit)",
    }
    summary_lines = [f"Method: {method_labels.get(method, method)}"]
    summary_lines.append(
        f"nobs={int(raw.get('nobs') or 0)}, p_free={int(raw.get('p_free') or 0)}, dof={int(raw.get('dof') or 0)}"
    )
    summary_lines.append(
        f"SSE={float(raw.get('sse') or 0.0):.6g}, RMS={float(raw.get('rms') or 0.0):.6g}, s2={float(raw.get('s2') or 0.0):.6g}"
    )
    if sample_count is not None:
        summary_lines.append(f"bootstrap_samples={sample_count}")
    if n_success is not None or n_fail is not None:
        summary_lines.append(f"n_success={int(n_success or 0)}, n_fail={int(n_fail or 0)}")
    if stability_diag:
        summary_lines.append(f"Stability class: {_stability_classification(stability_diag)}")
        diag_summary = str(stability_diag.get("diag_summary") or "").strip()
        if diag_summary:
            summary_lines.append(diag_summary)
    if notes:
        summary_lines.append("")
        summary_lines.append("Acid-base diagnostics:")
        for note in notes:
            summary_lines.append(f"- {note}")
    diag_full = str(stability_diag.get("diag_full") or "").strip()
    if diag_full:
        summary_lines.append("")
        summary_lines.append(diag_full)

    diagnostics_rows = [
        {"metric": "classification", "value": _stability_classification(stability_diag)},
        {"metric": "condition_number", "value": stability_diag.get("cond_jjt")},
        {"metric": "effective_rank", "value": stability_diag.get("rank_eff")},
        {"metric": "p_free", "value": stability_diag.get("p_free")},
        {"metric": "max_abs_correlation", "value": stability_diag.get("max_abs_corr")},
    ]
    for idx, note in enumerate(notes, start=1):
        diagnostics_rows.append({"metric": f"note_{idx}", "value": note})

    export_frames: dict[str, Any] = {
        "Parameters": pd.DataFrame(
            primary_rows,
            columns=[
                "Parameter",
                "Estimate",
                "SE",
                "CI 2.5",
                "CI 97.5",
                "CI 16",
                "CI 84",
                "%err(linear)",
                "Fixed",
            ],
        ),
        "pKa": pka_df,
        "log_beta": log_beta_df,
        "Derived constants": linear_df,
        "Covariance": pd.DataFrame(cov, index=param_names, columns=param_names),
        "Correlation": pd.DataFrame(sample_corr, index=param_names, columns=param_names),
        "Error diagnostics": pd.DataFrame(diagnostics_rows),
    }
    if bootstrap_samples is not None and np.asarray(bootstrap_samples).size:
        export_frames["Bootstrap samples"] = pd.DataFrame(np.asarray(bootstrap_samples), columns=param_names)

    return {
        "param_names": param_names,
        "method": method,
        "B": options.get("B"),
        "seed": options.get("seed"),
        "wild": options.get("wild"),
        "lam": options.get("lam"),
        "max_iter": options.get("max_iter"),
        "tol": options.get("tol"),
        "n_success": n_success,
        "n_fail": n_fail,
        "dof": int(raw.get("dof") or 0),
        "rms": float(raw.get("rms") or 0.0),
        "s2": float(raw.get("s2") or 0.0),
        "corr": np.asarray(sample_corr, dtype=float),
        "summary": "\n".join(summary_lines),
        "results_columns": [
            "Parameter",
            "Estimate",
            "SE",
            "CI 2.5",
            "CI 97.5",
            "CI 16",
            "CI 84",
            "%err(linear)",
            "Fixed",
        ],
        "results_rows": primary_rows,
        "extra_tables": [
            {
                "name": "pKa",
                "columns": list(pka_df.columns),
                "rows": pka_df.values.tolist(),
            },
            {
                "name": "log_beta",
                "columns": list(log_beta_df.columns),
                "rows": log_beta_df.values.tolist(),
            },
            {
                "name": "Derived constants",
                "columns": list(linear_df.columns),
                "rows": linear_df.values.tolist(),
            },
            {
                "name": "Covariance",
                "columns": param_names,
                "rows": pd.DataFrame(cov).values.tolist(),
                "row_headers": param_names,
            },
        ],
        "export_frames": export_frames,
        "core_metrics": {
            "covariance_matrix": cov,
            "correlation_matrix": np.asarray(sample_corr, dtype=float),
            "stability_diag": stability_diag,
            "notes": notes,
            "pka_table": pka_df,
            "log_beta_table": log_beta_df,
            "derived_constants_table": linear_df,
            "bootstrap_samples": None if bootstrap_samples is None else np.asarray(bootstrap_samples, dtype=float),
        },
    }


def _serialize_bounds(bounds: Any) -> tuple[np.ndarray, np.ndarray]:
    return _normalize_bounds(bounds, len(bounds) if isinstance(bounds, list) else 0)


def _system_from_serializable_model(model_meta: Mapping[str, Any]) -> AcidBaseSystem:
    return acid_base_system_from_model(
        dict(model_meta.get("acid_base_model") or {}),
        temperature=float(model_meta.get("temperature", 298.15) or 298.15),
        ionic_strength=(
            None
            if model_meta.get("ionic_strength") in (None, "")
            else float(model_meta.get("ionic_strength"))
        ),
        kw=float(model_meta.get("kw", 1e-14) or 1e-14),
    )


def _system_from_context(context: Mapping[str, Any]):
    model_meta = dict(context.get("model") or {})
    return _system_from_serializable_model(model_meta)


def _potentiometry_objects(context: Mapping[str, Any], observed_override: np.ndarray | None = None):
    dataset = dict(context.get("dataset") or {})
    fit_options = dict(context.get("fit_options") or {})
    experiment = PotentiometryExperiment(
        initial_volume=float(dataset.get("initial_volume", 10.0) or 10.0),
        titrant_volumes=np.asarray(dataset.get("volume_mL") or [], dtype=float),
        measured_pH=(
            observed_override
            if observed_override is not None and str(dataset.get("observed_kind") or "ph") == "ph"
            else (
                None
                if dataset.get("pH_observed") is None
                else np.asarray(dataset.get("pH_observed"), dtype=float)
            )
        ),
        measured_emf=(
            observed_override
            if observed_override is not None and str(dataset.get("observed_kind") or "ph") == "emf"
            else (
                None
                if dataset.get("E_mV_observed") is None
                else np.asarray(dataset.get("E_mV_observed"), dtype=float)
            )
        ),
        analyte_concentration=dataset.get("analyte_concentration"),
        titrant_concentration=dataset.get("titrant_concentration"),
        titrant_type=str(dataset.get("titrant_type") or "base"),
        temperature=float(dataset.get("temperature", 298.15) or 298.15),
        electrode_e0=dataset.get("electrode_e0"),
        electrode_slope=dataset.get("electrode_slope"),
        volume_offset=float(dataset.get("volume_offset", 0.0) or 0.0),
    )
    return experiment, fit_options


def _spectroscopy_objects(context: Mapping[str, Any], observed_override: np.ndarray | None = None):
    dataset = dict(context.get("dataset") or {})
    fit_options = dict(context.get("fit_options") or {})
    signal = np.asarray(dataset.get("signal_observed"), dtype=float)
    if observed_override is not None:
        signal = np.asarray(observed_override, dtype=float)
    obj = SpectroscopicAcidBaseDataset(
        pH=np.asarray(dataset.get("pH") or [], dtype=float),
        signal=signal,
        wavelengths=(
            None
            if dataset.get("wavelengths") is None
            else np.asarray(dataset.get("wavelengths"), dtype=float)
        ),
    )
    return obj, fit_options


def _nmr_objects(context: Mapping[str, Any], observed_override: np.ndarray | None = None):
    dataset = dict(context.get("dataset") or {})
    fit_options = dict(context.get("fit_options") or {})
    shifts = np.asarray(dataset.get("shifts_observed"), dtype=float)
    if observed_override is not None:
        shifts = np.asarray(observed_override, dtype=float)
    obj = NMRAcidBaseDataset(
        pH=np.asarray(dataset.get("pH") or [], dtype=float),
        shifts=shifts,
        nuclei_labels=[str(value) for value in (dataset.get("nuclei_labels") or [])],
    )
    return obj, fit_options


def _residual_fn_from_context(context: Mapping[str, Any], observed_override: np.ndarray | None = None):
    analysis_kind = str(context.get("analysis_kind") or "").strip().lower()
    system = _system_from_context(context)
    if analysis_kind == "potentiometry":
        experiment, fit_options = _potentiometry_objects(context, observed_override=observed_override)
        return lambda theta: potentiometry_residuals(theta, experiment, system, fit_options)
    if analysis_kind == "spectroscopy":
        dataset, fit_options = _spectroscopy_objects(context, observed_override=observed_override)
        return lambda theta: spectroscopy_acid_base_residuals(theta, dataset, system, fit_options)
    if analysis_kind == "nmr":
        dataset, fit_options = _nmr_objects(context, observed_override=observed_override)
        return lambda theta: nmr_acid_base_residuals(theta, dataset, system, fit_options)
    raise ValueError(f"Unknown acid-base analysis kind: {analysis_kind}")


def _make_data_star_fn(context: Mapping[str, Any], fit_values: np.ndarray):
    dataset = dict(context.get("dataset") or {})
    analysis_kind = str(context.get("analysis_kind") or "").strip().lower()
    if analysis_kind == "potentiometry":
        observed_kind = str(dataset.get("observed_kind") or "ph")
        observed = dataset.get("E_mV_observed") if observed_kind == "emf" else dataset.get("pH_observed")
        fitted = dataset.get("E_mV_calculated") if observed_kind == "emf" else dataset.get("pH_calculated")
        observed_arr = np.asarray(observed, dtype=float).reshape(-1)
        fitted_arr = np.asarray(fitted, dtype=float).reshape(-1)
        residuals = observed_arr - fitted_arr

        def _fn(rng, wild: str = "rademacher"):
            multipliers = wild_multipliers(rng, residuals.size, kind=wild)
            return fitted_arr + residuals * multipliers

        return _fn

    if analysis_kind == "spectroscopy":
        observed = np.asarray(dataset.get("signal_observed"), dtype=float)
        fitted = np.asarray(dataset.get("signal_calculated"), dtype=float)
        residuals = observed - fitted

        def _fn(rng, wild: str = "rademacher"):
            multipliers = wild_multipliers(rng, residuals.size, kind=wild).reshape(residuals.shape)
            return fitted + residuals * multipliers

        return _fn

    if analysis_kind == "nmr":
        observed = np.asarray(dataset.get("shifts_observed"), dtype=float)
        fitted = np.asarray(dataset.get("shifts_calculated"), dtype=float)
        residuals = observed - fitted

        def _fn(rng, wild: str = "rademacher"):
            multipliers = wild_multipliers(rng, residuals.size, kind=wild).reshape(residuals.shape)
            return fitted + residuals * multipliers

        return _fn

    raise ValueError("Unknown acid-base analysis kind for bootstrap data generation.")


def _refit_from_context(
    context: Mapping[str, Any],
    data_star: np.ndarray,
    theta0: np.ndarray,
    *,
    max_iter: int = 30,
    tol: float = 1e-8,
):
    theta0 = np.asarray(theta0, dtype=float).reshape(-1)
    fixed_mask = _normalize_fixed_mask(context.get("fixed_mask"), theta0.size)
    lower, upper = _normalize_bounds(context.get("bounds"), theta0.size)
    free_idx = np.where(~fixed_mask)[0]
    if free_idx.size == 0:
        return theta0.copy(), True, {"message": "All parameters fixed."}

    residual_full = _residual_fn_from_context(context, observed_override=np.asarray(data_star, dtype=float))

    def residual_free(free_theta: np.ndarray) -> np.ndarray:
        theta = theta0.copy()
        theta[free_idx] = np.asarray(free_theta, dtype=float).reshape(-1)
        return residual_full(theta)

    max_nfev = max(50, int(max_iter) * max(int(free_idx.size), 1) * 10)
    opt = least_squares(
        residual_free,
        theta0[free_idx],
        bounds=(lower[free_idx], upper[free_idx]),
        xtol=float(tol),
        ftol=float(tol),
        gtol=float(tol),
        max_nfev=max_nfev,
    )
    theta_star = theta0.copy()
    theta_star[free_idx] = np.asarray(opt.x, dtype=float).reshape(-1)
    return theta_star, bool(opt.success), {"message": str(opt.message), "nfev": int(opt.nfev)}


def compute_errors_acid_base_from_context(
    context: Mapping[str, Any],
    options: Mapping[str, Any] | None = None,
    progress_cb=None,
    cancel_cb=None,
) -> dict[str, Any]:
    ctx = dict(context or {})
    opts = dict(options or {})
    theta_hat = _as_float_vector(ctx.get("theta_hat"), name="theta_hat")
    parameter_names = [str(value) for value in (ctx.get("parameter_names") or [])]
    fixed_mask = _normalize_fixed_mask(ctx.get("fixed_mask"), theta_hat.size)
    method = str(opts.get("method") or "analytic")

    residual_fn = _residual_fn_from_context(ctx)
    call_options = dict(opts)
    if ctx.get("jacobian") is not None:
        call_options["jacobian"] = np.asarray(ctx.get("jacobian"), dtype=float)
    if method == "bootstrap_full_refit_audit":
        call_options["make_data_star_fn"] = _make_data_star_fn(ctx, theta_hat)
        call_options["refit_from_data"] = lambda data_star, theta0, max_iter=30, tol=1e-8: _refit_from_context(
            ctx,
            np.asarray(data_star, dtype=float),
            np.asarray(theta0, dtype=float),
            max_iter=max_iter,
            tol=tol,
        )
    raw = compute_errors_acid_base(
        residual_fn,
        theta_hat,
        parameter_names,
        fixed_mask=fixed_mask,
        bounds=ctx.get("bounds"),
        method=method,
        options=call_options,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )
    return _build_acid_base_output(raw, ctx, method, opts)


__all__ = [
    "compute_errors_acid_base",
    "compute_errors_acid_base_from_context",
    "propagate_log_beta_to_pka",
    "propagate_pka_to_log_beta",
]
