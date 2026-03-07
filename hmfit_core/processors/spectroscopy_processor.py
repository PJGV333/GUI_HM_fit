# SPDX-License-Identifier: GPL-3.0-or-later
"""
Spectroscopy processor module.
Business logic for data processing (formerly used in FastAPI).
"""
import io
import base64
import numpy as onp
from ..utils.np_backend import xp as np, jit, jacrev, vmap, lax
import pandas as pd
import matplotlib
import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

# When used from a GUI (historical reference) we don't want to override the interactive backend.
# For headless/server usage, force a non-interactive backend before importing pyplot.
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
from matplotlib.figure import Figure
from scipy import optimize
from scipy.optimize import differential_evolution, dual_annealing, basinhopping
import warnings
import re
import logging
warnings.filterwarnings("ignore")
from ..utils.errors import compute_errors_spectro_varpro, pinv_cs, percent_error_log10K, sensitivities_wrt_logK
from ..utils.nnls_utils import solve_A_nnls_pgd, solve_A_nnls_pgd2
from ..utils.noncoop_utils import noncoop_derived_from_logK1
from ..utils.solver_param_transform import (
    SolverParamTransformWrapper,
    coerce_param_transform,
    expand_solver_params,
)

logger = logging.getLogger(__name__)

# === Progress tracking (Historical WebSocket support) ===
_progress_callback = None
_loop = None
_cancel_callback = None

def set_progress_callback(callback, loop=None):
    """Registrar callback para emitir progreso (p.ej. al WebSocket)."""
    global _progress_callback, _loop, _cancel_callback
    _progress_callback = callback
    _loop = loop
    _cancel_callback = getattr(callback, "_hmfit_cancel", None)


def _cancel_requested() -> bool:
    if _cancel_callback is None:
        return False
    try:
        return bool(_cancel_callback())
    except Exception:
        return False


def _raise_if_cancelled():
    if not _cancel_requested():
        return
    try:
        from hmfit_core.api import FitCancelled
    except Exception:
        raise RuntimeError("Fit cancelled.")
    raise FitCancelled("Fit cancelled.")

def log_progress(message: str):
    """Enviar mensaje de progreso si hay callback; si no, imprime a consola."""
    _raise_if_cancelled()
    if _progress_callback:
        if _loop:
            _loop.call_soon_threadsafe(_progress_callback, message)
        else:
            _progress_callback(message)
        return
    print(message)


def _is_fit_cancelled(exc: BaseException) -> bool:
    return exc.__class__.__name__ == "FitCancelled"

# === Shared helper utilities (used by both Spectroscopy and NMR) ===
_alias_re = re.compile(r"\(([^)]+)\)")

def _aliases_from_names(names):
    """Build a dictionary mapping aliases to column indices from column names."""
    alias2idx = {}
    for i, nm in enumerate(names):
        s = str(nm).lower()
        # Extract aliases from parentheses
        for a in _alias_re.findall(s):
            alias2idx[a.strip().lower()] = i
        # Also use first token as alias
        head = s.split("(")[0].strip().split()
        if head:
            alias2idx[head[0]] = i
    return alias2idx

def _build_bounds_list(bounds):
    """
    Convert list of bound dicts to list of (min, max) tuples.
    Each dict should have 'min' and 'max' keys, or be a list/tuple.
    Empty/None values default to ±inf.
    """
    def _to_bound(val, default):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return default
        if onp.isnan(v):
            return default
        return v
    
    processed = []
    for raw in bounds:
        if isinstance(raw, dict):
            min_val = _to_bound(raw.get('min'), -onp.inf)
            max_val = _to_bound(raw.get('max'), onp.inf)
        elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
            min_val = _to_bound(raw[0], -onp.inf)
            max_val = _to_bound(raw[1], onp.inf)
        else:
            min_val, max_val = -onp.inf, onp.inf
        processed.append((min_val, max_val))
    return processed


# === Result formatting (Historical reference to previous UI output) ===
def format_results_table(k, SE_log10K, percK, rms, covfit, lof=None, fixed_mask=None, param_names=None):
    """
    Build an ASCII table with aligned columns for constants and diagnostics.
    Build an ASCII table with aligned columns for constants and diagnostics.
    """
    # Helper: maximum widths per column
    def calculate_max_column_widths(headers, data_rows):
        widths = [len(h) for h in headers]
        for row in data_rows:
            for i, item in enumerate(row):
                widths[i] = max(widths[i], len(str(item)))
        return widths

    headers = [
        "Constant",
        "log10(K) ± SE(log10K)",
        "% Error (K, Δ-method)",
        "RMS",
        "s² (var. reducida)",
    ]

    rows = []
    for i in range(len(k)):
        is_fixed = bool(fixed_mask[i]) if fixed_mask is not None and i < len(fixed_mask) else False
        param_label = (
            str(param_names[i])
            if param_names is not None and i < len(param_names) and str(param_names[i]).strip()
            else f"K{i+1}"
        )
        if is_fixed:
            se_str = "const"
            perc_str = ""
        else:
            se_str = f"{SE_log10K[i]:.2e}"
            perc_str = f"{percK[i]:.2f} %"

        rows.append([
            param_label,
            f"{k[i]:.2e} ± {se_str}",
            perc_str,
            f"{rms:.2e}" if i == 0 else "",
            f"{covfit:.2e}" if i == 0 else "",
        ])

    max_widths = calculate_max_column_widths(headers, rows)

    table_lines = []
    header_line = " | ".join(f"{h.ljust(max_widths[i])}" for i, h in enumerate(headers))
    table_lines.append("-" * len(header_line))
    table_lines.append(header_line)
    table_lines.append("-" * len(header_line))
    for row in rows:
        line = " | ".join(f"{str(item).ljust(max_widths[i])}" for i, item in enumerate(row))
        table_lines.append(line)

    table = "\n".join(table_lines)
    return "=== RESULTADOS ===\n" + table

def pinv_cs_local(A, rcond=1e-12):
    """
    Pseudoinversa estable (ahora sin complex-step).
    Thin wrapper sobre np.linalg.pinv con fallback por si SVD no converge.
    """
    try:
        return np.linalg.pinv(A, rcond=rcond)
    except np.linalg.LinAlgError:
        # Fallback regularizado tipo ridge
        ATA = A.T @ A + (rcond if np.isscalar(rcond) else 1e-12) * np.eye(A.shape[1], dtype=A.dtype)
        return np.linalg.solve(ATA, A.T)

def _solve_A(C, YT, rcond=1e-10):
    # C: (m×s), YT: (nw×m)  →  A: (s×nw)
    A, *_ = np.linalg.lstsq(C, YT, rcond=rcond)
    return A

def apply_efa_svd(Y: onp.ndarray, ev_requested: int | None):
    """
    Apply global rank-k SVD reconstruction (EFA denoise).
    Returns (Y_denoised, ev_used, max_ev).
    """
    Y_arr = onp.asarray(Y)
    if Y_arr.ndim != 2:
        raise ValueError(f"EFA expects a 2D matrix, got shape={Y_arr.shape}")
    max_ev = int(min(Y_arr.shape))
    if max_ev < 1:
        return Y_arr, 0, max_ev
    if ev_requested is None:
        return Y_arr, max_ev, max_ev
    try:
        ev_used = int(ev_requested)
    except (TypeError, ValueError):
        ev_used = 0
    if ev_used <= 0:
        ev_used = max_ev
    ev_used = max(1, min(ev_used, max_ev))
    U, s, Vt = onp.linalg.svd(Y_arr, full_matrices=False)
    Y_rec = (U[:, :ev_used] * s[:ev_used]) @ Vt[:ev_used, :]
    return Y_rec, ev_used, max_ev


def _normalize_mode(raw, allowed, default):
    mode = str(raw or default).strip().lower()
    if mode not in allowed:
        return default
    return mode


def baseline_correct(
    Y,
    wavelengths,
    mode="off",
    start=450.0,
    end=600.0,
    auto_quantile=0.20,
    apply_per_spectrum=True,
):
    """
    Baseline correction over spectral axis (columns = wavelengths).
    Y shape: (m_points, n_lambda)
    """
    Y_arr = onp.asarray(Y, dtype=float)
    wl = onp.asarray(wavelengths, dtype=float).ravel()
    if Y_arr.ndim != 2:
        raise ValueError(f"baseline_correct expects 2D Y, got shape={Y_arr.shape}")
    if wl.size != Y_arr.shape[1]:
        raise ValueError(f"baseline_correct wavelength size mismatch: {wl.size} != {Y_arr.shape[1]}")

    mode_norm = _normalize_mode(mode, {"off", "range", "auto"}, "off")
    baseline_vals = onp.zeros(Y_arr.shape[0], dtype=float)
    meta = {
        "mode": mode_norm,
        "apply_per_spectrum": bool(apply_per_spectrum),
        "n_baseline_channels": 0,
    }
    if mode_norm == "off":
        return Y_arr.copy(), baseline_vals, meta

    if mode_norm == "range":
        lo, hi = sorted((float(start), float(end)))
        baseline_idx = (wl >= lo) & (wl <= hi)
        meta["start"] = lo
        meta["end"] = hi
        if not baseline_idx.any():
            # Guardrail: fallback automático si el rango no existe tras recorte.
            meta["requested_mode"] = "range"
            meta["warning"] = (
                "No channels found in selected baseline range. Falling back to auto baseline."
            )
            mode_norm = "auto"
            meta["mode"] = "auto"
            q = float(auto_quantile)
            if not onp.isfinite(q):
                q = 0.20
            q = min(max(q, 0.0), 1.0)
            std_per_lambda = onp.nanstd(Y_arr, axis=0)
            std_per_lambda = onp.nan_to_num(std_per_lambda, nan=0.0, posinf=0.0, neginf=0.0)
            thr = float(onp.quantile(std_per_lambda, q))
            baseline_idx = std_per_lambda <= thr
            if not baseline_idx.any():
                baseline_idx[onp.argmin(std_per_lambda)] = True
            meta["quantile"] = q
            meta["threshold"] = thr
    else:
        q = float(auto_quantile)
        if not onp.isfinite(q):
            q = 0.20
        q = min(max(q, 0.0), 1.0)
        std_per_lambda = onp.nanstd(Y_arr, axis=0)
        std_per_lambda = onp.nan_to_num(std_per_lambda, nan=0.0, posinf=0.0, neginf=0.0)
        thr = float(onp.quantile(std_per_lambda, q))
        baseline_idx = std_per_lambda <= thr
        if not baseline_idx.any():
            baseline_idx[onp.argmin(std_per_lambda)] = True
        meta["quantile"] = q
        meta["threshold"] = thr

    n_base = int(onp.count_nonzero(baseline_idx))
    meta["n_baseline_channels"] = n_base
    if n_base <= 0:
        return Y_arr.copy(), baseline_vals, meta

    if bool(apply_per_spectrum):
        baseline_vals = onp.nanmean(Y_arr[:, baseline_idx], axis=1)
        baseline_vals = onp.nan_to_num(baseline_vals, nan=0.0, posinf=0.0, neginf=0.0)
        Y_corr = Y_arr - baseline_vals[:, None]
    else:
        b0 = float(onp.nanmean(Y_arr[:, baseline_idx]))
        if not onp.isfinite(b0):
            b0 = 0.0
        baseline_vals = onp.full(Y_arr.shape[0], b0, dtype=float)
        Y_corr = Y_arr - b0

    return Y_corr, baseline_vals, meta


def compute_spectral_weights(Y, mode="none", eps=1e-12, power=1.0, normalize=True):
    """
    Build per-wavelength weights from Y shape (m_points, n_lambda).
    """
    Y_arr = onp.asarray(Y, dtype=float)
    if Y_arr.ndim != 2:
        raise ValueError(f"compute_spectral_weights expects 2D Y, got shape={Y_arr.shape}")
    n_lambda = int(Y_arr.shape[1])
    mode_norm = _normalize_mode(mode, {"none", "std", "max"}, "none")
    try:
        eps_val = float(eps)
    except (TypeError, ValueError):
        eps_val = 1e-12
    if not onp.isfinite(eps_val):
        eps_val = 1e-12
    eps_val = max(eps_val, 1e-16)
    try:
        p = float(power)
    except (TypeError, ValueError):
        p = 1.0
    if not onp.isfinite(p):
        p = 1.0

    if mode_norm == "none":
        w = onp.ones(n_lambda, dtype=float)
        return w, {
            "mode": mode_norm,
            "eps": eps_val,
            "power": p,
            "normalize": bool(normalize),
            "w_min": 1.0,
            "w_max": 1.0,
        }

    if mode_norm == "std":
        w = onp.nanstd(Y_arr, axis=0)
    elif mode_norm == "max":
        w = onp.nanmax(onp.abs(Y_arr), axis=0)
    else:
        w = onp.ones(n_lambda, dtype=float)

    w = onp.nan_to_num(onp.abs(w), nan=0.0, posinf=0.0, neginf=0.0)
    if bool(normalize):
        w_max = float(onp.max(w)) if w.size else 0.0
        if w_max > eps_val:
            w = w / (w_max + eps_val)
        else:
            w = onp.ones_like(w)
            warning = "Degenerate weighting profile (max≈0). Falling back to uniform weights."
            meta = {
                "mode": mode_norm,
                "eps": eps_val,
                "power": p,
                "normalize": bool(normalize),
                "w_min": 1.0,
                "w_max": 1.0,
                "warning": warning,
            }
            return w, meta
    if p != 1.0:
        w = w ** p
    w = onp.maximum(w, eps_val)
    if not onp.isfinite(w).all():
        w = onp.ones(n_lambda, dtype=float)
        warning = "Invalid weighting values encountered. Falling back to uniform weights."
    else:
        warning = None

    meta = {
        "mode": mode_norm,
        "eps": eps_val,
        "power": p,
        "normalize": bool(normalize),
        "w_min": float(onp.min(w)) if w.size else 0.0,
        "w_max": float(onp.max(w)) if w.size else 1.0,
    }
    if warning:
        meta["warning"] = warning
    return w, meta


def _compute_lower_bound(C, YT, delta_mode="off", delta_rel=0.01):
    mode_norm = _normalize_mode(delta_mode, {"off", "relative"}, "off")
    if mode_norm == "off":
        return None
    rel = float(delta_rel)
    if not onp.isfinite(rel) or rel <= 0.0:
        return None
    try:
        A0 = onp.linalg.pinv(onp.asarray(C, dtype=float)) @ onp.asarray(YT, dtype=float)
    except Exception:
        return None
    if A0.size == 0:
        return None
    amax = onp.nanmax(onp.abs(A0), axis=1)
    amax = onp.nan_to_num(amax, nan=0.0, posinf=0.0, neginf=0.0)
    if not onp.isfinite(amax).all() or float(onp.max(amax)) <= 0.0:
        return None
    return -rel * amax


def _build_smoothness_laplacian(n_lambda: int):
    """
    Build L = D2.T @ D2 for second-difference smoothness over wavelength axis.
    """
    n = int(max(n_lambda, 0))
    if n <= 0:
        return onp.zeros((0, 0), dtype=float)
    if n < 3:
        return onp.zeros((n, n), dtype=float)
    D2 = onp.zeros((n - 2, n), dtype=float)
    for i in range(n - 2):
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0
    return D2.T @ D2


def _solve_absorptivities(
    C,
    YT,
    eps_solver_mode="soft_penalty",
    mu=1e-2,
    delta_mode="off",
    delta_rel=0.01,
    alpha_smooth=0.0,
    smooth_matrix=None,
    max_iters=300,
):
    mode_norm = _normalize_mode(eps_solver_mode, {"soft_penalty", "soft_bound", "nnls_hard"}, "soft_penalty")
    if mode_norm == "nnls_hard":
        return solve_A_nnls_pgd2(C, YT, ridge=0.0, max_iters=max_iters), None
    lb_mode = "relative" if mode_norm == "soft_bound" else delta_mode
    lower_bound = _compute_lower_bound(C, YT, delta_mode=lb_mode, delta_rel=delta_rel)
    A = solve_A_nnls_pgd(
        C,
        YT,
        ridge=0.0,
        mu=float(mu),
        max_iters=max_iters,
        lower_bound=lower_bound,
        alpha_smooth=float(alpha_smooth),
        smooth_matrix=smooth_matrix,
    )
    return A, lower_bound


def _normalize_species_names(species_names, nspec):
    names = []
    raw = list(species_names) if isinstance(species_names, (list, tuple)) else []
    for idx in range(int(max(nspec, 0))):
        if idx < len(raw):
            label = str(raw[idx] or "").strip()
            names.append(label or f"sp{idx + 1}")
        else:
            names.append(f"sp{idx + 1}")
    return names


def _format_abs_group_label(group_id):
    gid = str(group_id or "").strip()
    if not gid:
        return "absgrp"
    if gid.lower().startswith("absgrp"):
        return gid
    return f"absgrp{gid}"


def _coerce_abs_group_map(group_map, n_abs):
    n_abs = int(max(n_abs, 0))
    if n_abs == 0:
        return onp.zeros((0, 0), dtype=float)
    if group_map is None:
        return onp.eye(n_abs, dtype=float)
    arr = onp.asarray(group_map, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"abs_group_map must be 2D, got shape={arr.shape}")
    if arr.shape[0] != n_abs:
        raise ValueError(f"abs_group_map row mismatch: expected {n_abs}, got {arr.shape[0]}")
    if arr.shape[1] <= 0:
        raise ValueError("abs_group_map must contain at least one absorptivity group.")
    return arr


def _apply_abs_group_map(C_abs, group_map=None):
    C_arr = onp.asarray(C_abs, dtype=float)
    G_map = _coerce_abs_group_map(group_map, C_arr.shape[1])
    return C_arr @ G_map, G_map


def _build_absorptivity_grouping(species_names, nas, abs_groups, nspec=None):
    n_species = int(nspec) if nspec is not None else len(species_names or [])
    species_all = _normalize_species_names(species_names, n_species)
    nas_idx = sorted(
        {
            int(idx)
            for idx in (nas or [])
            if isinstance(idx, (int, onp.integer)) or str(idx).lstrip("-").isdigit()
        }
    )
    nas_idx = [idx for idx in nas_idx if 0 <= idx < n_species]
    nas_idx_set = set(nas_idx)
    abs_indices = [idx for idx in range(n_species) if idx not in nas_idx_set]
    abs_species_names = [species_all[idx] for idx in abs_indices]

    if not abs_species_names:
        return {
            "species_names": species_all,
            "abs_species_names": [],
            "abs_indices": [],
            "group_ids": [],
            "group_labels": [],
            "group_members": [],
            "group_labels_with_members": [],
            "group_map": onp.zeros((0, 0), dtype=float),
            "active": False,
        }

    abs_lookup = {name: pos for pos, name in enumerate(abs_species_names)}
    if not abs_groups:
        ident = onp.eye(len(abs_species_names), dtype=float)
        return {
            "species_names": species_all,
            "abs_species_names": abs_species_names,
            "abs_indices": abs_indices,
            "group_ids": list(abs_species_names),
            "group_labels": list(abs_species_names),
            "group_members": [[name] for name in abs_species_names],
            "group_labels_with_members": list(abs_species_names),
            "group_map": ident,
            "active": False,
        }

    if not isinstance(abs_groups, dict):
        raise ValueError("abs_groups must be a mapping {group_id: [species, ...]}.")

    groups = []
    assigned_species: set[str] = set()
    for raw_group_id, raw_members in abs_groups.items():
        group_id = str(raw_group_id or "").strip()
        if not group_id:
            raise ValueError("abs_groups contains an empty group id.")
        if not isinstance(raw_members, (list, tuple)):
            raise ValueError(f"abs_groups[{group_id!r}] must be a list of species names.")
        members = []
        for raw_name in raw_members:
            species_name = str(raw_name or "").strip()
            if not species_name:
                continue
            if species_name not in abs_lookup:
                if species_name in species_all:
                    raise ValueError(
                        f"Species {species_name!r} is marked as non-absorbent and cannot be used in abs_groups."
                    )
                raise ValueError(f"Unknown species {species_name!r} in abs_groups[{group_id!r}].")
            if species_name in assigned_species:
                raise ValueError(f"Species {species_name!r} is assigned to multiple absorptivity groups.")
            assigned_species.add(species_name)
            members.append(species_name)
        if not members:
            raise ValueError(f"Absorptivity group {group_id!r} is empty.")
        groups.append({"id": group_id, "members": members})

    for species_name in abs_species_names:
        if species_name not in assigned_species:
            groups.append({"id": f"auto_{species_name}", "members": [species_name]})

    group_map = onp.zeros((len(abs_species_names), len(groups)), dtype=float)
    group_ids = []
    group_labels = []
    group_members = []
    group_labels_with_members = []
    for col, group in enumerate(groups):
        members = list(group["members"])
        for species_name in members:
            group_map[abs_lookup[species_name], col] = 1.0
        group_ids.append(str(group["id"]))
        label = _format_abs_group_label(group["id"])
        group_labels.append(label)
        group_members.append(members)
        if len(members) == 1 and members[0] == label:
            group_labels_with_members.append(label)
        else:
            group_labels_with_members.append(f"{label} ({','.join(members)})")

    return {
        "species_names": species_all,
        "abs_species_names": abs_species_names,
        "abs_indices": abs_indices,
        "group_ids": group_ids,
        "group_labels": group_labels,
        "group_members": group_members,
        "group_labels_with_members": group_labels_with_members,
        "group_map": group_map,
        "active": len(groups) < len(abs_species_names),
    }


def _solve_spectral_model(
    C_abs,
    YT,
    *,
    group_map=None,
    eps_solver_mode="soft_penalty",
    mu=1e-2,
    delta_mode="off",
    delta_rel=0.01,
    alpha_smooth=0.0,
    smooth_matrix=None,
    max_iters=300,
):
    C_eff, G_map = _apply_abs_group_map(C_abs, group_map=group_map)
    E_group, lower_bound = _solve_absorptivities(
        C_eff,
        YT,
        eps_solver_mode=eps_solver_mode,
        mu=mu,
        delta_mode=delta_mode,
        delta_rel=delta_rel,
        alpha_smooth=alpha_smooth,
        smooth_matrix=smooth_matrix,
        max_iters=max_iters,
    )
    E_group = onp.asarray(E_group, dtype=float)
    A_species = G_map @ E_group
    yfit = C_eff @ E_group
    return {
        "C_eff": C_eff,
        "group_map": G_map,
        "E_group": E_group,
        "A_species": A_species,
        "yfit": yfit,
        "lower_bound": lower_bound,
    }

def _resolve_render_profile(render_quality: str):
    q = str(render_quality or "preview").strip().lower()
    if q == "full":
        return (8, 6), 100
    if q == "draft":
        return (5.5, 3.5), 72
    # preview default: lighter rendering while keeping legibility.
    return (6.5, 4.2), 82


def generate_figure_base64(
    x,
    y,
    mark,
    ylabel,
    xlabel,
    title,
    *,
    figsize=None,
    dpi=None,
    series_labels=None,
):
    """Generate a matplotlib figure and return as base64 encoded PNG."""
    try:
        # Debug shapes
        x_shape = np.shape(x)
        y_shape = np.shape(y)
        logger.debug("plot %r: x=%s, y=%s", title, x_shape, y_shape)
    except Exception:
        pass

    if figsize is None:
        figsize = (8, 6)
    if dpi is None:
        dpi = 100
    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    
    # Normaliza dimensiones para evitar mismatches (se comporta como en la interfaz anterior)
    x_arr = np.asarray(x).reshape(-1)
    y_arr = np.asarray(y)
    labels = list(series_labels) if isinstance(series_labels, (list, tuple)) else []

    if y_arr.ndim == 1:
        ax.plot(x_arr, y_arr, mark, label=(labels[0] if labels else None))
    elif y_arr.ndim == 2:
        if y_arr.shape[0] == x_arr.shape[0]:
            for i in range(y_arr.shape[1]):
                ax.plot(x_arr, y_arr[:, i], mark, label=(labels[i] if i < len(labels) else None))
        elif y_arr.shape[1] == x_arr.shape[0]:
            for i in range(y_arr.shape[0]):
                ax.plot(x_arr, y_arr[i, :], mark, label=(labels[i] if i < len(labels) else None))
        elif y_arr.size == x_arr.shape[0]:
            ax.plot(x_arr, y_arr.reshape(-1), mark, label=(labels[0] if labels else None))
        else:
            m = min(x_arr.shape[0], y_arr.shape[0])
            ax.plot(x_arr[:m], y_arr[:m, 0], mark, label=(labels[0] if labels else None))
    else:
        ax.plot(x_arr, y_arr.reshape(-1), mark, label=(labels[0] if labels else None))
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if labels:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    
    # Save to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_base64

def generate_figure2_base64(x, y, y2, mark1, mark2, ylabel, xlabel, alpha, title, *, figsize=None, dpi=None):
    """Generate a matplotlib figure with two series and return as base64 encoded PNG."""
    try:
        # Debug shapes
        x_shape = np.shape(x)
        y_shape = np.shape(y)
        y2_shape = np.shape(y2)
        logger.debug("plot2 %r: x=%s, y=%s, y2=%s", title, x_shape, y_shape, y2_shape)
    except Exception:
        pass

    if figsize is None:
        figsize = (8, 6)
    if dpi is None:
        dpi = 100
    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    
    # Plot first series
    if isinstance(y, pd.DataFrame) or isinstance(y, onp.ndarray):
        y_arr = y.values if isinstance(y, pd.DataFrame) else y
        if y_arr.ndim == 2:
            # Check if dimensions match x
            if y_arr.shape[0] != len(x) and y_arr.shape[1] == len(x):
                y_arr = y_arr.T
                
            for i in range(y_arr.shape[1]):
                ax.plot(x, y_arr[:, i], mark1, alpha=alpha)
        else:
            ax.plot(x, y_arr, mark1, alpha=alpha)
    else:
        ax.plot(x, y, mark1, alpha=alpha)
    
    # Plot second series
    if isinstance(y2, pd.DataFrame) or isinstance(y2, onp.ndarray):
        y2_arr = y2.values if isinstance(y2, pd.DataFrame) else y2
        if y2_arr.ndim == 2:
            # Check if dimensions match x
            if y2_arr.shape[0] != len(x) and y2_arr.shape[1] == len(x):
                y2_arr = y2_arr.T
                
            for i in range(y2_arr.shape[1]):
                ax.plot(x, y2_arr[:, i], mark2, alpha=1.0)
        else:
            ax.plot(x, y2_arr, mark2, alpha=1.0)
    else:
        ax.plot(x, y2, mark2, alpha=1.0)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    # Save to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_base64

def build_spectroscopy_plot_data(
    nm,
    Ct,
    ct_columns,
    Y_exp,
    Y_fit,
    residuals,
    C,
    species_labels,
    eigenvalues=None,
    efa_forward=None,
    efa_backward=None,
):
    """
    Construye una estructura genérica de datos para gráficas
    que la GUI pueda usar de forma flexible.
    """
    import numpy as np
    ct_columns = list(ct_columns) if ct_columns is not None else []

    numerics = {
        "nm": np.asarray(nm).tolist(),           # eje espectral
        "Ct": np.asarray(Ct).tolist(),           # concentraciones totales
        "Y_exp": np.asarray(Y_exp).tolist(),     # espectros exp.
        "Y_fit": np.asarray(Y_fit).tolist(),     # espectros ajustados
        "residuals": np.asarray(residuals).tolist(),
        "species_conc": np.asarray(C).tolist(),  # concentraciones de especies
    }

    if eigenvalues is not None:
        numerics["eigenvalues"] = np.asarray(eigenvalues).tolist()
    if efa_forward is not None:
        numerics["efa_forward"] = np.asarray(efa_forward).tolist()
    if efa_backward is not None:
        numerics["efa_backward"] = np.asarray(efa_backward).tolist()

    # Metadatos de ejes
    axes = {
        "wavelength": {
            "id": "wavelength",
            "label": "λ",
            "unit": "nm",
            "values_key": "nm",
        },
        "titration_step": {
            "id": "titration_step",
            "label": "Titration step",
            "unit": None,
            "length": len(numerics["Y_exp"][0]) if numerics["Y_exp"] else 0,
        },
    }

    # Añadir cada columna de Ct como eje posible
    for idx, col in enumerate(ct_columns):
        axes[f"Ct_{idx}"] = {
            "id": f"Ct_{idx}",
            "label": str(col),
            "unit": "M",
            "values_key": "Ct",
            "column": idx,
        }

    # Metadatos de series
    series = {
        "Y_exp": {
            "id": "Y_exp",
            "label": "Experimental spectra",
            "data_key": "Y_exp",
            "dims": ["wavelength", "titration_step"],
        },
        "Y_fit": {
            "id": "Y_fit",
            "label": "Fitted spectra",
            "data_key": "Y_fit",
            "dims": ["wavelength", "titration_step"],
        },
        "residuals": {
            "id": "residuals",
            "label": "Residuals",
            "data_key": "residuals",
            "dims": ["wavelength", "titration_step"],
        },
        "species_conc": {
            "id": "species_conc",
            "label": "Species concentrations",
            "data_key": "species_conc",
            "dims": ["titration_step", "species"],
            "categories": list(species_labels),
        },
    }

    if "eigenvalues" in numerics:
        series["eigenvalues"] = {
            "id": "eigenvalues",
            "label": "EFA eigenvalues",
            "data_key": "eigenvalues",
            "dims": ["eigenvalue_index"],
        }

    presets = [
        {
            "id": "main_spectra",
            "name": "Experimental vs fitted spectra",
            "x_axis": "wavelength",
            "y_series": ["Y_exp", "Y_fit"],
            "vary_along": "titration_step",
        },
        {
            "id": "residuals_vs_wavelength",
            "name": "Residuals vs λ",
            "x_axis": "wavelength",
            "y_series": ["residuals"],
            "vary_along": "titration_step",
        },
        {
            "id": "species_vs_titrant",
            "name": "Species distribution vs titrant",
            "x_axis": "Ct_0",  # luego mapearemos esto al huésped real
            "y_series": ["species_conc"],
            "vary_along": "species",
        },
    ]

    plot_meta = {"axes": axes, "series": series, "presets": presets}
    return {"numerics": numerics, "plot_meta": plot_meta}

def _sanitize_for_json(obj):
    """
    Replace NaN/Inf with None so Starlette's JSON encoder doesn't explode.
    Works recursively on dicts/lists/tuples.
    """
    import math
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ _sanitize_for_json(v) for v in obj ]
    if isinstance(obj, (float, int)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    return obj


def _resolve_ms_workers(ms_runs: int, requested: int | None) -> int:
    cpu_count = int(os.cpu_count() or 1)
    # Leave one CPU free by default to avoid UI starvation.
    max_recommended = max(1, cpu_count - 1)
    if requested is None:
        return max(1, min(int(ms_runs), max_recommended))
    try:
        req = int(requested)
    except (TypeError, ValueError):
        req = max_recommended
    return max(1, min(int(ms_runs), max_recommended, req))


def _objective_cache_key(k_vec) -> bytes:
    arr = onp.asarray(k_vec, dtype=float).ravel()
    return arr.tobytes()


def _new_objective_state(start_k) -> dict:
    return {
        "eval_count": 0,
        "cache": {},
        "cache_max": 8192,
        "error_count": 0,
        "best_rms": float("inf"),
        "best_k": onp.asarray(start_k, dtype=float).ravel().copy(),
    }


def _evaluate_spectro_objective(
    k_vec,
    *,
    solver,
    y_transposed,
    weights_row,
    eps_solver_mode,
    eps_mu,
    delta_mode,
    delta_rel,
    alpha_smooth,
    smooth_matrix,
    abs_group_map=None,
    objective_state: dict,
):
    k_arr = onp.asarray(k_vec, dtype=float).ravel()
    if onp.any(onp.isnan(k_arr)) or onp.any(onp.isinf(k_arr)):
        return 1e50

    key = _objective_cache_key(k_arr)
    cached = objective_state["cache"].get(key)
    if cached is not None:
        return float(cached)

    if objective_state["eval_count"] % 16 == 0:
        _raise_if_cancelled()

    objective_state["eval_count"] += 1
    try:
        C = solver.concentraciones(k_arr)[0]
        if onp.any(onp.isnan(C)) or onp.any(onp.isinf(C)):
            val = 1e50
        else:
            fit = _solve_spectral_model(
                C,
                y_transposed,
                group_map=abs_group_map,
                eps_solver_mode=eps_solver_mode,
                mu=eps_mu,
                delta_mode=delta_mode,
                delta_rel=delta_rel,
                alpha_smooth=alpha_smooth,
                smooth_matrix=smooth_matrix,
                max_iters=300,
            )
            r = fit["yfit"] - y_transposed
            r_w = r * weights_row
            rms = float(np.sqrt(np.mean(np.square(r_w))))
            val = rms if onp.isfinite(rms) else 1e50
    except Exception:
        objective_state["error_count"] += 1
        val = 1e50

    cache = objective_state["cache"]
    if len(cache) >= int(objective_state["cache_max"]):
        cache.clear()
    cache[key] = float(val)

    if onp.isfinite(val) and float(val) < float(objective_state["best_rms"]):
        objective_state["best_rms"] = float(val)
        objective_state["best_k"] = k_arr.copy()
    return float(val)


def _infer_n_complex_from_model(modelo) -> int:
    arr = onp.asarray(modelo, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Invalid model matrix shape: {arr.shape}")
    return max(int(abs(arr.shape[1] - arr.shape[0])), 0)


def _build_equilibrium_solver(
    algorithm,
    c_t_array,
    modelo,
    non_abs_species,
    model_settings,
    *,
    solver_param_transform=None,
    solver_model_settings=None,
):
    c_t_df = pd.DataFrame(onp.asarray(c_t_array, dtype=float))
    effective_model_settings = str(solver_model_settings or model_settings or "Free")
    if algorithm == "Newton-Raphson":
        from ..solvers import NewtonRaphson

        solver = NewtonRaphson(c_t_df, modelo, non_abs_species, effective_model_settings)
    elif algorithm == "Levenberg-Marquardt":
        from ..solvers import LevenbergMarquardt

        solver = LevenbergMarquardt(c_t_df, modelo, non_abs_species, effective_model_settings)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    transform = coerce_param_transform(
        solver_param_transform,
        expected_rows=_infer_n_complex_from_model(modelo),
    )
    if transform is None:
        return solver
    return SolverParamTransformWrapper(solver=solver, transform=transform)


def _run_spectro_single_start(
    *,
    optimizer,
    algorithm,
    c_t_array,
    modelo,
    non_abs_species,
    model_settings,
    solver_param_transform,
    solver_model_settings,
    start_k,
    processed_bounds,
    seed,
    y_transposed,
    weights_row,
    eps_solver_mode,
    eps_mu,
    delta_mode,
    delta_rel,
    alpha_smooth,
    smooth_matrix,
    abs_group_map=None,
):
    solver = _build_equilibrium_solver(
        algorithm,
        c_t_array,
        modelo,
        non_abs_species,
        model_settings,
        solver_param_transform=solver_param_transform,
        solver_model_settings=solver_model_settings,
    )
    state = _new_objective_state(start_k)
    local_best = {"rms": float("inf"), "k": onp.asarray(start_k, dtype=float).ravel().copy()}
    callback_logs = []
    opt_result_obj = None

    def f_m(k_vec):
        return _evaluate_spectro_objective(
            k_vec,
            solver=solver,
            y_transposed=y_transposed,
            weights_row=weights_row,
            eps_solver_mode=eps_solver_mode,
            eps_mu=eps_mu,
            delta_mode=delta_mode,
            delta_rel=delta_rel,
            alpha_smooth=alpha_smooth,
            smooth_matrix=smooth_matrix,
            abs_group_map=abs_group_map,
            objective_state=state,
        )

    def callback_log(xk, convergence=None):
        del convergence
        key = _objective_cache_key(xk)
        val = state["cache"].get(key)
        if val is None:
            return
        if float(val) < float(local_best["rms"]):
            local_best["rms"] = float(val)
            local_best["k"] = onp.asarray(xk, dtype=float).ravel().copy()
            callback_logs.append(
                f"Iter: f(x)={float(val):.6e} | x={[float(xi) for xi in onp.asarray(xk, dtype=float).ravel()]}"
            )

    def _bounds_finite(bounds_list):
        for lb, ub in bounds_list:
            if lb is None or ub is None:
                return False
            if not onp.isfinite(lb) or not onp.isfinite(ub):
                return False
        return True

    try:
        if optimizer == "differential_evolution":
            opt_result_obj = differential_evolution(
                f_m,
                processed_bounds,
                x0=onp.asarray(start_k, dtype=float).ravel(),
                strategy="best1bin",
                maxiter=1000,
                popsize=15,
                tol=0.01,
                mutation=(0.5, 1),
                recombination=0.7,
                init="latinhypercube",
                callback=callback_log,
                seed=seed,
            )
        elif optimizer == "dual_annealing":
            opt_result_obj = dual_annealing(f_m, processed_bounds, seed=seed)
        elif optimizer == "basinhopping":
            minimizer_kwargs = {"method": "L-BFGS-B", "bounds": processed_bounds}
            opt_result_obj = basinhopping(
                f_m,
                onp.asarray(start_k, dtype=float).ravel(),
                niter=100,
                minimizer_kwargs=minimizer_kwargs,
                seed=seed,
            )
        elif optimizer == "global_local":
            if not _bounds_finite(processed_bounds):
                raise ValueError(
                    "Global-local optimization requires all bounds to be finite. Please set Min/Max for each parameter."
                )
            global_res = differential_evolution(
                f_m,
                processed_bounds,
                x0=onp.asarray(start_k, dtype=float).ravel(),
                maxiter=200,
                popsize=10,
                tol=0.05,
                seed=seed,
            )
            opt_result_obj = optimize.minimize(
                f_m,
                global_res.x,
                method="L-BFGS-B",
                bounds=processed_bounds,
                callback=callback_log,
            )
        else:
            opt_result_obj = optimize.minimize(
                f_m,
                onp.asarray(start_k, dtype=float).ravel(),
                method=optimizer,
                bounds=processed_bounds,
                callback=callback_log,
            )
    except Exception as exc:
        if _is_fit_cancelled(exc):
            raise
        return {
            "ok": False,
            "error": str(exc),
            "run_k": onp.asarray(start_k, dtype=float).ravel().tolist(),
            "run_rms": float("inf"),
            "objective_evaluations": int(state["eval_count"]),
            "nfev": 0,
            "success": False,
            "message": str(exc),
            "callback_logs": callback_logs[-5:],
        }

    run_k = onp.asarray(getattr(opt_result_obj, "x", start_k), dtype=float).ravel()
    try:
        run_rms = float(getattr(opt_result_obj, "fun"))
    except Exception:
        run_rms = float("nan")

    if not onp.isfinite(run_rms):
        cached = state["cache"].get(_objective_cache_key(run_k))
        if cached is not None:
            run_rms = float(cached)
    if onp.isfinite(local_best["rms"]) and (not onp.isfinite(run_rms) or float(local_best["rms"]) < float(run_rms)):
        run_rms = float(local_best["rms"])
        run_k = onp.asarray(local_best["k"], dtype=float).ravel().copy()
    if onp.isfinite(state["best_rms"]) and (not onp.isfinite(run_rms) or float(state["best_rms"]) < float(run_rms)):
        run_rms = float(state["best_rms"])
        run_k = onp.asarray(state["best_k"], dtype=float).ravel().copy()

    return {
        "ok": True,
        "error": "",
        "run_k": run_k.tolist(),
        "run_rms": float(run_rms if onp.isfinite(run_rms) else float("inf")),
        "objective_evaluations": int(state["eval_count"]),
        "nfev": int(getattr(opt_result_obj, "nfev", 0) or 0),
        "success": bool(getattr(opt_result_obj, "success", onp.isfinite(run_rms))),
        "message": str(getattr(opt_result_obj, "message", "")),
        "callback_logs": callback_logs[-5:],
    }


def _run_spectro_single_start_worker(payload: dict):
    return _run_spectro_single_start(**payload)

def process_spectroscopy_data(
    file_path,
    spectra_sheet,
    conc_sheet,
    column_names,
    receptor_label,
    guest_label,
    efa_enabled,
    efa_eigenvalues,
    modelo,
    non_abs_species,
    species_names,
    abs_groups,
    solver_param_transform,
    solver_model_settings,
    param_names,
    algorithm,
    model_settings,
    optimizer,
    initial_k,
    bounds,
    channels_raw=None,
    channels_mode=None,
    channels_resolved=None,
    show_stability_diagnostics: bool = False,
    multi_start_runs: int = 1,
    multi_start_seeds=None,
    baseline_mode: str = "off",
    baseline_start: float = 450.0,
    baseline_end: float = 600.0,
    baseline_auto_quantile: float = 0.20,
    baseline_apply_per_spectrum: bool = True,
    weighting_mode: str = "none",
    weighting_eps: float = 1e-12,
    weighting_power: float = 1.0,
    weighting_normalize: bool = True,
    eps_solver_mode: str = "soft_penalty",
    eps_mu: float = 1e-2,
    delta_mode: str = "off",
    delta_rel: float = 0.01,
    alpha_smooth: float = 0.0,
    render_graphs: bool = True,
    render_quality: str = "preview",
    skip_optimization: bool = False,
    preset_k=None,
    multi_start_parallel: bool = False,
    multi_start_max_workers: int | None = None,
):
    """
    Main processing function.
    Returns dict with results and optional base64-encoded graphs.
    """
    log_lines = []
    t_total_start = time.perf_counter()
    stage_times = {
        "parse_s": 0.0,
        "solver_s": 0.0,
        "optimization_s": 0.0,
        "postprocess_s": 0.0,
        "graphs_s": 0.0,
        "total_s": 0.0,
    }

    def log(msg: str):
        try:
            log_lines.append(str(msg))
        except Exception:
            pass
        log_progress(str(msg))

    log_progress("Iniciando procesamiento...")
    t_parse_start = time.perf_counter()

    baseline_mode = _normalize_mode(baseline_mode, {"off", "range", "auto"}, "off")
    weighting_mode = _normalize_mode(weighting_mode, {"none", "std", "max"}, "none")
    eps_solver_mode = _normalize_mode(eps_solver_mode, {"soft_penalty", "soft_bound", "nnls_hard"}, "soft_penalty")
    delta_mode = _normalize_mode(delta_mode, {"off", "relative"}, "off")
    try:
        baseline_start = float(baseline_start)
    except (TypeError, ValueError):
        baseline_start = 450.0
    try:
        baseline_end = float(baseline_end)
    except (TypeError, ValueError):
        baseline_end = 600.0
    try:
        baseline_auto_quantile = float(baseline_auto_quantile)
    except (TypeError, ValueError):
        baseline_auto_quantile = 0.20
    try:
        weighting_eps = float(weighting_eps)
    except (TypeError, ValueError):
        weighting_eps = 1e-12
    try:
        weighting_power = float(weighting_power)
    except (TypeError, ValueError):
        weighting_power = 1.0
    try:
        eps_mu = float(eps_mu)
    except (TypeError, ValueError):
        eps_mu = 1e-2
    try:
        delta_rel = float(delta_rel)
    except (TypeError, ValueError):
        delta_rel = 0.01
    try:
        alpha_smooth = float(alpha_smooth)
    except (TypeError, ValueError):
        alpha_smooth = 0.0
    render_quality = str(render_quality or "preview").strip().lower()
    if render_quality not in {"preview", "full", "draft"}:
        render_quality = "preview"
    render_figsize, render_dpi = _resolve_render_profile(render_quality)
    
    # Read Excel data
    spec = pd.read_excel(file_path, spectra_sheet, header=0, index_col=0)
    axis_numeric = pd.to_numeric(spec.index, errors="coerce")
    spec = spec.loc[axis_numeric.notna()].copy()
    spec.index = axis_numeric[axis_numeric.notna()].astype(float)

    channels_total = int(len(spec))
    warnings_list = []

    resolved = channels_resolved or []
    if isinstance(resolved, (list, tuple)) and len(resolved) > 0:
        resolved = [float(x) for x in resolved]
        axis_vals = spec.index.to_numpy(dtype=float)
        mask = onp.zeros_like(axis_vals, dtype=bool)
        for target in resolved:
            mask |= onp.isclose(axis_vals, float(target), rtol=0.0, atol=1e-8)
        if not mask.any():
            raise ValueError("No matching channels found in Spectra axis for the provided channels_resolved.")
        spec = spec.iloc[mask].copy()

    # Guardrail: EFA needs at least 2 spectral channels
    if efa_enabled and int(len(spec)) < 2:
        efa_enabled = False
        warnings_list.append("EFA disabled: requires at least 2 channels.")

    nm = spec.index.to_numpy()
    
    concentracion = pd.read_excel(file_path, conc_sheet, header=0)
    C_T = concentracion[column_names].to_numpy()
    n_comp = len(column_names)
    model_raw = np.asarray(modelo, dtype=float) if modelo is not None else np.zeros((0, 0))
    model_cols = model_raw.shape[1] if model_raw.ndim == 2 else 0
    model_rows = model_raw.shape[0] if model_raw.ndim == 2 else 0
    
    # Create column index mapping
    column_indices_in_C_T = {name: index for index, name in enumerate(column_names)}
    
    receptor_index_in_C_T = column_indices_in_C_T.get(receptor_label, -1)
    guest_index_in_C_T = column_indices_in_C_T.get(guest_label, -1)
    
    G = None
    H = None
    if guest_index_in_C_T != -1:
        G = C_T[:, guest_index_in_C_T]
    if receptor_index_in_C_T != -1:
        H = C_T[:, receptor_index_in_C_T]

    guest_missing = (guest_label is None) or (str(guest_label).strip() == "") or (G is None)
    expand_dummy_guest = guest_missing and model_cols == 1 and model_rows > 0
    if expand_dummy_guest:
        dummy_name = "__DUMMY_GUEST__"
        if dummy_name not in concentracion.columns:
            concentracion[dummy_name] = 0.0
            column_names = list(column_names) + [dummy_name]
            C_T = concentracion[column_names].to_numpy()
            n_comp = len(column_names)
        G = None
        if model_rows > 0:
            dummy_col = np.zeros((model_rows, 1), dtype=float)
            model_raw = np.concatenate([model_raw, dummy_col], axis=1)
            dummy_row = np.zeros((1, model_cols + 1), dtype=float)
            dummy_row[0, model_cols] = 1.0
            model_raw = np.concatenate(
                [model_raw[:model_cols, :], dummy_row, model_raw[model_cols:, :]], axis=0
            )
            modelo = model_raw.tolist()
        if non_abs_species:
            non_abs_species = [
                (int(idx) + 1) if int(idx) >= model_cols else int(idx)
                for idx in non_abs_species
            ]
    guest_label_display = str(receptor_label or "") if guest_missing else str(guest_label or "")
    n_comp_plot = n_comp - 1 if expand_dummy_guest else n_comp
    
    nc = len(C_T)
    nw = len(spec)
    channels_used = int(nw)

    spec_arr = spec.to_numpy(dtype=float)
    # Fit matrix convention: A_exp = (n_points × n_channels).
    A_exp_raw = spec_arr.T
    A_exp_used = A_exp_raw
    efa_ev_used = None
    efa_max_ev = int(min(A_exp_raw.shape)) if A_exp_raw.ndim == 2 else 0
    if efa_enabled:
        if efa_max_ev < 1:
            efa_enabled = False
            warnings_list.append("EFA disabled: empty spectra matrix.")
        else:
            A_exp_used, efa_ev_used, efa_max_ev = apply_efa_svd(A_exp_raw, efa_eigenvalues)
            log_progress(
                "EFA SVD: Y_raw shape=%s, Y_used shape=%s, ev_requested=%s, ev_used=%s, max_ev=%s"
                % (A_exp_raw.shape, A_exp_used.shape, efa_eigenvalues, efa_ev_used, efa_max_ev)
            )
            assert A_exp_used.shape == A_exp_raw.shape, "EFA shape mismatch"
            if A_exp_used.shape != A_exp_raw.shape:
                raise ValueError(
                    f"EFA shape mismatch: raw={A_exp_raw.shape}, used={A_exp_used.shape}"
                )

    Y_used_matrix_raw = onp.asarray(A_exp_used, dtype=float)  # (m_points x n_lambda)
    Y_used_matrix_corr, baseline_vals, baseline_meta = baseline_correct(
        Y_used_matrix_raw,
        nm,
        mode=baseline_mode,
        start=baseline_start,
        end=baseline_end,
        auto_quantile=baseline_auto_quantile,
        apply_per_spectrum=baseline_apply_per_spectrum,
    )
    if baseline_meta.get("warning"):
        warnings_list.append(str(baseline_meta["warning"]))

    baseline_mode_used = str(baseline_meta.get("mode", baseline_mode))
    if baseline_mode_used == "range":
        log_progress(
            "Baseline: range %.3g-%.3g nm (%d channels)"
            % (
                float(baseline_meta.get("start", baseline_start)),
                float(baseline_meta.get("end", baseline_end)),
                int(baseline_meta.get("n_baseline_channels", 0)),
            )
        )
    elif baseline_mode_used == "auto":
        log_progress(
            "Baseline: auto (quantile=%.3g, channels=%d)"
            % (
                float(baseline_meta.get("quantile", baseline_auto_quantile)),
                int(baseline_meta.get("n_baseline_channels", 0)),
            )
        )
    else:
        log_progress("Baseline: off")

    weights_per_lambda, weighting_meta = compute_spectral_weights(
        Y_used_matrix_corr,
        mode=weighting_mode,
        eps=weighting_eps,
        power=weighting_power,
        normalize=weighting_normalize,
    )
    if weighting_meta.get("warning"):
        warnings_list.append(str(weighting_meta["warning"]))
        log_progress(str(weighting_meta["warning"]))
    log_progress(
        "Weighting: %s (min=%.3g, max=%.3g, power=%.3g)"
        % (
            str(weighting_meta.get("mode", weighting_mode)),
            float(weighting_meta.get("w_min", 0.0)),
            float(weighting_meta.get("w_max", 1.0)),
            float(weighting_meta.get("power", weighting_power)),
        )
    )

    Y = np.asarray(Y_used_matrix_corr.T)
    Y_raw = np.asarray(Y_used_matrix_raw.T)
    weights_lambda = np.asarray(weights_per_lambda, dtype=float)
    weights_row = weights_lambda[None, :]
    n_lambda = int(Y.shape[0])
    smooth_matrix = None
    if alpha_smooth > 0.0:
        smooth_matrix = _build_smoothness_laplacian(n_lambda)
        if smooth_matrix.size == 0 or n_lambda < 3:
            warnings_list.append("alpha_smooth requested but fewer than 3 channels are available; smoothing disabled.")
            smooth_matrix = None
            alpha_smooth = 0.0
        else:
            smax_l = float(onp.linalg.svd(smooth_matrix, compute_uv=False).max())
            log_progress(
                "Smoothness: alpha=%.3g, n_lambda=%d, ||L||2=%.3g"
                % (alpha_smooth, n_lambda, smax_l)
            )
    
    graphs = {
        "fit": "",
        "concentrations": "",
        "absorptivities": "",
        "eigenvalues": "",
        "efa": "",
        "residuals": "",
    }
    
    # SVD/EFA function
    def SVD_EFA(spec, ev_used):
        spec_arr = onp.asarray(spec, dtype=float)
        n_channels = spec_arr.shape[0]
        n_points = spec_arr.shape[1] if spec_arr.ndim > 1 else 0
        n_max = int(min(n_points, n_channels))

        u, s_full, vh = onp.linalg.svd(spec_arr, full_matrices=False)

        if n_max < 1:
            EV = 0
        elif ev_used is None:
            EV = n_max
        else:
            try:
                EV = int(ev_used)
            except (TypeError, ValueError):
                EV = n_max
            if EV <= 0:
                EV = n_max
            EV = max(1, min(EV, n_max))
        log_progress(f"Eigenvalues used: {EV}")
        logger.debug(
            "EFA SVD shapes: spec=%s u=%s s=%s vh=%s EV=%s",
            spec_arr.shape,
            u.shape,
            s_full.shape,
            vh.shape,
            EV,
        )

        forward = np.full((n_points, EV), np.nan)
        backward = np.full((n_points, EV), np.nan)
        for i in range(1, n_points + 1):
            s_sub = onp.linalg.svd(spec_arr.T[:i, :], compute_uv=False)
            m = min(EV, s_sub.size)
            forward[i - 1, :m] = s_sub[:m]
        for i in range(n_points):
            s_sub = onp.linalg.svd(spec_arr.T[i:, :], compute_uv=False)
            m = min(EV, s_sub.size)
            backward[i, :m] = s_sub[:m]

        if EV <= 0:
            Y = spec_arr
        else:
            Y = (u[:, :EV] * s_full[:EV]) @ vh[:EV, :]
        return Y, EV, s_full, forward, backward
    
    eigenvalues = None
    efa_forward = None
    efa_backward = None
    if efa_enabled:
        _, EV, eigenvalues, efa_forward, efa_backward = SVD_EFA(spec_arr, efa_ev_used)
    else:
        EV = nc
    
    C_T_df = pd.DataFrame(C_T)
    modelo = np.array(modelo).T if isinstance(modelo, list) else np.array(modelo)
    nas = non_abs_species

    # ---- Inicialización de parámetros y límites (replica flujo anterior) ----
    def _safe_float_list(seq):
        vals = []
        for v in seq:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        return vals

    k = np.asarray(_safe_float_list(initial_k), dtype=float)
    solver_param_transform_arr = coerce_param_transform(
        solver_param_transform,
        expected_rows=_infer_n_complex_from_model(modelo),
    )
    solver_model_settings = str(solver_model_settings or model_settings or "Free")
    if solver_param_transform_arr is not None and k.size:
        if int(k.size) != int(solver_param_transform_arr.shape[1]):
            raise ValueError(
                "Equation-parameter mapping mismatch: "
                f"expected {solver_param_transform_arr.shape[1]} constants, got {k.size}."
            )
    param_names = list(param_names or [])
    if len(param_names) != int(k.size):
        param_names = [f"K{i+1}" for i in range(int(k.size))]
    if solver_param_transform_arr is not None:
        log_progress(
            "Equation parameter mapping active: %d reaction constants -> %d solver betas."
            % (int(solver_param_transform_arr.shape[1]), int(solver_param_transform_arr.shape[0]))
        )

    def _to_bound(val, default):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return default
        if np.isnan(v):
            return default
        return v

    n_params = len(k) if len(k) else len(bounds)
    processed_bounds = []
    for i in range(n_params):
        raw = bounds[i] if i < len(bounds) else None
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            min_raw, max_raw = raw[0], raw[1]
        else:
            min_raw, max_raw = None, None

        # Vacíos -> ±inf (sin introducir rangos artificiales)
        min_val = _to_bound(min_raw, -np.inf)
        max_val = _to_bound(max_raw, np.inf)
        processed_bounds.append((min_val, max_val))
    stage_times["parse_s"] = time.perf_counter() - t_parse_start
    t_solver_start = time.perf_counter()
    
    # Select algorithm
    res = _build_equilibrium_solver(
        algorithm,
        C_T_df,
        modelo,
        nas,
        model_settings,
        solver_param_transform=solver_param_transform_arr,
        solver_model_settings=solver_model_settings,
    )

    species_all = _normalize_species_names(species_names, int(getattr(res, "nspec", 0) or 0))
    abs_grouping = _build_absorptivity_grouping(
        species_all,
        nas,
        abs_groups,
        nspec=int(getattr(res, "nspec", 0) or 0),
    )
    abs_group_map = abs_grouping["group_map"]
    abs_species_labels = list(abs_grouping["abs_species_names"])
    group_labels = list(abs_grouping["group_labels"])
    group_members = [list(members) for members in abs_grouping["group_members"]]
    group_summary = "; ".join(
        f"{label}: {','.join(members)}"
        for label, members in zip(group_labels, group_members)
    )
    grouped_abs_active = bool(abs_grouping["active"])
    if group_labels:
        log_progress(f"Absorptivity groups: {len(group_labels)} ({group_summary})")
    else:
        log_progress("Absorptivity groups: 0")
    if grouped_abs_active:
        log_progress("Grouped absorptivities active.")
    
    if eps_solver_mode == "soft_bound" and delta_mode == "off":
        delta_mode = "relative"
    log(
        "Epsilon solver: %s (mu=%.3g, delta_mode=%s, delta_rel=%.3g, alpha_smooth=%.3g)"
        % (eps_solver_mode, eps_mu, delta_mode, delta_rel, alpha_smooth)
    )
    try:
        C_preview = onp.asarray(res.concentraciones(k)[0], dtype=float)
        preview_fit = _solve_spectral_model(
            C_preview,
            onp.asarray(Y.T, dtype=float),
            group_map=abs_group_map,
            eps_solver_mode=eps_solver_mode,
            mu=eps_mu,
            delta_mode=delta_mode,
            delta_rel=delta_rel,
            alpha_smooth=alpha_smooth,
            smooth_matrix=smooth_matrix,
            max_iters=300,
        )
        lb_preview = preview_fit.get("lower_bound")
    except Exception:
        lb_preview = None
    if lb_preview is not None:
        lb_arr = onp.asarray(lb_preview, dtype=float).ravel()
        log_progress(
            "Soft lower bound preview: min=%.3g, max=%.3g"
            % (float(onp.min(lb_arr)), float(onp.max(lb_arr)))
        )
    else:
        log_progress("Soft lower bound preview: off")
    if alpha_smooth > 0:
        log_progress("Smoothness penalty is active.")
    stage_times["solver_s"] = time.perf_counter() - t_solver_start

    # Objective helpers
    def f_m2(k):
        C = res.concentraciones(k)[0]
        fit = _solve_spectral_model(
            C,
            Y.T,
            group_map=abs_group_map,
            eps_solver_mode=eps_solver_mode,
            mu=eps_mu,
            delta_mode=delta_mode,
            delta_rel=delta_rel,
            alpha_smooth=alpha_smooth,
            smooth_matrix=smooth_matrix,
            max_iters=300,
        )
        r = fit["yfit"] - onp.asarray(Y.T, dtype=float)
        r_w = r * weights_row
        rms = np.sqrt(np.mean(np.square(r_w)))
        return rms, r

    # Best-so-far tracking
    best_result = {"rms": float("inf"), "k": onp.copy(k)}
    optimizer_summary = {"success": False, "message": "", "nfev": 0}
    objective_eval_count = 0

    def _random_start(base, bounds_list, rng):
        start = onp.empty_like(onp.asarray(base, dtype=float), dtype=float)
        for ii, (lb, ub) in enumerate(bounds_list):
            lb_val = lb if lb is not None and onp.isfinite(lb) else None
            ub_val = ub if ub is not None and onp.isfinite(ub) else None
            if lb_val is not None and ub_val is not None:
                start[ii] = rng.uniform(lb_val, ub_val)
            else:
                start[ii] = onp.asarray(base, dtype=float)[ii] + rng.uniform(-1.0, 1.0)
        return start

    # Optimization
    log(f"Optimizer: {optimizer}")
    log(f"Bounds (procesados): {processed_bounds}")  # keep visible for debugging powell with ±inf

    if optimizer in ("differential_evolution", "dual_annealing", "global_local"):
        # differential_evolution/dual_annealing/global_local requieren límites finitos
        for (min_val, max_val) in processed_bounds:
            if np.isinf(min_val) or np.isinf(max_val):
                if optimizer == "dual_annealing":
                    msg = "Dual annealing requires all bounds to be finite. Please set Min/Max for each parameter."
                elif optimizer == "global_local":
                    msg = "Global-local optimization requires all bounds to be finite. Please set Min/Max for each parameter."
                else:
                    msg = "Differential evolution requires all bounds to be finite. Please set Min/Max for each parameter."
                log_progress(msg)
                raise ValueError(msg)

    ms_runs = max(int(multi_start_runs), 1)
    seed_list = None
    if multi_start_seeds is not None:
        try:
            seeds = [int(s) for s in multi_start_seeds]
        except Exception:
            seeds = []
        if seeds and len(seeds) == ms_runs:
            seed_list = seeds

    start_points = []
    for ms_iter in range(ms_runs):
        seed = seed_list[ms_iter] if seed_list else None
        if ms_iter == 0:
            start_k = onp.asarray(k, dtype=float)
        else:
            rng = onp.random.default_rng(seed)
            start_k = _random_start(k, processed_bounds, rng)
        start_points.append((start_k, seed))
    if skip_optimization:
        preset_arr = onp.asarray(preset_k if preset_k is not None else k, dtype=float).ravel()
        if preset_arr.size != onp.asarray(k, dtype=float).ravel().size:
            preset_arr = onp.asarray(k, dtype=float).ravel().copy()
            warnings_list.append("preset_k size mismatch; using initial_k for deferred render.")
        best_result["k"] = preset_arr.copy()
        best_result["rms"] = float("inf")
        optimizer_summary["success"] = True
        optimizer_summary["message"] = "Optimization skipped (preset_k)."
        optimizer_summary["nfev"] = 0
        start_points = []
        ms_runs = 0
        log_progress("Optimization skipped; using preset_k.")

    t_opt_start = time.perf_counter()
    y_transposed = onp.asarray(Y.T, dtype=float)
    common_payload = {
        "optimizer": optimizer,
        "algorithm": algorithm,
        "c_t_array": onp.asarray(C_T, dtype=float),
        "modelo": onp.asarray(modelo, dtype=float),
        "non_abs_species": list(nas),
        "model_settings": model_settings,
        "solver_param_transform": (
            None if solver_param_transform_arr is None else onp.asarray(solver_param_transform_arr, dtype=float)
        ),
        "solver_model_settings": solver_model_settings,
        "processed_bounds": list(processed_bounds),
        "y_transposed": y_transposed,
        "weights_row": onp.asarray(weights_row, dtype=float),
        "eps_solver_mode": eps_solver_mode,
        "eps_mu": float(eps_mu),
        "delta_mode": delta_mode,
        "delta_rel": float(delta_rel),
        "alpha_smooth": float(alpha_smooth),
        "smooth_matrix": onp.asarray(smooth_matrix, dtype=float) if smooth_matrix is not None else None,
        "abs_group_map": onp.asarray(abs_group_map, dtype=float) if abs_group_map is not None else None,
    }

    def _consume_run(ms_idx, run_result):
        nonlocal objective_eval_count
        objective_eval_count += int(run_result.get("objective_evaluations", 0) or 0)
        if not run_result.get("ok", False):
            log_progress(f"[DEBUG] optimization run {ms_idx+1}/{ms_runs} failed: {run_result.get('error', '')}")
        else:
            for line in run_result.get("callback_logs", []):
                log(line)
            run_rms = float(run_result.get("run_rms", float("inf")))
            run_k = onp.asarray(run_result.get("run_k", []), dtype=float).ravel()
            if onp.isfinite(run_rms) and (run_k.size > 0):
                if run_rms < best_result["rms"]:
                    best_result["rms"] = run_rms
                    best_result["k"] = run_k.copy()
                    optimizer_summary["success"] = bool(run_result.get("success", True))
                    optimizer_summary["message"] = str(run_result.get("message", ""))
                    optimizer_summary["nfev"] = int(run_result.get("nfev", 0) or 0)
        if ms_runs > 1:
            best_msg = f"{best_result['rms']:.6e}" if onp.isfinite(best_result["rms"]) else "inf"
            log_progress(f"[MS] run {ms_idx+1}/{ms_runs} best={best_msg}")

    parallel_enabled = bool(multi_start_parallel and ms_runs > 1)
    if parallel_enabled:
        workers = _resolve_ms_workers(ms_runs, multi_start_max_workers)
        log_progress(f"[MS] parallel enabled with {workers} workers ({ms_runs} runs).")
        payloads = []
        for start_k, seed in start_points:
            payload = dict(common_payload)
            payload["start_k"] = onp.asarray(start_k, dtype=float).ravel()
            payload["seed"] = seed
            payloads.append(payload)
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_run_spectro_single_start_worker, payload): idx
                    for idx, payload in enumerate(payloads)
                }
                pending = set(futures.keys())
                while pending:
                    _raise_if_cancelled()
                    done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                    if not done:
                        continue
                    for fut in done:
                        ms_idx = futures[fut]
                        try:
                            run_result = fut.result()
                        except Exception as run_exc:
                            run_result = {
                                "ok": False,
                                "error": str(run_exc),
                                "objective_evaluations": 0,
                            }
                        _consume_run(ms_idx, run_result)
        except Exception as parallel_exc:
            if _is_fit_cancelled(parallel_exc):
                raise
            log_progress(f"[MS] parallel fallback to sequential: {parallel_exc}")
            parallel_enabled = False

    if not parallel_enabled:
        for ms_idx, (start_k, seed) in enumerate(start_points):
            payload = dict(common_payload)
            payload["start_k"] = onp.asarray(start_k, dtype=float).ravel()
            payload["seed"] = seed
            run_result = _run_spectro_single_start(**payload)
            _consume_run(ms_idx, run_result)

    stage_times["optimization_s"] = time.perf_counter() - t_opt_start
    k = np.ravel(best_result["k"])
    k_solver = expand_solver_params(k, solver_param_transform_arr)

    log("Optimización completada")
    log_progress(f"Objective evaluations (f_m): {int(objective_eval_count)}")
    t_post_start = time.perf_counter()
    
    # Compute errors
    k_names = list(param_names) if param_names else [f"K{i+1}" for i in range(len(k))]
    metrics = compute_errors_spectro_varpro(
        k=k, res=res, Y=Y, modelo=modelo, nas=nas,
        rcond=1e-10, use_projector=True,
        param_names=k_names,
        weights=weights_per_lambda,
        group_map=abs_group_map,
        param_transform=solver_param_transform_arr,
    )
    
    SE_log10K = metrics["SE_log10K"]
    SE_K = metrics["SE_K"]
    percK = metrics["percK"]
    rms = metrics["RMS"]
    covfit = metrics["s2"]
    A = metrics["A"]
    A_species = metrics.get("A_species")
    yfit = metrics["yfit"]
    stability_diag = metrics.get("stability_diag", {})

    
    C, Co = res.concentraciones(k)
    species_labels = abs_species_labels if C is not None else []
    final_fit = _solve_spectral_model(
        C,
        Y.T,
        group_map=abs_group_map,
        eps_solver_mode=eps_solver_mode,
        mu=eps_mu,
        delta_mode=delta_mode,
        delta_rel=delta_rel,
        alpha_smooth=alpha_smooth,
        smooth_matrix=smooth_matrix,
        max_iters=300,
    )
    A_solver = final_fit["E_group"]
    A_species_solver = final_fit["A_species"]
    y_cal = final_fit["yfit"]
    lower_bound_final = final_fit["lower_bound"]
    if lower_bound_final is not None:
        lb_fin = onp.asarray(lower_bound_final, dtype=float).ravel()
        log_progress(
            "Soft lower bound final: min=%.3g, max=%.3g"
            % (float(onp.min(lb_fin)), float(onp.max(lb_fin)))
        )
    else:
        log_progress("Soft lower bound final: off")
    
    # Decide "main plot" mode based on channels actually used
    channels_used_values = nm.tolist() if hasattr(nm, "tolist") else list(nm)
    k_used = int(len(channels_used_values))
    plot_mode = "isotherms" if k_used <= 10 else "spectra"
    x_titrant = G if G is not None else H
    if x_titrant is None:
        plot_mode = "spectra"

    # Generate concentration and spectra plots
    t_graph_start = time.perf_counter()
    if render_graphs:
        if efa_enabled and eigenvalues is not None:
            eig_idx = range(len(onp.asarray(eigenvalues)))
            graphs["eigenvalues"] = generate_figure_base64(
                eig_idx,
                np.log10(onp.asarray(eigenvalues, dtype=float)),
                "o",
                "log(EV)",
                "# de autovalores",
                "Eigenvalues",
                figsize=render_figsize,
                dpi=render_dpi,
            )
            if G is not None and efa_forward is not None and efa_backward is not None:
                graphs["efa"] = generate_figure2_base64(
                    G,
                    np.log10(onp.asarray(efa_forward, dtype=float)),
                    np.log10(onp.asarray(efa_backward, dtype=float)),
                    "k-o",
                    "b:o",
                    "log(EV)",
                    "[G], M",
                    1,
                    "EFA",
                    figsize=render_figsize,
                    dpi=render_dpi,
                )
        if n_comp_plot == 1 and H is not None:
            graphs["concentrations"] = generate_figure_base64(
                H,
                C,
                ":o",
                "[Especies], M",
                "[H], M",
                "Perfil de concentraciones",
                figsize=render_figsize,
                dpi=render_dpi,
            )

            A_plot = A_solver
            eps_mark = "-" if k_used > 1 else "o-"
            graphs["absorptivities"] = generate_figure_base64(
                nm,
                A_plot.T,
                eps_mark,
                "Epsilon (u. a.)",
                "$\\lambda$ (nm)",
                "Absortividades molares",
                figsize=render_figsize,
                dpi=render_dpi,
                series_labels=abs_grouping["group_labels_with_members"],
            )
            if plot_mode == "isotherms":
                idx = slice(0, min(k_used, 10))
                graphs["fit"] = generate_figure2_base64(
                    x_titrant,
                    Y.T[:, idx],
                    y_cal[:, idx],
                    "ko",
                    ":",
                    "Y observada (u. a.)",
                    "[X], M",
                    1,
                    "Ajuste",
                    figsize=render_figsize,
                    dpi=render_dpi,
                )
            else:
                graphs["fit"] = generate_figure2_base64(
                    nm,
                    Y,
                    y_cal.T,
                    "-k",
                    "k:",
                    "Y observada (u. a.)",
                    "$\\lambda$ (nm)",
                    0.5,
                    "Ajuste",
                    figsize=render_figsize,
                    dpi=render_dpi,
                )
        elif G is not None:
            graphs["concentrations"] = generate_figure_base64(
                G,
                C,
                ":o",
                "[Species], M",
                "[G], M",
                "Perfil de concentraciones",
                figsize=render_figsize,
                dpi=render_dpi,
            )

            A_plot = A_solver

            if plot_mode == "isotherms":
                eps_mark = "-" if k_used > 1 else "o-"
                graphs["absorptivities"] = generate_figure_base64(
                    nm,
                    A_plot.T,
                    eps_mark,
                    "Epsilon (u. a.)",
                    "$\\lambda$ (nm)",
                    "Absortividades molares",
                    figsize=render_figsize,
                    dpi=render_dpi,
                    series_labels=abs_grouping["group_labels_with_members"],
                )
                idx = slice(0, min(k_used, 10))
                graphs["fit"] = generate_figure2_base64(
                    G,
                    Y.T[:, idx],
                    y_cal[:, idx],
                    "ko",
                    ":",
                    "Y observada (u. a.)",
                    "[X], M",
                    1,
                    "Ajuste",
                    figsize=render_figsize,
                    dpi=render_dpi,
                )
            else:
                graphs["absorptivities"] = generate_figure_base64(
                    nm,
                    A_plot.T,
                    "-",
                    "Epsilon (u. a.)",
                    "$\\lambda$ (nm)",
                    "Absortividades molares",
                    figsize=render_figsize,
                    dpi=render_dpi,
                    series_labels=abs_grouping["group_labels_with_members"],
                )
                graphs["fit"] = generate_figure2_base64(
                    nm,
                    Y,
                    y_cal.T,
                    "-k",
                    "k:",
                    "Y observada (u. a.)",
                    "$\\lambda$ (nm)",
                    0.5,
                    "Ajuste",
                    figsize=render_figsize,
                    dpi=render_dpi,
                )
    else:
        log_progress("Graph rendering skipped (render_graphs=False).")
    stage_times["graphs_s"] = time.perf_counter() - t_graph_start
    
    ssq, r0 = f_m2(k)
    residuals_matrix = onp.asarray(r0).T if r0 is not None else []
    Y_exp = onp.asarray(Y)
    Y_fit_arr = onp.asarray(yfit) if yfit is not None else []
    
    # Statistics
    Yvec = Y.T
    SS_res = float(np.sum(r0**2))
    SS_tot = float(np.sum((Yvec - np.mean(Yvec))**2))
    lof = 0.0 if SS_tot <= 1e-30 else 100.0 * SS_res / SS_tot
    
    MAE = np.mean(abs(r0))
    if H is not None:
        dif_en_ct = round(max(100 - (np.sum(C, 1) * 100 / max(H))), 2)
    else:
        dif_en_ct = 0.0

    # Tabla formateada para resultados (alineada como en la interfaz anterior)
    results_text = format_results_table(k, SE_log10K, percK, rms, covfit, lof=lof, param_names=k_names)

    # Stability Diagnostics
    if stability_diag:
        status = stability_diag.get("status")
        summary = stability_diag.get("diag_summary", "")
        full = stability_diag.get("diag_full", "")

        if status == "critical":
            results_text += f"\n\n>>> CRITICAL WARNING: Ill-conditioned system ({summary}).\nParameters might not be identifiable. Review correlations/model."
        elif status == "warn":
            results_text += f"\n\n>>> WARNING: Poor conditioning ({summary}).\nHigh correlations might be present."
        
        if show_stability_diagnostics:
            results_text += f"\n\n{full}"

    derived_noncoop = None
    if str(model_settings) in ("Non-cooperative", "Statistical") and np.asarray(k).size == 1:
        try:
            derived = noncoop_derived_from_logK1(
                onp.asarray(modelo, dtype=float),
                float(np.asarray(k).ravel()[0]),
            )
            derived_noncoop = {
                "N": int(derived["N"]),
                "logK_by_j": onp.asarray(derived["logK_by_j"], dtype=float).tolist(),
            }

            lines = ["", "Derived (Non-cooperative):", "j | log10(Kj)"]
            for j in range(1, int(derived["N"]) + 1):
                lines.append(
                    f"{j} | {derived['logK_by_j'][j-1]:.6g}"
                )
            results_text += "\n" + "\n".join(lines)
        except Exception:
            derived_noncoop = None

    # Estadísticas sin duplicar RMS/s² (ya aparecen en la tabla)
    extra_stats = [
        f"LOF: {lof:.2e} %",
        f"MAE: {MAE:.2e}",
        f"Diferencia en C total (%): {dif_en_ct:.2f}",
        f"Eigenvalues: {EV}",
        f"Optimizer: {optimizer}",
        f"Baseline: {baseline_mode_used}",
        f"Weighting: {weighting_mode}",
        f"Epsilon solver: {eps_solver_mode}",
        f"Absorptivity groups: {len(group_labels)}",
    ]
    if group_summary:
        extra_stats.append(f"Abs groups map: {group_summary}")
    results_text += "\n\nEstadísticas:\n" + "\n".join(extra_stats)
    
    # Export payload to mimic previous save_results (DataFrames per sheet)
    # Preparar payload de exportación (imitando el flujo anterior)
    A_export = None
    A_solver_export = None
    A_species_export = None
    A_species_solver_export = None
    nm_list = nm.tolist() if hasattr(nm, "tolist") else []

    def _orient_abs_matrix(matrix):
        if matrix is None:
            return None
        arr = np.asarray(matrix)
        if nm_list:
            if arr.shape[1] == len(nm_list):
                return arr.T
            return arr
        return arr

    if A is not None:
        A_export = _orient_abs_matrix(A)
    if A_solver is not None:
        A_solver_export = _orient_abs_matrix(A_solver)
    if A_species is not None:
        A_species_export = _orient_abs_matrix(A_species)
    if A_species_solver is not None:
        A_species_solver_export = _orient_abs_matrix(A_species_solver)

    export_data = {
        "modelo": modelo.tolist() if modelo is not None else [],
        "C": np.asarray(C).tolist() if C is not None else [],
        "Co": np.asarray(Co).tolist() if Co is not None else [],
        "C_T": np.asarray(C_T).tolist() if C_T is not None else [],
        "A": A_export.tolist() if A_export is not None else [],
        "A_solver": A_solver_export.tolist() if A_solver_export is not None else [],
        "A_species": A_species_export.tolist() if A_species_export is not None else [],
        "A_species_solver": (
            A_species_solver_export.tolist() if A_species_solver_export is not None else []
        ),
        "A_index": nm_list,
        "k": np.asarray(k).tolist(),
        "k_solver": np.asarray(k_solver).tolist(),
        "k_ini": np.asarray(initial_k).tolist() if initial_k is not None else [],
        "k_ini_solver": (
            expand_solver_params(initial_k if initial_k is not None else [], solver_param_transform_arr).tolist()
            if initial_k is not None
            else []
        ),
        "percK": np.asarray(percK).tolist(),
        "SE_log10K": np.asarray(SE_log10K).tolist(),
        "param_names": list(k_names),
        "solver_param_transform": (
            None
            if solver_param_transform_arr is None
            else onp.asarray(solver_param_transform_arr, dtype=float).tolist()
        ),
        "solver_model_settings": str(solver_model_settings),
        "nm": nm_list,
        "Y": np.asarray(Y).tolist() if Y is not None else [],
        "Y_raw": np.asarray(Y_raw).tolist() if Y_raw is not None else [],
        "Y_corr": np.asarray(Y).tolist() if Y is not None else [],
        "A_exp_raw": onp.asarray(A_exp_raw).tolist() if A_exp_raw is not None else [],
        "A_exp_used": onp.asarray(A_exp_used).tolist() if A_exp_used is not None else [],
        "yfit": np.asarray(yfit).tolist() if yfit is not None else [],
        "baseline_vals": onp.asarray(baseline_vals, dtype=float).tolist(),
        "baseline_meta": baseline_meta,
        "weights": onp.asarray(weights_per_lambda, dtype=float).tolist(),
        "weighting_meta": weighting_meta,
        "eps_solver_mode": eps_solver_mode,
        "mu": float(eps_mu),
        "delta_mode": delta_mode,
        "delta_rel": float(delta_rel),
        "alpha_smooth": float(alpha_smooth),
        "lower_bound": (
            onp.asarray(lower_bound_final, dtype=float).ravel().tolist()
            if lower_bound_final is not None
            else None
        ),
        "stats_table": [
            ["RMS", float(rms)],
            ["Error absoluto medio", float(MAE)],
            ["Diferencia en C total (%)", float(dif_en_ct)],
            ["covfit", float(covfit)],
            ["optimizer", optimizer],
            ["objective_evaluations", int(objective_eval_count)],
        ],
        "non_abs_species": non_abs_species,
        "render_graphs": bool(render_graphs),
        "render_quality": render_quality,
        "objective_evaluations": int(objective_eval_count),
        "species_names": species_all,
        "abs_species_labels": abs_species_labels,
        "abs_group_map": onp.asarray(abs_group_map, dtype=float).tolist(),
        "abs_group_ids": list(abs_grouping["group_ids"]),
        "abs_group_labels": list(group_labels),
        "abs_group_members": [list(members) for members in group_members],
        "abs_group_labels_with_members": list(abs_grouping["group_labels_with_members"]),
        "abs_group_count": int(len(group_labels)),
        "grouped_absorptivities_active": grouped_abs_active,
    }
    if derived_noncoop is not None:
        export_data["derived_noncoop"] = derived_noncoop

    plot_data = None
    try:
        plot_data = build_spectroscopy_plot_data(
            nm=nm,
            Ct=C_T,
            ct_columns=column_names,
            Y_exp=Y_exp,
            Y_fit=Y_fit_arr,
            residuals=residuals_matrix,
            C=C,
            species_labels=species_labels,
            eigenvalues=eigenvalues,
            efa_forward=efa_forward,
            efa_backward=efa_backward,
        )
    except Exception as exc:
        # No interrumpir la respuesta por errores de plot_data
        logger.exception("Error construyendo plot_data: %s", exc)
        plot_data = None
    legacy_graphs = graphs

    # Build availablePlots list (ordered pages for carousel navigation)
    availablePlots = []
    
    # Always add these core plots if data is available
    if graphs.get('fit'):
        availablePlots.append({
            "id": "spec_fit_overlay",
            "title": "Experimental vs fitted spectra",
            "kind": "image"
        })
    
    # Species distribution - interactive (kind: plotly)
    if C is not None and len(C) > 0:
        availablePlots.append({
            "id": "spec_species_distribution",
            "title": "Species distribution",
            "kind": "plotly"
        })
    
    if graphs.get('absorptivities'):
        availablePlots.append({
            "id": "spec_molar_absorptivities",
            "title": "Molar absorptivities",
            "kind": "image"
        })
    
    # EFA plots only when EFA is enabled
    if efa_enabled:
        if graphs.get('eigenvalues'):
            availablePlots.append({
                "id": "spec_efa_eigenvalues",
                "title": "EFA eigenvalues",
                "kind": "image"
            })
        if graphs.get('efa'):
            availablePlots.append({
                "id": "spec_efa_components",
                "title": "EFA forward/backward",
                "kind": "image"
            })
    
    # Build axis options for species distribution
    axis_options = [{"id": "titrant_total", "label": f"[{guest_label_display}] total"}]
    axis_vectors = {"titrant_total": (G if G is not None else H).tolist() if (G is not None or H is not None) else []}
    
    # Build species options with id/label and C_by_species for direct lookup
    species_options = [{"id": sp, "label": sp} for sp in species_labels]
    C_by_species = {}
    for i, sp_label in enumerate(species_labels):
        C_by_species[sp_label] = np.asarray(C[:, i]).tolist() if C is not None else []
        # Also add species as potential X axis
        axis_options.append({"id": f"species:{sp_label}", "label": f"[{sp_label}]"})
        axis_vectors[f"species:{sp_label}"] = C_by_species[sp_label]
    
    # Build plotData with PNG base64 keyed by plot ID (images) and arrays (plotly)
    spec_plot_data = {
        "spec_fit_overlay": {"png_base64": graphs.get('fit', '')},
        "spec_species_distribution": {
            "axisOptions": axis_options,
            "axisVectors": axis_vectors,
            "speciesOptions": species_options,
            "C_by_species": C_by_species,
            "x_default": axis_vectors.get("titrant_total", []),
        },
        "spec_molar_absorptivities": {
            "png_base64": graphs.get('absorptivities', ''),
            "speciesOptions": [
                {"id": lbl, "label": lbl}
                for lbl in abs_grouping["group_labels_with_members"]
            ],
        },
        "spec_efa_eigenvalues": {"png_base64": graphs.get('eigenvalues', '')},
        "spec_efa_components": {"png_base64": graphs.get('efa', '')},
    }
    post_elapsed = time.perf_counter() - t_post_start
    stage_times["postprocess_s"] = max(0.0, post_elapsed - stage_times["graphs_s"])
    stage_times["total_s"] = time.perf_counter() - t_total_start
    export_data["timings"] = dict(stage_times)

    # Format results
    results = {
        "success": True,
        "constants": [
            {
                "name": str(k_names[i]),
                "log10K": float(k[i]),
                "SE_log10K": float(SE_log10K[i]),
                "K": float(10**k[i]),
                "SE_K": float(SE_K[i]),
                "percent_error": float(percK[i]),
            }
            for i in range(len(k))
        ],
        "derived_noncoop": export_data.get("derived_noncoop"),
        "statistics": {
            "RMS": float(rms),
            "lof": float(lof),
            "MAE": float(MAE),
            "dif_en_ct": float(dif_en_ct),
            "eigenvalues": int(EV),
            "covfit": float(covfit),
            "optimizer": optimizer,
        },
        "graphs": legacy_graphs,
        "legacy_graphs": legacy_graphs,
        "plot_data": plot_data,
        "availablePlots": availablePlots,
        "plotData": {"spec": spec_plot_data},
        "results_text": results_text,
        "export_data": export_data,
        "optimizer_result": {
            "success": bool(optimizer_summary.get("success", onp.isfinite(best_result["rms"]))),
            "message": str(optimizer_summary.get("message", "")),
            "nfev": int(optimizer_summary.get("nfev", 0) or 0),
            "objective_evaluations": int(objective_eval_count),
        },
        "timings": dict(stage_times),
        "render_graphs": bool(render_graphs),
        "render_quality": render_quality,
        "channels_total": int(channels_total),
        "channels_used": int(channels_used),
        "channels_mode": "custom"
        if (isinstance(resolved, (list, tuple)) and len(resolved) > 0)
        else "all",
        "plot_mode": plot_mode,
        "warnings": warnings_list,
        "stability_status": stability_diag.get("status") if stability_diag else None,
        "condition_number": stability_diag.get("cond_jjt") if stability_diag else None,
        "stability_indicator": stability_diag.get("stability_indicator") if stability_diag else None,
        "diagnostics_summary": stability_diag.get("diag_summary") if stability_diag else None,
        "diagnostics_full": stability_diag.get("diag_full") if stability_diag else None,
    }
    
    log_progress(
        "Timing (s): parse={parse_s:.3f} solver={solver_s:.3f} opt={optimization_s:.3f} "
        "post={postprocess_s:.3f} graphs={graphs_s:.3f} total={total_s:.3f}".format(**stage_times)
    )
    log_progress(
        "Optimizer summary: nfev=%d objective_evals=%d best_rms=%.6e"
        % (
            int(optimizer_summary.get("nfev", 0) or 0),
            int(objective_eval_count),
            float(best_result["rms"]),
        )
    )
    log_progress("Procesamiento completado exitosamente")
    
    try:
        log(results_text)
    except Exception:
        pass
    results["log_output"] = "\n".join(log_lines)
    return _sanitize_for_json(results)
