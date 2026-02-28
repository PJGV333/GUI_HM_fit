# SPDX-License-Identifier: GPL-3.0-or-later
"""
NMR Processor Module
--------------------
Handles the loading, processing, and fitting of NMR titration data.
This version includes a multi-start optimization strategy to improve
robustness when fitting NMR binding constants. The core algorithm
remains identical to the original HypNMR-inspired implementation
but wraps the optimizer in a loop over randomly perturbed initial
guesses. When multiple runs are specified via ``multi_start_runs``,
the optimizer is executed that many times and the best solution is
retained.

This file is auto-generated for testing purposes within the local
environment. It replicates the functionality of the original
``hmfit_core/processors/nmr_processor.py`` from the GUI_HM_fit
project, with minor additions to expose the multi-start feature.
"""
from __future__ import annotations

import io
import base64
import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from scipy import optimize
from scipy.optimize import differential_evolution, dual_annealing, basinhopping
import logging

logger = logging.getLogger(__name__)

from ..solvers import NewtonRaphson, LevenbergMarquardt  # type: ignore
from ..utils.errors import compute_errors_nmr_varpro  # type: ignore

from .spectroscopy_processor import (
    format_results_table,
    _build_bounds_list,
    _aliases_from_names,
    _alias_re,
    generate_figure_base64,
    generate_figure2_base64,
)
from ..utils.noncoop_utils import noncoop_derived_from_logK1  # type: ignore

# --- Progress callback (optional) ---
_progress_callback = None
_loop = None


def set_progress_callback(callback, loop=None):
    """Register callback to emit progress messages."""
    global _progress_callback, _loop
    _progress_callback = callback
    _loop = loop


def log_progress(message: str):
    """Send progress message if a callback is registered."""
    if _progress_callback:
        if _loop:
            _loop.call_soon_threadsafe(_progress_callback, message)
        else:
            _progress_callback(message)


def _is_fit_cancelled(exc: BaseException) -> bool:
    return exc.__class__.__name__ == "FitCancelled"


def pinv_cs(A, rcond=1e-12):
    """Compute the Moore-Penrose pseudo-inverse of a matrix, handling complex numbers."""
    A = np.asarray(A)
    if not np.iscomplexobj(A):
        return np.linalg.pinv(A, rcond=rcond)
    m, n = A.shape
    Ar = np.block([[A.real, -A.imag], [A.imag, A.real]])
    Pr = np.linalg.pinv(Ar, rcond=rcond)
    X = Pr[:n, :m]
    Y = Pr[n: 2 * n, :m]
    return X + 1j * Y


def build_D_cols(CT, conc_colnames, signal_colnames, default_idx=0):
    """
    Builds the D_cols matrix which maps signals to their parent species concentration column.
    """
    # Normalize CT to ndarray and align names
    if isinstance(CT, pd.DataFrame):
        CT_arr = CT.to_numpy(dtype=float)
        if not conc_colnames or len(conc_colnames) != CT.shape[1]:
            conc_colnames = list(CT.columns)
    else:
        CT_arr = np.asarray(CT, dtype=float)

    alias2idx = _aliases_from_names(conc_colnames)
    m_sig = len(signal_colnames)
    n_i = CT_arr.shape[0]

    D_cols = np.empty((n_i, m_sig), float)
    parent_idx = []

    for j, sname in enumerate(signal_colnames):
        s = str(sname).lower()
        found = None
        for a in _alias_re.findall(s):
            k = a.strip().lower()
            if k in alias2idx:
                found = alias2idx[k]
                break
        if found is None:
            head = s.split("(")[0].strip().split()
            if head:
                found = alias2idx.get(head[-1])
        if (found is None) or (found < 0) or (found >= CT_arr.shape[1]):
            found = default_idx
        D_cols[:, j] = CT_arr[:, found]
        parent_idx.append(found)
    return D_cols, parent_idx


def project_coeffs_block_onp_frac(dq_block, C_block, D_cols, mask_block, stoichiometry=None, rcond=1e-10, ridge=0.0):
    """
    Project chemical shift differences using mole fractions.
    This supports explicit stoichiometry for equivalent nuclei.
    """
    m, nP = dq_block.shape
    dq_calc = np.full_like(dq_block, np.nan, dtype=float)
    S = None
    if stoichiometry is not None:
        try:
            S = np.asarray(stoichiometry, dtype=float)
            if S.shape[0] != C_block.shape[1] or S.shape[1] != nP:
                S = None
        except Exception:
            S = None
    for j in range(nP):
        if S is not None:
            S_j = S[:, j]
            D = C_block @ S_j
            valid_D = (np.abs(D) > 1e-12)
        else:
            D = D_cols[:, j]
            valid_D = np.isfinite(D) & (np.abs(D) > 1e-12)
            S_j = None
        valid_C = np.isfinite(C_block).all(axis=1)
        mj = mask_block[:, j] & valid_C & valid_D & np.isfinite(D)
        nj = mj.sum()
        if nj < 2:
            continue
        if S is not None:
            C_sub = C_block[mj, :]
            D_sub = D[mj][:, None]
            F_j = (C_sub * S_j[None, :]) / D_sub
        else:
            F_j = C_block[mj, :] / D[mj][:, None]
        y = dq_block[mj, j]
        if not np.isfinite(F_j).all():
            continue
        try:
            delta_species, _, _, _ = np.linalg.lstsq(F_j, y, rcond=rcond)
        except np.linalg.LinAlgError:
            if ridge > 0.0:
                XtX = F_j.T @ F_j
                delta_species = np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), F_j.T @ y)
            else:
                delta_species = np.linalg.pinv(F_j, rcond=rcond) @ y
        pred_mask = valid_C & valid_D & np.isfinite(D)
        if pred_mask.any():
            if S is not None:
                F_pred = (C_block[pred_mask, :] * S_j[None, :]) / D[pred_mask][:, None]
            else:
                F_pred = C_block[pred_mask, :] / D[pred_mask][:, None]
            dq_calc[pred_mask, j] = F_pred @ delta_species
    return dq_calc


def sanitize_for_json(obj):
    """Recursively sanitize data structure to replace NaN and Inf with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.ndarray,)):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    else:
        return obj


def process_nmr_data(
    file_path: str,
    spectra_sheet: str,
    conc_sheet: str,
    column_names: List[str],
    signal_names: List[str],
    receptor_label: str,
    guest_label: str,
    model_matrix: List[List[float]],
    k_initial: List[float],
    k_bounds: List[Dict[str, float]],
    algorithm: str,
    optimizer: str,
    model_settings: str,
    non_absorbent_species: List[int],
    k_fixed: Optional[List[bool]] = None,
    stoichiometry_map: Optional[List[List[float]]] = None,
    show_stability_diagnostics: bool = False,
    multi_start_runs: int = 1,
    multi_start_seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Main entry point for processing NMR titration data and fitting binding constants.

    Parameters
    ----------
    file_path : str
        Path to the Excel workbook containing the NMR data.
    spectra_sheet : str
        Name of the sheet with chemical shift data.
    conc_sheet : str
        Name of the sheet with total concentrations.
    column_names : List[str]
        Columns in ``conc_sheet`` to use for concentration data.
    signal_names : List[str]
        Columns in ``spectra_sheet`` to use for chemical shifts.
    receptor_label : str
        Name of the receptor column in ``conc_sheet``.
    guest_label : str
        Name of the guest column in ``conc_sheet``.
    model_matrix : List[List[float]]
        Stoichiometric coefficients for each species.
    k_initial : List[float]
        Initial guesses for the binding constants (log10 values).
    k_bounds : List[Dict[str, float]]
        Bounds for each binding constant.
    algorithm : str
        Which algorithm to use for species distribution (Newton-Raphson or Levenberg-Marquardt).
    optimizer : str
        Optimizer to use (e.g. ``'powell'`` or ``'differential_evolution'``).
    model_settings : str
        Additional model settings (e.g. ``'Non-cooperative'`` or ``'Statistical'``).
    non_absorbent_species : List[int]
        Indices of species that do not absorb (excluded from fitting).
    k_fixed : Optional[List[bool]]
        Mask for binding constants that should be fixed during optimization.
    stoichiometry_map : Optional[List[List[float]]]
        Custom stoichiometry matrix per signal (overrides default mapping).
    show_stability_diagnostics : bool
        If True, include full stability diagnostics in the output.
    multi_start_runs : int
        Number of multi-start optimization runs. Set to >1 to enable random restarts.
    multi_start_seeds : Optional[List[int]]
        Optional RNG seeds for each restart (must match the number of runs).

    Returns
    -------
    Dict[str, Any]
        A JSON-serializable dictionary containing fit results, figures, and statistics.
    """
    # 1. Load Data
    try:
        chemshift_data = pd.read_excel(file_path, spectra_sheet, header=0)
        conc_data = pd.read_excel(file_path, conc_sheet, header=0)
    except Exception as e:
        return {"error": f"Error loading Excel file: {str(e)}"}

    # 2. Extract and prepare matrices
    try:
        Chem_Shift_T = chemshift_data[signal_names]
        Chem_Shift_T = Chem_Shift_T.apply(pd.to_numeric, errors='coerce')
        dq = np.asarray(Chem_Shift_T.to_numpy(dtype=float))
        mask = np.isfinite(dq)
        C_T_df = conc_data[column_names]
        C_T = C_T_df.to_numpy(dtype=float)
        H = None
        G = None
        try:
            h_idx = column_names.index(receptor_label)
            H = C_T[:, h_idx]
        except ValueError:
            pass
        try:
            g_idx = column_names.index(guest_label)
            G = C_T[:, g_idx]
        except ValueError:
            pass
        model_raw = np.asarray(model_matrix, dtype=float) if model_matrix is not None else np.zeros((0, 0))
        model_cols = model_raw.shape[1] if model_raw.ndim == 2 else 0
        model_rows = model_raw.shape[0] if model_raw.ndim == 2 else 0
        guest_missing = (guest_label is None) or (str(guest_label).strip() == "") or (G is None)
        expand_dummy_guest = guest_missing and model_cols == 1 and model_rows > 0
        if expand_dummy_guest:
            dummy_name = "__DUMMY_GUEST__"
            if dummy_name not in C_T_df.columns:
                C_T_df[dummy_name] = 0.0
                column_names = list(column_names) + [dummy_name]
                C_T = C_T_df.to_numpy(dtype=float)
            G = None
            if model_rows > 0:
                dummy_col = np.zeros((model_rows, 1), dtype=float)
                model_raw = np.concatenate([model_raw, dummy_col], axis=1)
                dummy_row = np.zeros((1, model_cols + 1), dtype=float)
                dummy_row[0, model_cols] = 1.0
                model_raw = np.concatenate(
                    [model_raw[:model_cols, :], dummy_row, model_raw[model_cols:, :]], axis=0
                )
                model_matrix = model_raw.tolist()
            if non_absorbent_species:
                non_absorbent_species = [
                    (int(idx) + 1) if int(idx) >= model_cols else int(idx)
                    for idx in non_absorbent_species
                ]
            if stoichiometry_map is not None and model_rows > 0:
                try:
                    stoich_arr = np.asarray(stoichiometry_map, dtype=float)
                    if stoich_arr.ndim == 2 and stoich_arr.shape[0] == model_rows:
                        zero_row = np.zeros((1, stoich_arr.shape[1]), dtype=float)
                        stoich_arr = np.concatenate(
                            [stoich_arr[:model_cols, :], zero_row, stoich_arr[model_cols:, :]], axis=0
                        )
                        stoichiometry_map = stoich_arr.tolist()
                except Exception:
                    pass
        if H is None and G is None:
            return {"error": "Could not identify Receptor or Guest columns."}
        if H is None:
            H = C_T[:, 0]
        nc = len(C_T)
        n_comp = C_T.shape[1]
        D_cols, parent_idx = build_D_cols(C_T_df, column_names, signal_names, default_idx=0)
        mask = mask & np.isfinite(D_cols)
        modelo = np.array(model_matrix).T
        nas = non_absorbent_species
        k = np.array(k_initial, dtype=float).ravel()
        bnds = _build_bounds_list(k_bounds)
        fixed_mask = np.zeros(len(k), dtype=bool)
        if k_fixed is not None:
            for i, f in enumerate(k_fixed[: len(k)]):
                fixed_mask[i] = bool(f)
        for i, (lb, ub) in enumerate(bnds):
            if np.isfinite(lb) and np.isfinite(ub) and lb == ub:
                fixed_mask[i] = True
        free_idx = np.where(~fixed_mask)[0]
        p0_full = k.copy()
        def pack(theta_free: np.ndarray) -> np.ndarray:
            k_full = p0_full.copy()
            if free_idx.size:
                k_full[free_idx] = np.asarray(theta_free, dtype=float).ravel()
            return k_full
        stoich_mat = None
        if stoichiometry_map is not None:
            try:
                stoich_mat = np.asarray(stoichiometry_map, dtype=float)
                n_signals = dq.shape[1]
                n_species_full = modelo.shape[1]
                nas_idx = sorted({int(x) for x in (nas or []) if isinstance(x, (int, np.integer)) or str(x).lstrip("-").isdigit()})
                nas_idx = [i for i in nas_idx if 0 <= i < n_species_full]
                n_species_eff = n_species_full - len(nas_idx)
                if stoich_mat.shape == (n_species_full, n_signals):
                    pass
                elif stoich_mat.shape == (n_signals, n_species_full):
                    stoich_mat = stoich_mat.T
                elif stoich_mat.shape == (n_species_eff, n_signals):
                    pass
                elif stoich_mat.shape == (n_signals, n_species_eff):
                    stoich_mat = stoich_mat.T
                else:
                    log_progress(
                        f"Warning: Stoichiometry shape {stoich_mat.shape} incompatible. Expected ({n_species_full}, {n_signals}). Ignoring."
                    )
                    stoich_mat = None
                if stoich_mat is not None and nas_idx and stoich_mat.shape[0] == n_species_full:
                    stoich_mat = np.delete(stoich_mat, nas_idx, axis=0)
                    if show_stability_diagnostics:
                        log_progress(
                            "[DEBUG] stoich_mat filtered for non_absorbent_species="
                            f"{nas_idx} -> shape={stoich_mat.shape}"
                        )
            except Exception as e:
                if _is_fit_cancelled(e):
                    raise
                log_progress(f"Warning: Error reading stoichiometry: {e}")
                stoich_mat = None
        if stoich_mat is None:
            log_progress("[DEBUG] stoich_mat=None -> using default mode (no per-signal stoichiometry).")
        else:
            log_progress(f"[DEBUG] stoich_mat OK shape={stoich_mat.shape}")
            sig = [tuple(np.unique(stoich_mat[:, i]).tolist()) for i in range(min(stoich_mat.shape[1], 8))]
            log_progress(f"[DEBUG] stoich_mat unique values per signal (first 8 cols): {sig}")
    except Exception as e:
        if _is_fit_cancelled(e):
            raise
        return {"error": f"Error preparing data matrices: {str(e)}"}
    log_progress("Iniciando procesamiento NMR…")
    log_progress(f"Optimizer: {optimizer} | Algorithm: {algorithm}")
    if stoich_mat is not None:
        log_progress("Using Stoichiometry-based signal calculation.")
    # 3. Initialize Algorithm
    try:
        if algorithm == "Newton-Raphson":
            res = NewtonRaphson(C_T_df, modelo, nas, model_settings)
        elif algorithm == "Levenberg-Marquardt":
            res = LevenbergMarquardt(C_T_df, modelo, nas, model_settings)
        else:
            return {"error": f"Unknown algorithm: {algorithm}"}
    except Exception as e:
        if _is_fit_cancelled(e):
            raise
        return {"error": f"Error initializing algorithm: {str(e)}"}
    # --- IRLS Loop Setup ---
    weights_per_signal = np.ones(dq.shape[1], dtype=float)
    # 4. Define Objective Function
    iter_state = {"cnt": 0, "best": np.inf}
    def f_m(theta_free):
        try:
            iter_state["cnt"] += 1
            k_curr_full = pack(theta_free)
            C = res.concentraciones(k_curr_full)[0]
            dq_cal = project_coeffs_block_onp_frac(
                dq, C, D_cols, mask, stoichiometry=stoich_mat, rcond=1e-10, ridge=1e-8
            )
            diff = dq - dq_cal
            valid_residuals = mask & np.isfinite(dq_cal)
            w_mat = weights_per_signal[None, :]
            residuals_flat = diff[valid_residuals]
            weights_flat = np.broadcast_to(w_mat, diff.shape)[valid_residuals]
            w_r2 = (residuals_flat**2) * weights_flat
            if (residuals_flat.size <= len(theta_free)) or (not np.isfinite(w_r2).all()):
                return 1e9
            weighted_rms = float(np.sqrt(np.mean(w_r2)))
            best = iter_state["best"]
            should_log = (
                iter_state["cnt"] == 1
                or (iter_state["cnt"] % 10 == 0)
                or (weighted_rms < best * 0.999)
            )
            if should_log:
                log_progress(
                    f"Iter {iter_state['cnt']}: f(x)={weighted_rms:.6e} | x={[float(xi) for xi in k_curr_full]}"
                )
                iter_state["best"] = min(best, weighted_rms)
            return weighted_rms
        except Exception as exc:
            if _is_fit_cancelled(exc):
                raise
            return 1e9
    # 5. Optimization (IRLS)
    MAX_IRLS_CYCLES = 5
    cycle_count = 0
    k_opt_full = p0_full.copy()
    old_best_rms = np.inf
    while cycle_count < MAX_IRLS_CYCLES:
        cycle_count += 1
        iter_state["best"] = np.inf  # Reset for current weights
        log_progress(f"--- IRLS Cycle {cycle_count}/{MAX_IRLS_CYCLES} ---")
        param_opt_failed = False
        try:
            if free_idx.size == 0:
                k_opt_full = p0_full.copy()
            else:
                k_free0 = p0_full[free_idx]
                bounds_free = [bnds[i] for i in free_idx]
                def _bounds_finite(bounds_list: list[tuple[float | None, float | None]]) -> bool:
                    for lb, ub in bounds_list:
                        if lb is None or ub is None:
                            return False
                        if not np.isfinite(lb) or not np.isfinite(ub):
                            return False
                    return True

                def _run_optimizer(start_free, seed):
                    if optimizer == "differential_evolution" and cycle_count == 1:
                        return differential_evolution(
                            f_m,
                            bounds_free,
                            x0=start_free,
                            strategy='best1bin',
                            maxiter=1000,
                            popsize=15,
                            tol=0.01,
                            seed=seed,
                        )
                    if optimizer == "dual_annealing":
                        if not _bounds_finite(bounds_free):
                            msg = "Dual annealing requires finite bounds. Set Min/Max for each K."
                            log_progress(msg)
                            raise ValueError(msg)
                        return dual_annealing(f_m, bounds_free, seed=seed)
                    if optimizer == "basinhopping":
                        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds_free}
                        return basinhopping(
                            f_m, start_free, niter=100, minimizer_kwargs=minimizer_kwargs, seed=seed
                        )
                    if optimizer == "global_local":
                        if not _bounds_finite(bounds_free):
                            msg = "Global-local optimization requires finite bounds. Set Min/Max for each K."
                            log_progress(msg)
                            raise ValueError(msg)
                        global_res = differential_evolution(
                            f_m, bounds_free, x0=start_free, maxiter=200, popsize=10, tol=0.05, seed=seed
                        )
                        return optimize.minimize(
                            f_m, global_res.x, method="L-BFGS-B", bounds=bounds_free
                        )
                    method = optimizer if optimizer != "differential_evolution" else "powell"
                    return optimize.minimize(f_m, start_free, method=method, bounds=bounds_free)
                # Multi-start optimization: run the optimizer multiple times with different initial guesses
                best_rms_val = np.inf
                best_full_params = None
                ms_runs = max(int(multi_start_runs), 1)
                seed_list = None
                if multi_start_seeds is not None:
                    try:
                        seeds = [int(s) for s in multi_start_seeds]
                    except Exception:
                        seeds = []
                    if seeds and len(seeds) == ms_runs:
                        seed_list = seeds
                for ms_iter in range(ms_runs):
                    # Reset iteration state for each run
                    iter_state["best"] = np.inf
                    iter_state["cnt"] = 0
                    seed = seed_list[ms_iter] if seed_list else None
                    if ms_iter == 0:
                        start_free = k_free0
                    else:
                        rng = np.random.default_rng(seed)
                        start_free = np.empty_like(k_free0, dtype=float)
                        for ii, (lb, ub) in enumerate(bounds_free):
                            lb_val = lb if lb is not None and np.isfinite(lb) else None
                            ub_val = ub if ub is not None and np.isfinite(ub) else None
                            if lb_val is not None and ub_val is not None:
                                start_free[ii] = rng.uniform(lb_val, ub_val)
                            else:
                                perturb = rng.uniform(-1.0, 1.0)
                                start_free[ii] = k_free0[ii] + perturb
                    run_failed = False
                    try:
                        opt_res = _run_optimizer(start_free, seed)
                        try:
                            fval = float(opt_res.fun)
                            current_full = pack(opt_res.x)
                        except Exception:
                            fval = np.inf
                            current_full = None
                        if np.isfinite(fval) and fval < best_rms_val:
                            best_rms_val = fval
                            best_full_params = current_full
                    except Exception as run_exc:
                        if _is_fit_cancelled(run_exc):
                            raise
                        run_failed = True
                        log_progress(f"[DEBUG] optimization run {ms_iter+1}/{ms_runs} failed: {run_exc}")
                    if ms_runs > 1:
                        best_msg = f"{best_rms_val:.6e}" if np.isfinite(best_rms_val) else "inf"
                        log_progress(f"[MS] run {ms_iter+1}/{ms_runs} best={best_msg}")
                    if run_failed:
                        continue
                if best_full_params is not None:
                    k_opt_full = best_full_params
                else:
                    opt_res = _run_optimizer(k_free0, seed_list[0] if seed_list else None)
                    k_opt_full = pack(opt_res.x)
                p0_full = k_opt_full
        except Exception as e:
            if _is_fit_cancelled(e):
                raise
            log_progress(f"Optimization cycle failed: {e}")
            param_opt_failed = True
            break
        # Update weights
        C_opt = res.concentraciones(k_opt_full)[0]
        dq_fit = project_coeffs_block_onp_frac(
            dq, C_opt, D_cols, mask, stoichiometry=stoich_mat
        )
        residuals_mat = dq - dq_fit
        new_weights = []
        for j in range(dq.shape[1]):
            mj = mask[:, j] & np.isfinite(dq_fit[:, j])
            rj = residuals_mat[mj, j]
            if rj.size > 2:
                sigma_j = np.sqrt(np.mean(rj**2))
                sigma_j = max(sigma_j, 1e-4)
                w_j = 1.0 / (sigma_j**2)
            else:
                w_j = weights_per_signal[j]
            new_weights.append(w_j)
        new_weights = np.array(new_weights)
        diff_w = np.max(np.abs(weights_per_signal - new_weights) / (weights_per_signal + 1e-9))
        weights_per_signal = new_weights
        log_progress(f"Cycle {cycle_count} done. Max weight change: {diff_w:.2%}")
        current_best_rms = iter_state["best"]
        if cycle_count > 1 and np.abs(current_best_rms - old_best_rms) < 1e-6 * old_best_rms:
            log_progress("IRLS Converged (RMS stability).")
            break
        if diff_w < 0.05 and cycle_count > 1:
            log_progress("IRLS Converged (Weights stable).")
            break
        old_best_rms = current_best_rms
    # 6. Calculate Results & Statistics
    try:
        C_opt, Co_opt = res.concentraciones(k_opt_full)
        dq_fit = project_coeffs_block_onp_frac(
            dq, C_opt, D_cols, mask, stoichiometry=stoich_mat, rcond=1e-10, ridge=1e-8
        )
        diff_final = dq - dq_fit
        valid_residuals_final = mask & np.isfinite(dq_fit)
        residuals_vec = diff_final[valid_residuals_final].ravel()
        SE_log10K_full = np.zeros_like(k_opt_full, dtype=float)
        percK_full = np.zeros_like(k_opt_full, dtype=float)
        covfit_val = np.nan
        try:
            k_names = [f"K{i+1}" for i in range(len(k_opt_full))]
            err_res = compute_errors_nmr_varpro(
                k_opt_full,
                res,
                dq,
                D_cols,
                modelo,
                nas,
                stoichiometry=stoich_mat,
                mask=mask,
                fixed_mask=fixed_mask,
                weights=weights_per_signal,
                rcond=1e-10,
                rcond_cov=1e-10,
                ridge=1e-8,
                ridge_cov=0.0,
                use_projector=True,
                debug=False,
                param_names=k_names,
            )
            stability_diag = err_res.get("stability_diag", {})
            SE_log10K_full = err_res["SE_log10K"]
            percK_full = err_res["percK"]
            covfit_val = err_res["covfit"]
            rms = err_res["rms"]
            residuals_vec = err_res.get("residuals_vec", residuals_vec)
        except Exception as e:
            if _is_fit_cancelled(e):
                raise
            log_progress(f"Error calculando errores: {e}")
            import traceback
            logger.error(traceback.format_exc())
            covfit_val = np.nan
        SE_log10K_full[fixed_mask] = 0.0
        percK_full[fixed_mask] = 0.0
        rms = np.sqrt(np.mean(residuals_vec**2)) if residuals_vec.size > 0 else 0.0
        dq_vec = dq.ravel()
        dq_fit_vec = dq_fit.ravel()
        residuals_masked = residuals_vec
        SS_res = float(np.sum(residuals_masked**2))
        dq_obs = dq[valid_residuals_final]
        SS_tot = float(np.sum((dq_obs - np.mean(dq_obs))**2))
        lof = 0.0 if SS_tot <= 1e-30 else 100.0 * SS_res / SS_tot
        MAE = float(np.mean(np.abs(residuals_masked)))
        if H is not None:
            dif_en_ct = round(max(100 - (np.sum(C_opt, 1) * 100 / max(H))), 2)
        else:
            dif_en_ct = 0.0
        results_text = format_results_table(
            k_opt_full,
            SE_log10K_full,
            percK_full,
            rms,
            covfit_val,
            lof=lof,
            fixed_mask=fixed_mask,
        )
        if 'stability_diag' in locals() and stability_diag:
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
        if str(model_settings) in ("Non-cooperative", "Statistical") and np.asarray(k_opt_full).size == 1:
            try:
                derived = noncoop_derived_from_logK1(
                    np.asarray(modelo, dtype=float), float(np.asarray(k_opt_full).ravel()[0])
                )
                derived_noncoop = {
                    "N": int(derived["N"]),
                    "logK_by_j": np.asarray(derived["logK_by_j"], dtype=float).tolist(),
                }
                lines = ["", "Derived (Non-cooperative):", "j | log10(Kj)"]
                for j in range(1, int(derived["N"]) + 1):
                    lines.append(f"{j} | {derived['logK_by_j'][j-1]:.6g}")
                results_text += "\n" + "\n".join(lines)
            except Exception:
                derived_noncoop = None
        extra_stats = [
            f"LOF: {lof:.2e} %",
            f"MAE: {MAE:.2e}",
            f"Diferencia en C total (%): {dif_en_ct:.2f}",
            f"Optimizer: {optimizer}",
            f"Algorithm: {algorithm}",
            f"Model settings: {model_settings}",
        ]
        results_text += "\n\nEstadísticas:\n" + "\n".join(extra_stats)
        graphs = {}
        x_axis_conc = G if G is not None else H
        x_label_conc = f"[{guest_label}] Total (M)" if G is not None else f"[{receptor_label}] Total (M)"
        graphs['concentrations'] = generate_figure_base64(
            x_axis_conc, C_opt, ":o", "[Species], M", x_label_conc, "Concentration Profile"
        )
        graphs['fit'] = generate_figure2_base64(
            x_axis_conc, dq, dq_fit, "o", ":", "δ (ppm)", x_label_conc, 0.5, "Chemical Shifts Fit"
        )
        residuals = dq - dq_fit
        graphs['residuals'] = generate_figure_base64(
            x_axis_conc, residuals, "o", "Residuals (ppm)", x_label_conc, "Residuals"
        )
        export_data = {
            "modelo": modelo.T.tolist() if modelo is not None else [],
            "C": C_opt.tolist() if C_opt is not None else [],
            "Co": Co_opt.tolist() if Co_opt is not None else [],
            "C_T": C_T.tolist() if C_T is not None else [],
            "Chemical_Shifts": dq.tolist() if dq is not None else [],
            "Calculated_Chemical_Shifts": dq_fit.tolist() if dq_fit is not None else [],
            "k": k_opt_full.tolist() if k_opt_full is not None else [],
            "k_ini": k_initial if k_initial is not None else [],
            "percK": percK_full.tolist() if percK_full is not None else [],
            "SE_log10K": SE_log10K_full.tolist() if SE_log10K_full is not None else [],
            "fixed_mask": fixed_mask.tolist(),
            "derived_noncoop": derived_noncoop,
            "signal_names": signal_names,
            "column_names": column_names,
            "stoichiometry_map": stoich_mat.tolist() if stoich_mat is not None else [],
            "stats_table": [
                ["RMS", float(rms)],
                ["Error absoluto medio", float(MAE)],
                ["Diferencia en C total (%)", float(dif_en_ct)],
                ["covfit", float(covfit_val)],
                ["LOF", float(lof)],
                ["optimizer", optimizer],
                ["algorithm", algorithm],
            ],
            "non_absorbent_species": non_absorbent_species,
        }
        log_progress("Procesamiento NMR completado.")
        availablePlots = []
        if dq is not None:
            availablePlots.append({"id": "nmr_shifts_fit", "title": "Chemical shifts fit", "kind": "plotly"})
        if C_opt is not None and len(C_opt) > 0:
            availablePlots.append({"id": "nmr_species_distribution", "title": "Species distribution", "kind": "plotly"})
        if dq is not None:
            availablePlots.append({"id": "nmr_residuals", "title": "Residuals", "kind": "plotly"})
        x_axis_values = (G if G is not None else H).tolist() if (G is not None or H is not None) else []
        x_label = f"[{guest_label}] Total (M)" if G is not None else f"[{receptor_label}] Total (M)"
        def make_signal_id(name):
            return name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        residuals_arr = (dq - dq_fit) if dq is not None and dq_fit is not None else np.zeros_like(dq)
        signal_options = []
        signals_data = {}
        for i, sig_name in enumerate(signal_names):
            sig_id = make_signal_id(sig_name)
            signal_options.append({"id": sig_id, "label": sig_name})
            signals_data[sig_id] = {
                "obs": dq[:, i].tolist() if dq is not None else [],
                "fit": dq_fit[:, i].tolist() if dq_fit is not None else [],
                "resid": residuals_arr[:, i].tolist() if residuals_arr is not None else [],
            }
        species_labels = [f"sp{j+1}" for j in range(C_opt.shape[1])] if C_opt is not None else []
        species_options = [{"id": sp, "label": sp} for sp in species_labels]
        C_by_species = {}
        for j, sp_label in enumerate(species_labels):
            C_by_species[sp_label] = C_opt[:, j].tolist() if C_opt is not None else []
        axis_options = [{"id": "titrant_total", "label": x_label}]
        axis_vectors = {"titrant_total": x_axis_values}
        for j, sp_label in enumerate(species_labels):
            axis_options.append({"id": f"species:{sp_label}", "label": f"[{sp_label}]"})
            axis_vectors[f"species:{sp_label}"] = C_opt[:, j].tolist() if C_opt is not None else []
        nmr_plot_data = {
            "nmr_shifts_fit": {
                "x": x_axis_values,
                "xLabel": x_label,
                "signalOptions": signal_options,
                "signals": signals_data,
            },
            "nmr_species_distribution": {
                "axisOptions": axis_options,
                "axisVectors": axis_vectors,
                "speciesOptions": species_options,
                "C_by_species": C_by_species,
                "x_default": x_axis_values,
            },
            "nmr_residuals": {
                "x": x_axis_values,
                "xLabel": x_label,
                "signalOptions": signal_options,
                "signals": signals_data,
            },
        }
        constants = []
        for i in range(len(k_opt_full)):
            is_fixed = bool(fixed_mask[i]) if fixed_mask is not None else False
            se_val = 0.0 if is_fixed else float(SE_log10K_full[i]) if np.isfinite(SE_log10K_full[i]) else None
            perc_val = 0.0 if is_fixed else float(percK_full[i]) if np.isfinite(percK_full[i]) else None
            constants.append({
                "name": f"K{i+1}",
                "log10K": float(k_opt_full[i]),
                "SE_log10K": se_val,
                "percent_error": perc_val,
                "fixed": is_fixed,
            })
        return sanitize_for_json({
            "success": True,
            "results_text": results_text,
            "graphs": graphs,
            "availablePlots": availablePlots,
            "plotData": {"nmr": nmr_plot_data},
            "export_data": export_data,
            "constants": constants,
            "derived_noncoop": export_data.get("derived_noncoop"),
            "stability_status": stability_diag.get("status") if 'stability_diag' in locals() else None,
            "condition_number": stability_diag.get("cond_jjt") if 'stability_diag' in locals() else None,
            "stability_indicator": stability_diag.get("stability_indicator") if 'stability_diag' in locals() else None,
            "diagnostics_summary": stability_diag.get("diag_summary") if 'stability_diag' in locals() else None,
            "diagnostics_full": stability_diag.get("diag_full") if 'stability_diag' in locals() else None,
        })
    except Exception as e:
        if _is_fit_cancelled(e):
            raise
        logger.exception("Error calculating NMR results")
        return {"error": f"Error calculating results: {str(e)}"}

def list_sheets_from_bytes(file_bytes: bytes) -> List[str]:
    """Return available sheet names from an Excel workbook."""
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    return xl.sheet_names

def list_columns_from_bytes(file_bytes: bytes, sheet_name: str) -> List[str]:
    """Return column headers for a given sheet."""
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, nrows=0)
    return list(df.columns)
