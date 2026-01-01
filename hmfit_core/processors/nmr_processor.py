"""
NMR Processor Module
--------------------
Handles the loading, processing, and fitting of NMR titration data.
Handles the loading, processing, and fitting of NMR titration data.
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
from scipy.optimize import differential_evolution
import logging

logger = logging.getLogger(__name__)

# Import existing algorithms from the submodules
from ..solvers import NewtonRaphson, LevenbergMarquardt
from ..utils.errors import compute_errors_nmr_varpro

# Import helper functions from spectroscopy_processor (relative import)
from .spectroscopy_processor import (
    format_results_table, 
    _build_bounds_list, 
    _aliases_from_names,
    _alias_re,
    generate_figure_base64,
    generate_figure2_base64
)
from ..utils.noncoop_utils import noncoop_derived_from_logK1

# --- Progreso vía callback opcional (Historical reference to FastAPI WebSocket) ---
_progress_callback = None
_loop = None


def set_progress_callback(callback, loop=None):
    """Registrar callback para emitir progreso (p.ej. al WebSocket)."""
    global _progress_callback, _loop
    _progress_callback = callback
    _loop = loop


def log_progress(message: str):
    """Enviar mensaje de progreso si hay callback."""
    if _progress_callback:
        if _loop:
            _loop.call_soon_threadsafe(_progress_callback, message)
        else:
            _progress_callback(message)

def pinv_cs(A, rcond=1e-12):
    """
    Compute the Moore-Penrose pseudo-inverse of a matrix, handling complex numbers.
    """
    A = np.asarray(A)
    if not np.iscomplexobj(A):
        return np.linalg.pinv(A, rcond=rcond)
    m, n = A.shape
    Ar = np.block([[A.real, -A.imag],
                   [A.imag,  A.real]])      # (2m x 2n)
    Pr = np.linalg.pinv(Ar, rcond=rcond)    # (2n x 2m)
    X = Pr[:n,    :m]
    Y = Pr[n:2*n, :m]
    return X + 1j*Y

def build_D_cols(CT, conc_colnames, signal_colnames, default_idx=0):
    """
    Builds the D_cols matrix which maps signals to their parent species concentration column.
    """
    # --- normaliza CT a ndarray y alinea nombres ---
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
        # alias entre paréntesis
        for a in _alias_re.findall(s):
            k = a.strip().lower()
            if k in alias2idx:
                found = alias2idx[k]; break
        # fallback por token
        if found is None:
            head = s.split("(")[0].strip().split()
            if head:
                found = alias2idx.get(head[-1])

        if (found is None) or (found < 0) or (found >= CT_arr.shape[1]):
            found = default_idx  # último recurso

        D_cols[:, j] = CT_arr[:, found]
        parent_idx.append(found)

    return D_cols, parent_idx

def project_coeffs_block_onp_frac(dq_block, C_block, D_cols, mask_block, stoichiometry=None, rcond=1e-10, ridge=0.0):
    """
    Proyección por señal usando fracción molar.
    Soporta estequiometría explícita para núcleos equivalentes.
    
    Args:
        dq_block: (n_points, n_signals) Observed chemical shifts (absolute).
        C_block: (n_points, n_species) Concentrations of species.
        D_cols: (n_points, n_signals) Analytical total concentration (fallback denominator).
        mask_block: (n_points, n_signals) Boolean mask of observed data.
        stoichiometry: Optional (n_species, n_signals) array.
                       If provided, Denominator = C @ stoichiometry.
                       Numerator_k = C_k * stoichiometry[k, signal].
    """
    m, nP = dq_block.shape
    dq_calc = np.full_like(dq_block, np.nan, dtype=float)
    
    # Pre-process stoichiometry if provided
    S = None
    if stoichiometry is not None:
        try:
            S = np.asarray(stoichiometry, dtype=float)
            # Validation: (n_species, n_signals)
            # Assuming caller validated dimensions or we handle robustness
            if S.shape[0] != C_block.shape[1] or S.shape[1] != nP:
                 S = None
        except:
            S = None

    for j in range(nP):
        # Determine Denominator (D) and Validity
        if S is not None:
            # New HypNMR logic:
            S_j = S[:, j] # (n_species,)
            D = C_block @ S_j # (n_points,)
            valid_D = (np.abs(D) > 1e-12)
        else:
            # Old logic: Denominator is analytical parent concentration
            D = D_cols[:, j]
            valid_D = np.isfinite(D) & (np.abs(D) > 1e-12)
            S_j = None
            
        valid_C = np.isfinite(C_block).all(axis=1)
        mj = mask_block[:, j] & valid_C & valid_D & np.isfinite(D)
        
        nj = mj.sum()
        if nj < 2:
             continue

        # Construct Basis F_j (nj, n_species)
        if S is not None:
            # F_k = (C_k * S_kj) / D
            C_sub = C_block[mj, :]
            D_sub = D[mj][:, None]
            F_j = (C_sub * S_j[None, :]) / D_sub
        else:
            # F_j = C / D
            F_j = C_block[mj, :] / D[mj][:, None]
        
        y = dq_block[mj, j]
        
        if not np.isfinite(F_j).all():
            continue

        # Solve F_j * delta = y
        try:
            delta_species, _, _, _ = np.linalg.lstsq(F_j, y, rcond=rcond)
        except np.linalg.LinAlgError:
             if ridge > 0.0:
                 XtX = F_j.T @ F_j
                 delta_species = np.linalg.solve(XtX + ridge*np.eye(XtX.shape[0]), F_j.T @ y)
             else:
                 delta_species = np.linalg.pinv(F_j, rcond=rcond) @ y
        
        # Predict
        pred_mask = valid_C & valid_D & np.isfinite(D)
        
        if pred_mask.any():
            if S is not None:
                F_pred = (C_block[pred_mask, :] * S_j[None, :]) / D[pred_mask][:, None]
            else:
                F_pred = C_block[pred_mask, :] / D[pred_mask][:, None]
            
            dq_calc[pred_mask, j] = F_pred @ delta_species
            
    return dq_calc


def sanitize_for_json(obj):
    """
    Recursively sanitize data structure to replace NaN and Inf with None for JSON serialization.
    """
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
    spectra_sheet: str,  # In NMR context, this is the Chemical Shifts sheet
    conc_sheet: str,
    column_names: List[str],
    signal_names: List[str], # Columns selected from spectra_sheet
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
    stoichiometry_map: Optional[List[List[float]]] = None, # New Argument
    show_stability_diagnostics: bool = False,
) -> Dict[str, Any]:
    
    # 1. Load Data
    try:
        chemshift_data = pd.read_excel(file_path, spectra_sheet, header=0)
        conc_data = pd.read_excel(file_path, conc_sheet, header=0)
    except Exception as e:
        return {"error": f"Error loading Excel file: {str(e)}"}

    # 2. Extract and Prepare Matrices
    try:
        # Chemical Shifts (dq)
        Chem_Shift_T = chemshift_data[signal_names]
        Chem_Shift_T = Chem_Shift_T.apply(pd.to_numeric, errors='coerce')
        dq = np.asarray(Chem_Shift_T.to_numpy(dtype=float))
        mask = np.isfinite(dq)

        # Concentrations (C_T)
        C_T_df = conc_data[column_names]
        C_T = C_T_df.to_numpy(dtype=float)
        
        # Identify H and G
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

        if H is None and G is None:
             return {"error": "Could not identify Receptor or Guest columns."}
        if H is None: H = C_T[:, 0]
        
        nc = len(C_T)
        n_comp = C_T.shape[1]
        
        # Build D_cols (Analytical fallback)
        D_cols, parent_idx = build_D_cols(C_T_df, column_names, signal_names, default_idx=0)
        mask = mask & np.isfinite(D_cols)
        
        modelo = np.array(model_matrix).T # Transpose to match expected shape (n_species x n_components)
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
            
        # Validate Stoichiometry
        stoich_mat = None
        if stoichiometry_map is not None:
            try:
                stoich_mat = np.array(stoichiometry_map, dtype=float)
                if stoich_mat.shape[0] != modelo.shape[0]:
                    log_progress(f"Warning: Stoichiometry rows ({stoich_mat.shape[0]}) != Species ({modelo.shape[0]}). Ignoring.")
                    stoich_mat = None
                elif stoich_mat.shape[1] != dq.shape[1]:
                    if stoich_mat.shape[0] == dq.shape[1] and stoich_mat.shape[1] == modelo.shape[0]:
                         stoich_mat = stoich_mat.T
                    else:
                        log_progress(f"Warning: Stoichiometry shape {stoich_mat.shape} incompatible with Species/Signal ({modelo.shape[0]}, {dq.shape[1]}). Ignoring.")
                        stoich_mat = None
            except Exception as e:
                log_progress(f"Warning: Error reading stoichiometry: {e}")
                stoich_mat = None
        
    except Exception as e:
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
            
            # Weighted Residuals
            w_mat = weights_per_signal[None, :] 
            residuals_flat = diff[valid_residuals]
            weights_flat   = np.broadcast_to(w_mat, diff.shape)[valid_residuals]
            
            w_r2 = (residuals_flat**2) * weights_flat
            
            if (residuals_flat.size <= len(theta_free)) or (not np.isfinite(w_r2).all()):
                 return 1e9
                 
            # Weighted RMS
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
        except Exception:
            return 1e9

    # 5. Optimization (IRLS)
    MAX_IRLS_CYCLES = 5
    cycle_count = 0
    k_opt_full = p0_full.copy()
    old_best_rms = np.inf
    
    while cycle_count < MAX_IRLS_CYCLES:
        cycle_count += 1
        iter_state["best"] = np.inf # Reset for current weights
        log_progress(f"--- IRLS Cycle {cycle_count}/{MAX_IRLS_CYCLES} ---")
        param_opt_failed = False
        
        try:
            if free_idx.size == 0:
                k_opt_full = p0_full.copy()
            else:
                k_free0 = p0_full[free_idx]
                bounds_free = [bnds[i] for i in free_idx]

                if optimizer == "differential_evolution" and cycle_count == 1:
                     opt_res = differential_evolution(
                        f_m, bounds_free, x0=k_free0, strategy='best1bin',
                        maxiter=1000, popsize=15, tol=0.01
                    )
                else:
                    method = optimizer if optimizer != "differential_evolution" else "powell"
                    opt_res = optimize.minimize(f_m, k_free0, method=method, bounds=bounds_free)
                
                k_opt_full = pack(opt_res.x)
                p0_full = k_opt_full
                
        except Exception as e:
            log_progress(f"Optimization cycle failed: {e}")
            param_opt_failed = True
            break
            
        # Update Weights
        C_opt = res.concentraciones(k_opt_full)[0]
        dq_fit = project_coeffs_block_onp_frac(
             dq, C_opt, D_cols, mask, stoichiometry=stoich_mat
        )
        
        residuals_mat = dq - dq_fit
        # Sigma per signal
        new_weights = []
        for j in range(dq.shape[1]):
            mj = mask[:, j] & np.isfinite(dq_fit[:, j])
            rj = residuals_mat[mj, j]
            if rj.size > 2:
                sigma_j = np.sqrt(np.mean(rj**2))
                sigma_j = max(sigma_j, 1e-4) # floor
                w_j = 1.0 / (sigma_j**2)
            else:
                w_j = weights_per_signal[j]
            new_weights.append(w_j)
            
        new_weights = np.array(new_weights)
        diff_w = np.max(np.abs(weights_per_signal - new_weights) / (weights_per_signal + 1e-9))
        weights_per_signal = new_weights
        
        log_progress(f"Cycle {cycle_count} done. Max weight change: {diff_w:.2%}")
        
        # Convergence Check: RMS stability (as suggested by user)
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
        
        # Final calculated shifts
        dq_fit = project_coeffs_block_onp_frac(dq, C_opt, D_cols, mask, stoichiometry=stoich_mat, rcond=1e-10, ridge=1e-8)
        
        diff_final = dq - dq_fit
        valid_residuals_final = mask & np.isfinite(dq_fit)
        residuals_vec = diff_final[valid_residuals_final].ravel()

        # Custom Error Calculation for NMR
        SE_log10K_full = np.zeros_like(k_opt_full, dtype=float)
        percK_full = np.zeros_like(k_opt_full, dtype=float)
        covfit_val = np.nan

        try:
            k_names = [f"K{i+1}" for i in range(len(k_opt_full))]
            err_res = compute_errors_nmr_varpro(
                k_opt_full, res, dq, D_cols, modelo, nas,
                mask=mask,
                fixed_mask=fixed_mask,
                weights=weights_per_signal, # Use final weights
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
            log_progress(f"Error calculando errores: {e}")
            import traceback
            logger.error(traceback.format_exc())
            covfit_val = np.nan
        
        # For fixed parameters, always report zero uncertainty.
        SE_log10K_full[fixed_mask] = 0.0
        percK_full[fixed_mask] = 0.0
        
        rms = np.sqrt(np.mean(residuals_vec**2)) if residuals_vec.size > 0 else 0.0
        
        # Calculate additional statistics
        dq_vec = dq.ravel()
        dq_fit_vec = dq_fit.ravel()
        residuals_masked = residuals_vec
        
        SS_res = float(np.sum(residuals_masked**2))
        dq_obs = dq[valid_residuals_final]
        SS_tot = float(np.sum((dq_obs - np.mean(dq_obs))**2))
        
        lof = 0.0 if SS_tot <= 1e-30 else 100.0 * SS_res / SS_tot
        MAE = float(np.mean(np.abs(residuals_masked)))
        
        # Difference in total concentration (%)
        nc = len(C_opt)
        if H is not None:
            dif_en_ct = round(max(100 - (np.sum(C_opt, 1) * 100 / max(H))), 2)
        else:
            dif_en_ct = 0.0
        
        # Format Text Report
        results_text = format_results_table(
            k_opt_full,
            SE_log10K_full,
            percK_full,
            rms,
            covfit_val,
            lof=lof,
            fixed_mask=fixed_mask,
        )

        # Semáforo de Estabilidad
        if 'stability_diag' in locals() and stability_diag:
            status = stability_diag.get("status")
            summary = stability_diag.get("diag_summary", "")
            full = stability_diag.get("diag_full", "")

            if status == "critical":
                results_text += f"\\n\\n>>> CRITICAL WARNING: Ill-conditioned system ({summary}).\\nParameters might not be identifiable. Review correlations/model."
            elif status == "warn":
                results_text += f"\\n\\n>>> WARNING: Poor conditioning ({summary}).\\nHigh correlations might be present."
            
            if show_stability_diagnostics:
                results_text += f"\\n\\n{full}"

        derived_noncoop = None
        if str(model_settings) in ("Non-cooperative", "Statistical") and np.asarray(k_opt_full).size == 1:
            try:
                derived = noncoop_derived_from_logK1(
                    np.asarray(modelo, dtype=float),
                    float(np.asarray(k_opt_full).ravel()[0]),
                )
                derived_noncoop = {
                    "N": int(derived["N"]),
                    "logK_by_j": np.asarray(derived["logK_by_j"], dtype=float).tolist(),
                }

                lines = ["", "Derived (Non-cooperative):", "j | log10(Kj)"]
                for j in range(1, int(derived["N"]) + 1):
                    lines.append(
                        f"{j} | {derived['logK_by_j'][j-1]:.6g}"
                    )
                results_text += "\\n" + "\\n".join(lines)
            except Exception:
                derived_noncoop = None

        # Add extra statistics to results text
        extra_stats = [
            f"LOF: {lof:.2e} %",
            f"MAE: {MAE:.2e}",
            f"Diferencia en C total (%): {dif_en_ct:.2f}",
            f"Optimizer: {optimizer}",
            f"Algorithm: {algorithm}",
            f"Model settings: {model_settings}",
        ]
        results_text += "\\n\\nEstadísticas:\\n" + "\\n".join(extra_stats)
        
        # Graphs
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

        # Export Data
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

        # Build availablePlots
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
        import traceback
        traceback.print_exc()
        return {"error": f"Error calculating results: {str(e)}"}

# Helper functions for list_sheets and list_columns can remain or be imported
def list_sheets_from_bytes(file_bytes: bytes) -> List[str]:
    """Return available sheet names from an Excel workbook."""
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    return xl.sheet_names

def list_columns_from_bytes(file_bytes: bytes, sheet_name: str) -> List[str]:
    """Return column headers for a given sheet."""
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, nrows=0)
    return list(df.columns)
