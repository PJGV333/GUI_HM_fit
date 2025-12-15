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

# Add root directory to path to import algorithm modules
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import existing algorithms from the root directory
from NR_conc_algoritm import NewtonRaphson
from LM_conc_algoritm import LevenbergMarquardt
from errors import compute_errors_nmr_varpro

# Import helper functions from spectroscopy_processor (relative import)
from .spectroscopy_processor import (
    format_results_table, 
    _build_bounds_list, 
    _aliases_from_names,
    _alias_re,
    generate_figure_base64,
    generate_figure2_base64
)
from noncoop_utils import noncoop_derived_from_logK1

# --- Progreso vía callback opcional (WebSocket en main.py) ---
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

def project_coeffs_block_onp_frac(dq_block, C_block, D_cols, mask_block, rcond=1e-10, ridge=0.0):
    """
    Proyección por señal usando fracción molar del 'padre' de cada señal.
    Calcula los desplazamientos químicos de las especies (delta_species) y predice dq_calc.
    
    Args:
        dq_block: (n_points, n_signals) Observed chemical shifts (absolute).
        C_block: (n_points, n_species) Concentrations of all species.
        D_cols: (n_points, n_signals) Total concentration of the nucleus parent (e.g. H or G).
        mask_block: (n_points, n_signals) Boolean mask of observed data.
        
    Returns:
        dq_calc: (n_points, n_signals) Calculated chemical shifts.
    """
    m, nP = dq_block.shape
    dq_calc = np.full_like(dq_block, np.nan, dtype=float) # Default to NaN
    
    # Calculate fractions: F = C / D_cols
    # Note: D_cols is (n_points, n_signals), C_block is (n_points, n_species)
    # We need to handle this per signal because D_cols varies per signal (could be H or G)
    
    for j in range(nP):
        # Select valid points for this signal
        # Valid means:
        # 1. Observed in dq_block (mask_block)
        # 2. Parent concentration D_cols is finite and > 0
        # 3. Concentrations C_block are finite
        
        D = D_cols[:, j]
        valid_C = np.isfinite(C_block).all(axis=1)
        mj = mask_block[:, j] & valid_C & np.isfinite(D) & (np.abs(D) > 1e-12)
        
        nj = mj.sum()
        if nj < 2: # Need at least 2 points to fit? Or just > number of species?
             # If not enough points, we can't fit deltas for this signal.
             # Leave as NaN or 0? 
             continue

        # F_j = C[mj] / D[mj]  -> shape (nj, n_species)
        F_j = C_block[mj, :] / D[mj][:, None]
        
        y = dq_block[mj, j] # (nj,)
        
        # Solve F_j * delta_species = y
        try:
            delta_species, _, _, _ = np.linalg.lstsq(F_j, y, rcond=rcond)
        except np.linalg.LinAlgError:
             # Fallback
             if ridge > 0.0:
                 XtX = F_j.T @ F_j
                 delta_species = np.linalg.solve(XtX + ridge*np.eye(XtX.shape[0]), F_j.T @ y)
             else:
                 delta_species = np.linalg.pinv(F_j, rcond=rcond) @ y
        
        # Calculate predicted values for ALL points where we have concentrations
        # But we need D to be valid to calculate F
        # We can predict even where dq is missing, as long as we have C and D.
        
        # Prediction mask: where C and D are valid (regardless of dq observation)
        pred_mask = valid_C & np.isfinite(D) & (np.abs(D) > 1e-12)
        
        if pred_mask.any():
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
        # Convert numpy array to list and sanitize
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
        # dq1 = Chem_Shift_T - Chem_Shift_T.iloc[0] # REMOVED: Do not subtract first row
        
        # Convert to numeric, coercing errors to NaN
        Chem_Shift_T = Chem_Shift_T.apply(pd.to_numeric, errors='coerce')
        
        dq = np.asarray(Chem_Shift_T.to_numpy(dtype=float))
        mask = np.isfinite(dq)

        # Concentrations (C_T)
        C_T_df = conc_data[column_names]
        C_T = C_T_df.to_numpy(dtype=float)
        
        # Identify H and G
        H = None
        G = None
        
        # Simple matching based on labels provided
        # In a real scenario, we might need more robust matching if labels don't match exactly
        # But here we assume the frontend sends the exact column names selected
        
        # Find index of receptor and guest in the selected columns
        try:
            h_idx = column_names.index(receptor_label)
            H = C_T[:, h_idx]
        except ValueError:
            pass # Should not happen if frontend validation works
            
        try:
            g_idx = column_names.index(guest_label)
            G = C_T[:, g_idx]
        except ValueError:
            pass

        if H is None and G is None:
             return {"error": "Could not identify Receptor or Guest columns."}
        
        # Default H to first column if not found (fallback, though risky)
        if H is None: H = C_T[:, 0]
        
        nc = len(C_T)
        n_comp = C_T.shape[1]
        
        # Build D_cols
        D_cols, parent_idx = build_D_cols(C_T_df, column_names, signal_names, default_idx=0)
        mask = mask & np.isfinite(D_cols) & (np.abs(D_cols) > 0)
        
        modelo = np.array(model_matrix).T # Transpose to match expected shape (n_species x n_components)
        nas = non_absorbent_species
        
        k = np.array(k_initial, dtype=float).ravel()
        bnds = _build_bounds_list(k_bounds)

        fixed_mask = np.zeros(len(k), dtype=bool)
        if k_fixed is not None:
            for i, f in enumerate(k_fixed[: len(k)]):
                fixed_mask[i] = bool(f)

        # Compatibility: treat equal bounds as fixed
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
        
    except Exception as e:
         return {"error": f"Error preparing data matrices: {str(e)}"}

    log_progress("Iniciando procesamiento NMR…")
    log_progress(f"Optimizer: {optimizer} | Algorithm: {algorithm}")

    # 3. Initialize Algorithm
    try:
        # Algorithms need DataFrame, not numpy array
        if algorithm == "Newton-Raphson":
            res = NewtonRaphson(C_T_df, modelo, nas, model_settings)
        elif algorithm == "Levenberg-Marquardt":
            res = LevenbergMarquardt(C_T_df, modelo, nas, model_settings)
        else:
            return {"error": f"Unknown algorithm: {algorithm}"}
    except Exception as e:
        return {"error": f"Error initializing algorithm: {str(e)}"}

    # 4. Define Objective Function
    iter_state = {"cnt": 0, "best": np.inf}

    def f_m(theta_free):
        try:
            iter_state["cnt"] += 1
            k_curr_full = pack(theta_free)
            C = res.concentraciones(k_curr_full)[0]
            dq_cal = project_coeffs_block_onp_frac(
                dq, C, D_cols, mask, rcond=1e-10, ridge=1e-8
            )
            
            # Residuals only on observed points
            # dq and dq_cal should match on mask
            # Note: dq_cal might be NaN where mask is False, or even where mask is True if fit failed for that signal
            
            diff = dq - dq_cal
            # Filter by mask AND where we actually got a calculation
            valid_residuals = mask & np.isfinite(dq_cal)
            
            r = diff[valid_residuals].ravel()
            if (r.size <= len(np.asarray(theta_free).ravel())) or (not np.isfinite(r).all()):
                log_progress(f"Iter {iter_state['cnt']}: evaluación descartada (Nobs≤p o residuales no finitos)")
                return 1e9
            rms = float(np.sqrt(np.mean(r * r)))

            best = iter_state["best"]
            should_log = (
                iter_state["cnt"] == 1
                or (iter_state["cnt"] % 5 == 0)
                or (rms < best * 0.999)
            )
            if should_log:
                log_progress(
                    f"Iter {iter_state['cnt']}: f(x)={rms:.6e} | x={[float(xi) for xi in k_curr_full]}"
                )
                iter_state["best"] = min(best, rms)
            return rms
        except Exception:
            return 1e9

    # 5. Optimization
    try:
        if free_idx.size == 0:
            k_opt_full = p0_full.copy()
            log_progress("Optimización omitida: todas las constantes están fijas.")
        else:
            k_free0 = p0_full[free_idx]
            bounds_free = [bnds[i] for i in free_idx]

            if optimizer == "differential_evolution":
                for (lb, ub) in bounds_free:
                    if np.isinf(lb) or np.isinf(ub):
                        return {"error": "Differential Evolution requires finite bounds for free parameters. Please set Min/Max (or mark as Fixed)."}

                opt_res = differential_evolution(
                    f_m,
                    bounds_free,
                    x0=k_free0,
                    strategy='best1bin',
                    maxiter=1000,
                    popsize=15,
                    tol=0.01,
                    mutation=(0.5, 1),
                    recombination=0.7,
                    init='latinhypercube'
                )
            else:
                opt_res = optimize.minimize(f_m, k_free0, method=optimizer, bounds=bounds_free)

            k_opt_full = pack(opt_res.x)
            log_progress(f"Optimización completada (iter={iter_state['cnt']}, best_f={iter_state['best']:.6e})")
        
    except Exception as e:
        return {"error": f"Optimization failed: {str(e)}"}

    # 6. Calculate Results & Statistics
    try:
        # Final concentrations
        C_opt, Co_opt = res.concentraciones(k_opt_full)
        
        # Final calculated shifts
        # Note: We use project_coeffs_block_onp_frac for consistency with f_m
        dq_fit = project_coeffs_block_onp_frac(dq, C_opt, D_cols, mask)
        
        # Calculate Errors
        # We need to adapt compute_errors_nmr_varpro or implement a custom one here
        # because we changed how residuals are calculated (absolute shifts)
        
        # Custom Error Calculation for NMR (HypNMR style)
        # 1. Jacobian J = d(residual)/dK
        # 2. Residuals vector r
        # 3. s^2 = r.T @ r / (N_res - N_par)
        # 4. Cov = s^2 * (J.T @ J)^-1
        
        # Re-calculate residuals for final K
        diff_final = dq - dq_fit
        valid_residuals_final = mask & np.isfinite(dq_fit)
        residuals_vec = diff_final[valid_residuals_final].ravel()
        
        N_res = residuals_vec.size
        N_par = int(free_idx.size)
        dof = N_res - N_par

        SE_log10K_full = np.zeros_like(k_opt_full, dtype=float)
        percK_full = np.zeros_like(k_opt_full, dtype=float)
        covfit_val = np.nan

        # If non-absorbent species are selected, use VarPro-based error estimates.
        if nas and len(nas) > 0:
            try:
                err_res = compute_errors_nmr_varpro(
                    k_opt_full, res, dq, H, modelo, nas, mask=mask
                )
                J_full = np.asarray(err_res.get("J", np.zeros((len(k_opt_full), 0))), dtype=float)
                rms_varpro = float(err_res.get("rms", np.nan))
                rms = rms_varpro if np.isfinite(rms_varpro) else rms

                if J_full.ndim == 2 and J_full.shape[0] == len(k_opt_full):
                    J_free = J_full[free_idx, :] if free_idx.size else np.zeros((0, J_full.shape[1]), dtype=float)
                    m_eff = int(J_free.shape[1])
                    dof_free = max(m_eff - N_par, 1)
                    sse = float((rms * rms) * m_eff) if np.isfinite(rms) else float(np.sum(residuals_vec**2))
                    s2 = sse / dof_free
                    covfit_val = float(s2)

                    if N_par > 0:
                        JtJ = J_free @ J_free.T
                        cov_k = np.linalg.pinv(JtJ)
                        covfit_mat = s2 * cov_k
                        SE_log10K_free = np.sqrt(np.clip(np.diag(covfit_mat), 0.0, np.inf))
                        percK_free = 100.0 * np.log(10.0) * SE_log10K_free
                        SE_log10K_full[free_idx] = SE_log10K_free
                        percK_full[free_idx] = percK_free
                    else:
                        SE_log10K_full[:] = 0.0
                        percK_full[:] = 0.0
                else:
                    covfit_val = float(err_res.get("covfit", np.nan))
            except Exception as e:
                log_progress(f"Error calculando errores (VarPro): {e}")
                covfit_val = np.nan
        else:
            SS_res = float(np.sum(residuals_vec**2)) if residuals_vec.size else 0.0
            s2 = (SS_res / dof) if dof > 0 else np.nan

            if dof > 0 and N_par > 0:
                eps = 1e-8
                J = np.zeros((N_res, N_par))

                for j, idx in enumerate(free_idx):
                    k_temp = k_opt_full.copy()
                    k_temp[idx] += eps * max(abs(k_temp[idx]), 1.0)
                    step = k_temp[idx] - k_opt_full[idx]

                    try:
                        C_temp, _ = res.concentraciones(k_temp)
                        dq_fit_temp = project_coeffs_block_onp_frac(dq, C_temp, D_cols, mask)
                        diff_temp = dq - dq_fit_temp
                        r_temp = diff_temp[valid_residuals_final].ravel()
                        J[:, j] = (r_temp - residuals_vec) / step
                    except Exception:
                        J[:, j] = 0.0

                try:
                    JtJ = J.T @ J
                    cov_k = np.linalg.pinv(JtJ)
                    covfit_mat = s2 * cov_k
                    sigma_k = np.sqrt(np.diag(covfit_mat))

                    denom_log = np.log(10) * np.abs(k_opt_full[free_idx])
                    with np.errstate(divide='ignore', invalid='ignore'):
                        SE_log10K_free = np.where(denom_log > 0, sigma_k / denom_log, np.nan)
                        percK_free = np.where(np.abs(k_opt_full[free_idx]) > 0, (sigma_k / np.abs(k_opt_full[free_idx])) * 100.0, np.nan)

                    covfit_val = float(s2)
                    SE_log10K_full[free_idx] = SE_log10K_free
                    percK_full[free_idx] = percK_free
                except Exception as e:
                    log_progress(f"Error calculando errores: {e}")
                    covfit_val = float(s2)
            else:
                covfit_val = float(s2) if np.isfinite(s2) else np.nan

        # For fixed parameters, always report zero uncertainty.
        SE_log10K_full[fixed_mask] = 0.0
        percK_full[fixed_mask] = 0.0
            
        rms = np.sqrt(np.mean(residuals_vec**2)) if residuals_vec.size > 0 else 0.0
        
        # Calculate additional statistics
        # LOF (Lack of Fit) calculation
        dq_vec = dq.ravel()
        dq_fit_vec = dq_fit.ravel()
        residuals = dq_vec - dq_fit_vec
        
        # Filter by mask
        residuals_masked = residuals_vec # Already computed above
        
        SS_res = float(np.sum(residuals_masked**2))
        # LOF: compare to variance of data? Or just SS_res?
        # Standard LOF definition usually requires replicates. 
        # Here we just use SS_res / SS_tot as a % of unexplained variance
        
        # SS_tot should be calculated on the observed data
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
                results_text += "\n" + "\n".join(lines)
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
        results_text += "\n\nEstadísticas:\n" + "\n".join(extra_stats)
        
        # Prepare Graphs Data (Base64 images using matplotlib)
        graphs = {}
        
        # Graph 1: Concentrations Profile
        # X axis: [G] or [H] depending on what varies more or user preference
        x_axis_conc = G if G is not None else H
        x_label_conc = f"[{guest_label}] Total (M)" if G is not None else f"[{receptor_label}] Total (M)"
        
        graphs['concentrations'] = generate_figure_base64(
            x_axis_conc, C_opt, ":o", "[Species], M", x_label_conc, "Concentration Profile"
        )
        
        # Graph 2: Chemical Shifts Fit
        # Plot observed vs calculated for each signal
        # We can plot them all in one or separate. wxPython plots them together usually.
        # generate_figure2_base64 takes (x, y1, y2, ...)
        # Here we have multiple signals. generate_figure2_base64 handles matrices.
        
        graphs['fit'] = generate_figure2_base64(
            x_axis_conc, dq, dq_fit, "o", ":", "δ (ppm)", x_label_conc, 0.5, "Chemical Shifts Fit"
        )
        
        # Graph 3: Residuals
        residuals = dq - dq_fit
        graphs['residuals'] = generate_figure_base64(
            x_axis_conc, residuals, "o", "Residuals (ppm)", x_label_conc, "Residuals"
        )

        # Prepare export data structure (matching Spectroscopy pattern)
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
        }

        log_progress("Procesamiento NMR completado.")

        # Build availablePlots list (ordered pages for carousel navigation)
        availablePlots = []
        
        # Chemical shifts fit - interactive (kind: plotly)
        if dq is not None:
            availablePlots.append({
                "id": "nmr_shifts_fit",
                "title": "Chemical shifts fit",
                "kind": "plotly"
            })
        
        # Species distribution - interactive (kind: plotly)
        if C_opt is not None and len(C_opt) > 0:
            availablePlots.append({
                "id": "nmr_species_distribution",
                "title": "Species distribution",
                "kind": "plotly"
            })
        
        # Residuals - interactive (kind: plotly)
        if dq is not None:
            availablePlots.append({
                "id": "nmr_residuals",
                "title": "Residuals",
                "kind": "plotly"
            })
        
        # X axis for all plots
        x_axis_values = (G if G is not None else H).tolist() if (G is not None or H is not None) else []
        x_label = f"[{guest_label}] Total (M)" if G is not None else f"[{receptor_label}] Total (M)"
        
        # Build signal options with stable IDs and labels
        def make_signal_id(name):
            """Create stable ID from signal name"""
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
        
        # Build species options with id/label for species distribution
        species_labels = [f"sp{j+1}" for j in range(C_opt.shape[1])] if C_opt is not None else []
        species_options = [{"id": sp, "label": sp} for sp in species_labels]
        
        # Build C_by_species for direct lookup
        C_by_species = {}
        for j, sp_label in enumerate(species_labels):
            C_by_species[sp_label] = C_opt[:, j].tolist() if C_opt is not None else []
        
        # Build axis options for species distribution
        axis_options = [{"id": "titrant_total", "label": x_label}]
        axis_vectors = {"titrant_total": x_axis_values}
        
        for j, sp_label in enumerate(species_labels):
            axis_options.append({"id": f"species:{sp_label}", "label": f"[{sp_label}]"})
            axis_vectors[f"species:{sp_label}"] = C_opt[:, j].tolist() if C_opt is not None else []
        
        # Build plotData with arrays for interactive plots
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
            "graphs": graphs,  # legacy PNG graphs
            "availablePlots": availablePlots,
            "plotData": {"nmr": nmr_plot_data},
            "export_data": export_data,
            "constants": constants,
            "derived_noncoop": export_data.get("derived_noncoop"),
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
