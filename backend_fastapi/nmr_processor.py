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
    """Proyección por señal usando fracción molar del 'padre' de cada señal."""
    m, nP = dq_block.shape
    dq_calc = np.zeros_like(dq_block, float)
    Xbase = np.asarray(C_block, float)
    finite_rows = np.isfinite(Xbase).all(axis=1)
    nonzero_rows = np.linalg.norm(Xbase, axis=1) > 0.0
    goodX = finite_rows & nonzero_rows
    for j in range(nP):
        D = D_cols[:, j]
        mj = mask_block[:, j] & goodX & np.isfinite(D) & (np.abs(D) > 0)
        if mj.sum() < 2: continue
        Xj = Xbase[mj, :] / D[mj][:, None]
        y  = dq_block[mj, j]
        try:
            coef, *_ = np.linalg.lstsq(Xj, y, rcond=rcond)
        except np.linalg.LinAlgError:
            if ridge > 0.0:
                XtX = Xj.T @ Xj
                coef = np.linalg.solve(XtX + ridge*np.eye(XtX.shape[0]), Xj.T @ y)
            else:
                coef = np.linalg.pinv(Xj, rcond=rcond) @ y
        dq_calc[mj, j] = Xj @ coef
    return dq_calc

def project_coeffs_block_onp(dq_block, X_block, mask_block, rcond=1e-10, ridge=0.0):
    """
    Ajusta coeficientes por señal usando SOLO filas válidas.
    """
    m, nP = dq_block.shape
    dq_calc = np.zeros_like(dq_block, dtype=float)

    # filas de X que son numéricamente utilizables
    finite_rows = np.isfinite(X_block).all(axis=1)
    nonzero_rows = np.linalg.norm(X_block, axis=1) > 0.0
    good_rows_X = finite_rows & nonzero_rows

    for j in range(nP):
        mj = mask_block[:, j] & good_rows_X
        nj = int(mj.sum())
        if nj < 2:
            continue  # no hay suficientes puntos para esa señal

        X = X_block[mj, :]      # (nj × n_abs)
        y = dq_block[mj, j]     # (nj,)

        try:
            # camino estable
            coef, *_ = np.linalg.lstsq(X, y, rcond=rcond)
        except np.linalg.LinAlgError:
            if ridge > 0.0:
                # ridge mínimo si lstsq no converge
                XtX = X.T @ X
                coef = np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), X.T @ y)
            else:
                # fallback pinv
                coef = np.linalg.pinv(X, rcond=rcond) @ y

        # predicción solo en filas observadas
        dq_calc[mj, j] = X @ coef

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
        dq1 = Chem_Shift_T - Chem_Shift_T.iloc[0]
        dq = np.asarray(dq1.to_numpy(dtype=float))
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
        
        k = np.array(k_initial)
        bnds = _build_bounds_list(k_bounds)
        
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

    def f_m(k_curr):
        try:
            iter_state["cnt"] += 1
            C = res.concentraciones(k_curr)[0]
            dq_cal = project_coeffs_block_onp_frac(
                dq, C, D_cols, mask, rcond=1e-10, ridge=1e-8
            )
            r = (dq - dq_cal)[mask].ravel()
            if (r.size <= len(k_curr)) or (not np.isfinite(r).all()):
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
                    f"Iter {iter_state['cnt']}: f(x)={rms:.6e} | x={[float(xi) for xi in k_curr]}"
                )
                iter_state["best"] = min(best, rms)
            return rms
        except Exception:
            return 1e9

    # 5. Optimization
    try:
        if optimizer == "differential_evolution":
            # Check for infinite bounds
            for (lb, ub) in bnds:
                if np.isinf(lb) or np.isinf(ub):
                     return {"error": "Differential Evolution requires finite bounds. Please set Min/Max for all parameters."}
            
            opt_res = differential_evolution(
                f_m, bnds, x0=k, strategy='best1bin',
                maxiter=1000, popsize=15, tol=0.01,
                mutation=(0.5, 1), recombination=0.7,
                init='latinhypercube'
            )
        else:
            opt_res = optimize.minimize(f_m, k, method=optimizer, bounds=bnds)
            
        k_opt = opt_res.x
        k_opt = np.ravel(k_opt)
        log_progress(f"Optimización completada (iter={iter_state['cnt']}, best_f={iter_state['best']:.6e})")
        
    except Exception as e:
        return {"error": f"Optimization failed: {str(e)}"}

    # 6. Calculate Results & Statistics
    try:
        # Final concentrations
        C_opt, Co_opt = res.concentraciones(k_opt)
        
        # Final calculated shifts
        # Note: We use project_coeffs_block_onp_frac for consistency with f_m
        dq_fit = project_coeffs_block_onp_frac(dq, C_opt, D_cols, mask)
        
        # Calculate Errors
        metrics = compute_errors_nmr_varpro(
            k=k_opt, res=res, dq=dq, H=H, modelo=modelo, nas=nas,
            rcond=1e-10, use_projector=True, mask=mask
        )
        
        # Extract metrics
        percK = metrics.get("percK", np.zeros_like(k_opt))
        SE_log10K = metrics.get("SE_log10K", np.zeros_like(k_opt))
        rms = metrics.get("rms", 0.0)
        covfit = metrics.get("covfit", 0.0)
        
        # Calculate additional statistics
        # LOF (Lack of Fit) calculation
        dq_vec = dq.ravel()
        dq_fit_vec = dq_fit.ravel()
        residuals = dq_vec - dq_fit_vec
        
        # Filter by mask
        residuals_masked = (dq - dq_fit)[mask].ravel()
        
        SS_res = float(np.sum(residuals_masked**2))
        SS_tot = float(np.sum((dq[mask].ravel() - np.mean(dq[mask].ravel()))**2))
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
            k_opt, SE_log10K, percK, rms, covfit, lof=lof
        )
        
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
            x_axis_conc, dq, dq_fit, "o", ":", "Δδ (ppm)", x_label_conc, 0.5, "Chemical Shifts Fit"
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
            "k": k_opt.tolist() if k_opt is not None else [],
            "k_ini": k_initial if k_initial is not None else [],
            "percK": percK.tolist() if percK is not None else [],
            "SE_log10K": SE_log10K.tolist() if SE_log10K is not None else [],
            "signal_names": signal_names,
            "column_names": column_names,
            "stats_table": [
                ["RMS", float(rms)],
                ["Error absoluto medio", float(MAE)],
                ["Diferencia en C total (%)", float(dif_en_ct)],
                ["covfit", float(covfit)],
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

        return sanitize_for_json({
            "success": True,
            "results_text": results_text,
            "graphs": graphs,  # legacy PNG graphs
            "availablePlots": availablePlots,
            "plotData": {"nmr": nmr_plot_data},
            "export_data": export_data
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
