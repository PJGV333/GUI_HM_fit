"""
Spectroscopy processor module - refactored from Spectroscopy_controls.py
Extracted business logic without wx dependencies for use in FastAPI backend.
"""
import io
import base64
import numpy as onp
from np_backend import xp as np, jit, jacrev, vmap, lax
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import optimize
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings("ignore")
from errors import compute_errors_spectro_varpro, pinv_cs, percent_error_log10K, sensitivities_wrt_logK
from core_ad_probe import solve_A_nnls_pgd

# === Result formatting (shared with wx reference) ===
def format_results_table(k, SE_log10K, percK, rms, covfit, lof=None):
    """
    Build an ASCII table with aligned columns for constants and diagnostics.
    Mirrors the wxPython formatting used in NMR_controls.
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

    rows = [
        [
            f"K{i+1}",
            f"{k[i]:.2e} ± {SE_log10K[i]:.2e}",
            f"{percK[i]:.2f} %",
            f"{rms:.2e}" if i == 0 else "",
            f"{covfit:.2e}" if i == 0 else "",
        ]
        for i in range(len(k))
    ]

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

# Progress callback for WebSocket streaming
_progress_callback = None
_loop = None

def set_progress_callback(callback, loop=None):
    """Set the callback function for progress updates."""
    global _progress_callback, _loop
    _progress_callback = callback
    _loop = loop

def log_progress(message):
    """Send progress message through callback if set."""
    if _progress_callback:
        if _loop:
            # If loop is provided, schedule callback safely
            _loop.call_soon_threadsafe(_progress_callback, message)
        else:
            # Direct call (synchronous)
            _progress_callback(message)
    print(message)  # Also print to console

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

def generate_figure_base64(x, y, mark, ylabel, xlabel, title):
    """Generate a matplotlib figure and return as base64 encoded PNG."""
    try:
        # Debug shapes
        x_shape = np.shape(x)
        y_shape = np.shape(y)
        print(f"DEBUG plot '{title}': x={x_shape}, y={y_shape}")
    except:
        pass

    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # Normaliza dimensiones para evitar mismatches (se comporta como figura de wx)
    x_arr = np.asarray(x).reshape(-1)
    y_arr = np.asarray(y)

    if y_arr.ndim == 1:
        ax.plot(x_arr, y_arr, mark)
    elif y_arr.ndim == 2:
        if y_arr.shape[0] == x_arr.shape[0]:
            for i in range(y_arr.shape[1]):
                ax.plot(x_arr, y_arr[:, i], mark)
        elif y_arr.shape[1] == x_arr.shape[0]:
            for i in range(y_arr.shape[0]):
                ax.plot(x_arr, y_arr[i, :], mark)
        elif y_arr.size == x_arr.shape[0]:
            ax.plot(x_arr, y_arr.reshape(-1), mark)
        else:
            m = min(x_arr.shape[0], y_arr.shape[0])
            ax.plot(x_arr[:m], y_arr[:m, 0], mark)
    else:
        ax.plot(x_arr, y_arr.reshape(-1), mark)
    
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
    plt.close(fig)
    
    return img_base64

def generate_figure2_base64(x, y, y2, mark1, mark2, ylabel, xlabel, alpha, title):
    """Generate a matplotlib figure with two series and return as base64 encoded PNG."""
    try:
        # Debug shapes
        x_shape = np.shape(x)
        y_shape = np.shape(y)
        y2_shape = np.shape(y2)
        print(f"DEBUG plot2 '{title}': x={x_shape}, y={y_shape}, y2={y2_shape}")
    except:
        pass

    fig = Figure(figsize=(8, 6), dpi=100)
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
    plt.close(fig)
    
    return img_base64

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
    algorithm,
    model_settings,
    optimizer,
    initial_k,
    bounds
):
    """
    Main processing function refactored from Spectroscopy_controls.process_data
    Returns dict with results and base64-encoded graphs.
    """
    log_progress("Iniciando procesamiento...")
    
    # Read Excel data
    spec = pd.read_excel(file_path, spectra_sheet, header=0, index_col=0)
    nm = spec.index.to_numpy()
    
    concentracion = pd.read_excel(file_path, conc_sheet, header=0)
    C_T = concentracion[column_names].to_numpy()
    
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
    
    nc = len(C_T)
    n_comp = len(C_T.T)
    nw = len(spec)
    
    graphs = {}
    
    # SVD/EFA function
    def SVD_EFA(spec, nc):
        u, s, v = onp.linalg.svd(spec, full_matrices=False)
        
        L = range(1, (nc + 1), 1)
        L2 = range(0, nc, 1)
        
        X = []
        for i in L:
            uj, sj, vj = onp.linalg.svd(spec.T.iloc[:i,:], full_matrices=False)
            X.append(sj)
        
        ev_s = pd.DataFrame(X)
        ev_s0 = onp.array(ev_s)
        
        X2 = []
        for i in L2:
            ui, si, vi = onp.linalg.svd(spec.T.iloc[i:,:], full_matrices=False)
            X2.append(si)
        
        ev_s1 = pd.DataFrame(X2)
        ev_s10 = np.array(ev_s1)
        
        # Generate graphs
        # Evita mismatch de dimensiones: x del tamaño real de s
        graphs['eigenvalues'] = generate_figure_base64(
            range(len(s)), np.log10(s), "o", "log(EV)", "# de autovalores", "Eigenvalues"
        )
        
        if G is not None:
            graphs['efa'] = generate_figure2_base64(
                G, np.log10(ev_s0), np.log10(ev_s10), "k-o", "b:o", 
                "log(EV)", "[G], M", 1, "EFA"
            )
        
        EV = efa_eigenvalues if efa_eigenvalues > 0 else nc
        log_progress(f"Eigenvalues used: {EV}")
        
        Y = u[:,0:EV] @ np.diag(s[0:EV:]) @ v[0:EV:]
        return Y, EV
    
    if efa_enabled:
        Y, EV = SVD_EFA(spec, nc)
    else:
        Y = np.array(spec)
        EV = nc
    
    C_T_df = pd.DataFrame(C_T)
    modelo = np.array(modelo).T if isinstance(modelo, list) else np.array(modelo)
    nas = non_abs_species

    # ---- Inicialización de parámetros y límites (replica flujo wx) ----
    def _safe_float_list(seq):
        vals = []
        for v in seq:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        return vals

    k = np.asarray(_safe_float_list(initial_k), dtype=float)

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
    
    # Select algorithm
    if algorithm == "Newton-Raphson":
        from NR_conc_algoritm import NewtonRaphson
        res = NewtonRaphson(C_T_df, modelo, nas, model_settings)
    elif algorithm == "Levenberg-Marquardt":
        from LM_conc_algoritm import LevenbergMarquardt
        res = LevenbergMarquardt(C_T_df, modelo, nas, model_settings)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Objective functions
    def f_m2(k):
        C = res.concentraciones(k)[0]
        A = solve_A_nnls_pgd(C, Y.T, ridge=0.0, max_iters=300)
        r = C @ A - Y.T
        rms = np.sqrt(np.mean(np.square(r)))
        return rms, r
    
    def f_m(k):
        # Handle potential NaNs/Infs in parameters
        if np.any(np.isnan(k)) or np.any(np.isinf(k)):
            return 1e50

        try:
            C = res.concentraciones(k)[0]
            # Check for NaNs in concentration matrix
            if np.any(np.isnan(C)) or np.any(np.isinf(C)):
                return 1e50
                
            A = solve_A_nnls_pgd(C, Y.T, ridge=0.0, max_iters=300)
            r = C @ A - Y.T
            rms = np.sqrt(np.mean(np.square(r)))
            
            if np.isnan(rms) or np.isinf(rms):
                return 1e50
                
            # Log progress (throttled or every N iterations if needed, but here we log all for debugging)
            # log_progress(f"f(x): {float(rms):.6e}") 
            return rms
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1e50
    
    # Best-so-far tracking
    best_result = {"rms": float('inf'), "k": np.copy(k)}

    # Optimization callback
    def callback_log(xk, convergence=None):
        # convergence arg is for differential_evolution, minimize passes only xk
        val = f_m(xk)
        
        # Update best
        if val < best_result["rms"]:
            best_result["rms"] = val
            best_result["k"] = np.copy(xk)
            
        log_progress(f"Iter: f(x)={val:.6e} | x={[float(xi) for xi in xk]}")

    # Optimization
    log_progress(f"Optimizer: {optimizer}")
    log_progress(f"Bounds (procesados): {processed_bounds}")  # keep visible for debugging powell with ±inf
    
    if optimizer == "differential_evolution":
        # differential_evolution requiere límites finitos
        for (min_val, max_val) in processed_bounds:
            if np.isinf(min_val) or np.isinf(max_val):
                msg = "Differential evolution requires all bounds to be finite. Please set Min/Max for each parameter."
                log_progress(msg)
                raise ValueError(msg)
        
        r_0 = differential_evolution(
            f_m,
            processed_bounds,
            x0=k,
            strategy='best1bin',
            maxiter=1000,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            init='latinhypercube',
            callback=callback_log,
        )
    else:
        r_0 = optimize.minimize(f_m, k, method=optimizer, bounds=processed_bounds, callback=callback_log)
    
    # Check if final result is better or worse than best seen
    final_rms = f_m(r_0.x)
    if final_rms > best_result["rms"]:
        log_progress(f"Warning: Final result (RMS={final_rms:.6e}) is worse than best seen (RMS={best_result['rms']:.6e}). Restoring best parameters.")
        k = best_result["k"]
    else:
        k = r_0.x

    k = np.ravel(k)
    
    log_progress("Optimización completada")
    
    # Compute errors
    metrics = compute_errors_spectro_varpro(
        k=k, res=res, Y=Y, modelo=modelo, nas=nas,
        rcond=1e-10, use_projector=True
    )
    
    SE_log10K = metrics["SE_log10K"]
    SE_K = metrics["SE_K"]
    percK = metrics["percK"]
    rms = metrics["RMS"]
    covfit = metrics["s2"]
    A = metrics["A"]
    yfit = metrics["yfit"]
    
    C, Co = res.concentraciones(k)
    
    # Generate concentration and spectra plots
    if n_comp == 1 and H is not None:
        graphs['concentrations'] = generate_figure_base64(
            H, C, ":o", "[Especies], M", "[H], M", "Perfil de concentraciones"
        )
        
        y_cal = C @ np.linalg.pinv(C) @ Y.T
        ssq, r0 = f_m2(k)
        A_plot = solve_A_nnls_pgd(C, Y.T, ridge=0.0, max_iters=300)
        
        graphs['absorptivities'] = generate_figure_base64(
            nm, A_plot.T, "-", "Epsilon (u. a.)", "$\\lambda$ (nm)", "Absortividades molares"
        )
        graphs['fit'] = generate_figure2_base64(
            nm, Y, y_cal.T, "-k", "k:", "Y observada (u. a.)", "$\\lambda$ (nm)", 0.5, "Ajuste"
        )
    elif G is not None:
        graphs['concentrations'] = generate_figure_base64(
            G, C, ":o", "[Species], M", "[G], M", "Perfil de concentraciones"
        )
        
        y_cal = C @ np.linalg.pinv(C) @ Y.T
        ssq, r0 = f_m2(k)
        A_plot = solve_A_nnls_pgd(C, Y.T, ridge=0.0, max_iters=300)
        
        if not efa_enabled:
            graphs['fit'] = generate_figure2_base64(
                G, Y.T, y_cal, "ko", ":", "Y observada (u. a.)", "[X], M", 1, "Ajuste"
            )
        else:
            graphs['absorptivities'] = generate_figure_base64(
                nm, A_plot.T, "-", "Epsilon (u. a.)", "$\\lambda$ (nm)", "Absortividades molares"
            )
            graphs['fit'] = generate_figure2_base64(
                nm, Y, y_cal.T, "-k", "k:", "Y observada (u. a.)", "$\\lambda$ (nm)", 0.5, "Ajuste"
            )
    
    ssq, r0 = f_m2(k)
    
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

    # Tabla formateada para resultados (alineada como en wx)
    results_text = format_results_table(k, SE_log10K, percK, rms, covfit, lof=lof)
    # Estadísticas sin duplicar RMS/s² (ya aparecen en la tabla)
    extra_stats = [
        f"LOF: {lof:.2e} %",
        f"MAE: {MAE:.2e}",
        f"Diferencia en C total (%): {dif_en_ct:.2f}",
        f"Eigenvalues: {EV}",
        f"Optimizer: {optimizer}",
    ]
    results_text += "\n\nEstadísticas:\n" + "\n".join(extra_stats)
    
    # Export payload to mimic wx save_results (DataFrames per sheet)
    # Preparar payload de exportación (imitando wx save_results)
    A_export = None
    nm_list = nm.tolist() if hasattr(nm, "tolist") else []
    if A is not None:
        A_arr = np.asarray(A)
        if nm_list:
            if A_arr.shape[1] == len(nm_list):
                A_export = A_arr.T  # filas = nm
            else:
                A_export = A_arr
        else:
            A_export = A_arr

    export_data = {
        "modelo": modelo.tolist() if modelo is not None else [],
        "C": np.asarray(C).tolist() if C is not None else [],
        "Co": np.asarray(Co).tolist() if Co is not None else [],
        "C_T": np.asarray(C_T).tolist() if C_T is not None else [],
        "A": A_export.tolist() if A_export is not None else [],
        "A_index": nm_list,
        "k": np.asarray(k).tolist(),
        "k_ini": np.asarray(initial_k).tolist() if initial_k is not None else [],
        "percK": np.asarray(percK).tolist(),
        "SE_log10K": np.asarray(SE_log10K).tolist(),
        "nm": nm_list,
        "Y": np.asarray(Y).tolist() if Y is not None else [],
        "yfit": np.asarray(yfit).tolist() if yfit is not None else [],
        "stats_table": [
            ["RMS", float(rms)],
            ["Error absoluto medio", float(MAE)],
            ["Diferencia en C total (%)", float(dif_en_ct)],
            ["covfit", float(covfit)],
            ["optimizer", optimizer],
        ],
    }

    # Format results
    results = {
        "success": True,
        "constants": [
            {
                "name": f"K{i+1}",
                "log10K": float(k[i]),
                "SE_log10K": float(SE_log10K[i]),
                "K": float(10**k[i]),
                "SE_K": float(SE_K[i]),
                "percent_error": float(percK[i])
            }
            for i in range(len(k))
        ],
        "statistics": {
            "RMS": float(rms),
            "lof": float(lof),
            "MAE": float(MAE),
            "dif_en_ct": float(dif_en_ct),
            "eigenvalues": int(EV),
            "covfit": float(covfit),
            "optimizer": optimizer
        },
        "graphs": graphs,
        "results_text": results_text,
        "export_data": export_data,
        "optimizer_result": {
            "success": bool(r_0.success),
            "message": str(r_0.message) if hasattr(r_0, 'message') else "",
            "nfev": int(r_0.nfev) if hasattr(r_0, 'nfev') else 0
        }
    }
    
    log_progress("Procesamiento completado exitosamente")
    
    return results
