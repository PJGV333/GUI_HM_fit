"""
Spectroscopy processor module.
Extracted business logic for use in FastAPI backend.
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
import re
import logging
warnings.filterwarnings("ignore")
from errors import compute_errors_spectro_varpro, pinv_cs, percent_error_log10K, sensitivities_wrt_logK
from core_ad_probe import solve_A_nnls_pgd

logger = logging.getLogger(__name__)

# === Progress tracking (WebSocket support) ===
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


# === Result formatting (shared with wx reference) ===
def format_results_table(k, SE_log10K, percK, rms, covfit, lof=None):
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
    Main processing function.
    Returns dict with results and base64-encoded graphs.
    """
    log_lines = []

    def log(msg: str):
        try:
            log_lines.append(str(msg))
        except Exception:
            pass
        log_progress(str(msg))

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
        return Y, EV, s, ev_s0, ev_s10
    
    eigenvalues = None
    efa_forward = None
    efa_backward = None
    if efa_enabled:
        Y, EV, eigenvalues, efa_forward, efa_backward = SVD_EFA(spec, nc)
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
            
            log(f"Iter: f(x)={val:.6e} | x={[float(xi) for xi in xk]}")

    # Optimization
    log(f"Optimizer: {optimizer}")
    log(f"Bounds (procesados): {processed_bounds}")  # keep visible for debugging powell with ±inf
    
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
    
    log("Optimización completada")
    
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
    species_labels = [f"sp{i+1}" for i in range(C.shape[1])] if C is not None else []
    
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
    axis_options = [{"id": "titrant_total", "label": f"[{guest_label}] total"}]
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
        "spec_molar_absorptivities": {"png_base64": graphs.get('absorptivities', '')},
        "spec_efa_eigenvalues": {"png_base64": graphs.get('eigenvalues', '')},
        "spec_efa_components": {"png_base64": graphs.get('efa', '')},
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
        "graphs": legacy_graphs,
        "legacy_graphs": legacy_graphs,
        "plot_data": plot_data,
        "availablePlots": availablePlots,
        "plotData": {"spec": spec_plot_data},
        "results_text": results_text,
        "export_data": export_data,
        "optimizer_result": {
            "success": bool(r_0.success),
            "message": str(r_0.message) if hasattr(r_0, 'message') else "",
            "nfev": int(r_0.nfev) if hasattr(r_0, 'nfev') else 0
        }
    }
    
    log_progress("Procesamiento completado exitosamente")
    
    try:
        log(results_text)
    except Exception:
        pass
    results["log_output"] = "\n".join(log_lines)
    return _sanitize_for_json(results)
