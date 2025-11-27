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
    
    if isinstance(y, (list, np.ndarray)) and len(y) > 0 and isinstance(y[0], (list, np.ndarray)):
        # Multiple series
        for yi in y:
            ax.plot(x, yi, mark)
    else:
        ax.plot(x, y, mark)
    
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
        graphs['eigenvalues'] = generate_figure_base64(
            range(0, nc), np.log10(s), "o", "log(EV)", "# de autovalores", "Eigenvalues"
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
        C = res.concentraciones(k)[0]
        A = solve_A_nnls_pgd(C, Y.T, ridge=0.0, max_iters=300)
        r = C @ A - Y.T
        rms = np.sqrt(np.mean(np.square(r)))
        log_progress(f"f(x): {float(rms):.6e}")
        log_progress(f"x: {[float(ki) for ki in k]}")
        return rms
    
    # Optimization
    log_progress(f"Optimizer: {optimizer}")
    log_progress(f"Bounds: {bounds}")
    
    k = initial_k
    
    if optimizer == "differential_evolution":
        # Check bounds
        for (min_val, max_val) in bounds:
            if np.isinf(min_val) or np.isinf(max_val):
                raise ValueError("Los límites no deben contener valores infinitos.")
        
        r_0 = differential_evolution(f_m, bounds, x0=k, strategy='best1bin',
                                     maxiter=1000, popsize=15, tol=0.01,
                                     mutation=(0.5, 1), recombination=0.7,
                                     init='latinhypercube')
    else:
        r_0 = optimize.minimize(f_m, k, method=optimizer, bounds=bounds)
    
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
        "optimizer_result": {
            "success": bool(r_0.success),
            "message": str(r_0.message) if hasattr(r_0, 'message') else "",
            "nfev": int(r_0.nfev) if hasattr(r_0, 'nfev') else 0
        }
    }
    
    log_progress("Procesamiento completado exitosamente")
    
    return results
