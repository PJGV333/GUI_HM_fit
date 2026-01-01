# errors.py
import numpy as _onp
from .np_backend import xp as np, jit, jacrev, vmap, lax
from ..solvers.lm_conc import pinv_cs  # complex-step-safe

# ============================================================================
# DEEP DIAGNOSTICS: Análisis de identificabilidad y correlación
# ============================================================================

def compute_condition_from_psd_eigs(eigs, threshold=1e-16):
    """
    Calcula kappa = max(eigs)/min(eigs) para una matriz PSD.
    Si el mínimo es <= threshold * max, se considera singular (inf).
    """
    eigs = _onp.asarray(eigs)
    if eigs.size == 0:
        return 1.0
    wmax = _onp.max(eigs)
    if wmax <= 0:
        return float('inf')
    wmin = _onp.min(eigs)
    if wmin <= threshold * wmax:
        return float('inf')
    return float(wmax / wmin)

def classify_stability(cond, maxcorr, rank_eff, p_free,
                      cond_warn=1e4, cond_crit=1e8,
                      corr_warn=0.95, corr_crit=0.99):
    """
    Classifies stability using WORST case of cond, rank, and max correlation.
    """
    is_singular = (p_free > 0 and rank_eff < p_free)
    not_identifiable = (not _onp.isfinite(cond)) or (not _onp.isfinite(maxcorr))
    
    if is_singular:
        return "critical", "Ill-conditioned (Singular)"
    if not_identifiable or cond > cond_crit or maxcorr >= corr_crit:
        return "critical", "Ill-conditioned"
    if cond >= cond_warn or maxcorr >= corr_warn:
        return "warn", "Warning"
    return "excellent", "Stable"

def build_identifiability_report(JJT, Cov_free, param_names=None, thresholds=(1e4, 1e8)):
    """
    In-depth diagnostic: condition number, rank, and correlations.
    Returns a dict with metadata and text representations (English).
    """
    # 1. Eigenvalues
    try:
        eigs, vecs = _onp.linalg.eigh(JJT)
        eigs = _onp.sort(eigs)
    except Exception as e:
        return {"status": "error", "diag_summary": f"Error calculating eigenvalues: {e}"}

    cond = compute_condition_from_psd_eigs(eigs)
    
    # Effective rank calculation (eigs > 1e-16 * max_eig)
    wmax = _onp.max(eigs) if eigs.size > 0 else 0.0
    rank_eff = _onp.sum(eigs > 1e-16 * wmax) if wmax > 0 else 0
    p_free = eigs.size

    # 2. Correlation Matrix analysis
    # Use clip to avoid sqrt of negative or extremely small values
    d = _onp.sqrt(_onp.clip(_onp.diag(Cov_free), 1e-30, None))
    Corr = Cov_free / _onp.outer(d, d)
    Corr = _onp.clip(Corr, -1.0, 1.0)
    
    # Calculate max absolute off-diagonal correlation
    maxcorr = 0.0
    rows, cols = Corr.shape
    if rows > 1:
        off_diag = _onp.abs(Corr - _onp.eye(rows))
        # Use nanmax to handle any residual NaNs safely
        maxcorr = float(_onp.nanmax(off_diag))

    # 3. Final Combined Status
    warn_thr, crit_thr = thresholds
    status, label = classify_stability(cond, maxcorr, rank_eff, p_free, cond_warn=warn_thr, cond_crit=crit_thr)
    
    icon = "✅" if status == "excellent" else ("⚠️" if status == "warn" else "❌")
    
    # Reason flags for UI
    reasons = []
    if rank_eff < p_free: reasons.append("singular")
    if maxcorr >= 0.95: reasons.append("high correlation")
    if cond >= warn_thr: reasons.append("poor conditioning")

    # stability_indicator for the UI
    indicator = {
        "status": status,
        "label": label,
        "icon": icon,
        "cond": float(cond),
        "max_abs_corr": maxcorr,
        "rank_eff": int(rank_eff),
        "p_free": int(p_free),
        "reasons": reasons
    }

    # 4. Short summary
    summary = f"Stability: {icon} {label} (cond={cond:.2e}, max|r|={maxcorr:.3f})"

    # 5. Detailed report (full_text)
    lines = []
    lines.append("--- STABILITY DIAGNOSTICS ---")
    lines.append(f"Eigenvalues (min -> max):\n{eigs}")
    lines.append(f"Effective rank: {rank_eff} / {p_free}")
    lines.append(f"Condition number (κ): {cond:.2e}")
    lines.append(f"Max abs correlation: {maxcorr:.4f}")
    
    # Weakest mode
    if param_names and len(param_names) == eigs.size:
        weakest = vecs[:, 0]
        comps = [f"{param_names[i]}: {weakest[i]:.2f}" for i in range(len(weakest))]
        lines.append("\nWeakest mode (direction of highest uncertainty):")
        lines.append(", ".join(comps))

    # 6. Correlation Matrix
    lines.append("\n--- CORRELATION MATRIX ---")
    # Header with param names if possible, else indices
    col_labels = param_names if (param_names and len(param_names) == cols) else [str(i) for i in range(cols)]
    header = "      " + "  ".join([f"{str(l)[:5]:>5}" for l in col_labels])
    lines.append(header)
    
    for i in range(rows):
        row_label = col_labels[i]
        row_str = f"{str(row_label)[:4]:>4} |"
        for j in range(cols):
            val = Corr[i, j]
            mark = "*" if abs(val) > 0.95 and i != j else " "
            row_str += f"{val:6.2f}{mark}"
        lines.append(row_str)

    # High correlation pairs
    high_pairs = []
    lines.append("\nHighly correlated pairs (|r| > 0.95):")
    found = False
    for i in range(rows):
        for j in range(i+1, cols):
            if abs(Corr[i, j]) > 0.95:
                name_i = param_names[i] if (param_names and len(param_names) > i) else f"P{i}"
                name_j = param_names[j] if (param_names and len(param_names) > j) else f"P{j}"
                lines.append(f"  {name_i} <--> {name_j} : {Corr[i, j]:.4f}")
                high_pairs.append((name_i, name_j, float(Corr[i, j])))
                found = True
    
    if not found:
        lines.append("  None detected.")
    else:
        # Actionable suggestion only when high pairs exist
        lines.append("\nSuggestion: Check for components without signals (e.g., non-absorbing/non-observable species), fix one constant, or simplify the model.")
    
    lines.append("-----------------------------\n")

    return {
        "cond_jjt": float(cond),
        "eigs_jjt": eigs.tolist(),
        "status": status,
        "stability_indicator": indicator,
        "diag_summary": summary,
        "diag_full": "\n".join(lines),
        "high_corr_pairs": high_pairs,
        "max_abs_corr": maxcorr,
        "rank_eff": int(rank_eff),
        "p_free": int(p_free)
    }

def analyze_identifiability(Hess, param_names=None):
    """Legacy wrapper for console printing if debug is on."""
    # We'll see if we still need this. The processor should handle it now.
    report = build_identifiability_report(Hess, _onp.linalg.pinv(Hess), param_names)
    print(report["diag_full"])

def print_correlation_matrix(Cov, param_names=None):
    """Legacy wrapper."""
    report = build_identifiability_report(_onp.linalg.pinv(Cov), Cov, param_names)
    print(report["diag_full"])


# ============================================================================

def jacobian_cs(fun, x, delta=1e-20):
    x = np.asarray(x)
    rows = []
    for i in range(len(x)):
        h = np.zeros_like(x, dtype=np.complex128)
        h[i] = 1j * delta
        rows.append(np.imag(fun(x + h)) / delta)
    return np.asarray(rows)  # (p, m)

def sigma2_reduced(r, p, dof=None):
    """
    Varianza reducida: s^2 = SSE / (m - p)
    """
    r = np.asarray(r).ravel()
    m = r.size
    dof = (m - p) if dof is None else dof
    dof = max(int(dof), 1)
    return np.sum(r*r) / dof

def cov_params(J, s2, weights=None, rcond=1e-12):
    """
    Cov(θ) ≈ s^2 (J W^{1/2})(J W^{1/2})^+  con J de forma (p, m).
    weights: None, vector (m,) o matriz diagonal (m,m) con 1/σ_i^2.
    """
    if weights is None:
        JW = J
    else:
        if weights.ndim == 1:
            JW = J * np.sqrt(weights)[None, :]
        else:  # matriz (m,m)
            JW = J @ np.sqrt(weights)
    G = JW @ JW.T                     # (p,p)
    return s2 * pinv_cs(G, rcond=rcond)

def se_from_cov(C):
    return np.sqrt(np.clip(np.diag(C), 0, np.inf))

def percent_error_log10K(k_log10, se_log10):
    """
    %error en K (no en log10 K) por delta method:
    Var(K) = (ln10 * K)^2 Var(log10 K).
    Devuelve (perc_K, se_K, K).
    """
    K = 10.0**np.asarray(k_log10)
    se_K = np.log(10.0) * K * np.asarray(se_log10)
    perc = 100.0 * se_K / np.maximum(np.abs(K), 1e-300)
    return perc, se_K, K


# --- NUEVO: util para pasar de SE_log10K a métricas en K ---

def percent_metrics_from_log10K(log10K, SE_log10K):
    """
    Devuelve SE(K), %Error lineal (delta method) y % asimétrico log-normal.
    Entradas y salidas vectoriales.
    """
    log10K = _onp.asarray(log10K, dtype=float)
    SE     = _onp.asarray(SE_log10K, dtype=float)

    # K y SE(K) por delta method (linealización)
    K      = _onp.power(10.0, log10K)
    rel_SE = _onp.log(10.0) * SE              # SE(K)/K
    SE_K   = K * rel_SE
    perc_linear = 100.0 * rel_SE              # %Error (K) lineal

    # Banda asimétrica 1σ de una lognormal exacta
    m      = _onp.power(10.0, SE)             # factor multiplicativo 1σ
    perc_hi = 100.0 * (m - 1.0)               # % por arriba
    perc_lo = 100.0 * (1.0 - 1.0/m)           # % por abajo

    # (opcional) % relativo en escala log10 (sólo para diagnóstico)
    with _onp.errstate(divide='ignore', invalid='ignore'):
        perc_log10 = 100.0 * SE / _onp.maximum(_onp.abs(log10K), 1e-12)

    return {
        "K": K, "SE_K": SE_K,
        "perc_linear": perc_linear,
        "perc_hi": perc_hi, "perc_lo": perc_lo,
        "perc_log10K": perc_log10
    }

def basic_metrics(y_obs, y_cal, r):
    sse = np.sum(r*r)
    mae = np.mean(np.abs(r))
    rms = np.sqrt(np.mean(r*r))
    lof = 100.0 * np.sqrt(sse / np.maximum(np.sum(y_obs*y_obs), 1e-300))
    return {"SSE": sse, "RMS": rms, "MAE": mae, "LoF%": lof}


def param_errors_cs(residuals, k):
    r = residuals(k).ravel()
    J = jacobian_cs(residuals, k)      # (p, m)
    m, p = r.size, len(k)
    s2 = (r @ r) / max(m - p, 1)       # varianza reducida
    Cov = s2 * pinv_cs(J @ J.T)        # (p, p)
    SE_log10K = np.sqrt(np.clip(np.diag(Cov), 0, np.inf))
    K_num = 10.0**np.asarray(k)
    SE_K  = np.log(10.0) * K_num * SE_log10K
    percK = 100.0 * np.log(10.0) * SE_log10K  # % en K (delta)
    return SE_log10K, SE_K, percK, s2, Cov



def sensitivities_wrt_logK(c_spec, modelo, param_idx=None, rcond=1e-12):
    """
    Devuelve dCspec/d(log10 K) por diferenciación implícita.
    Acepta 'modelo' en cualquiera de las dos orientaciones:
      - (nspec, n_comp)  ó
      - (n_comp, nspec)
    """
    c = np.asarray(c_spec, dtype=float)           # (nspec,)
    M_in = np.asarray(modelo, dtype=float)

    nspec = c.size

    # --- Normalizar orientación de M ---
    # Ms: (nspec, n_comp), MT: (n_comp, nspec)
    if M_in.shape[0] == nspec:
        Ms = M_in
        MT = Ms.T
    elif M_in.shape[1] == nspec:
        Ms = M_in.T
        MT = Ms.T
    else:
        raise ValueError(
            f"modelo incompatible con c_spec: c:{c.shape}, modelo:{M_in.shape} "
            "(se espera (nspec, n_comp) o (n_comp, nspec))"
        )

    # --- Jacobiano respecto a u = ln c ---
    # Ju = M^T diag(c) M   -> (n_comp, n_comp)
    Ju = MT @ (c[:, None] * Ms)

    # RHS = M^T diag(c)    -> (n_comp, nspec)
    RHS = MT * c

    # Resolver Ju * (du/dlnK) = RHS
    try:
        du_dlnK = np.linalg.solve(Ju, RHS)          # (n_comp, nspec)
    except np.linalg.LinAlgError:
        du_dlnK = np.linalg.pinv(Ju, rcond=rcond) @ RHS

    dCspec_dlnK = (c[:, None]) * (np.eye(nspec) - Ms @ du_dlnK)

    # Pasar a log10: d/d(log10 K) = ln(10) * d/d(ln K)
    dCspec_dlog10K = np.log(10.0) * dCspec_dlnK

    # Seleccionar columnas si sólo un subconjunto de especies está parametrizado
    if param_idx is not None:
        dCspec_dlog10K = dCspec_dlog10K[:, param_idx]  # (nspec, p)

    return dCspec_dlog10K


# --- errors.py (añadir al final) ---
from .np_backend import xp as np, USE_JAX, jit
# Usaremos tu sensitivities_wrt_log10K ya existente:
_sens = sensitivities_wrt_logK

def _pinv_backend(A, rcond=1e-10):
    # pinv compatible con JAX/NumPy
    return np.linalg.pinv(A, rcond=rcond)

def _projector(C, rcond=1e-10):
    return C @ _pinv_backend(C, rcond=rcond)

def _stack_rows(lst):
    # stack que funcione en ambos backends
    return np.stack(lst, axis=0)

def _as_onp(x):
    try:
        import jax.numpy as jnp
        if isinstance(x, jnp.ndarray):
            return _onp.array(x)
    except Exception:
        pass
    return _onp.array(x)

def _normalize_modelo(modelo, nspec):
    Min = _as_onp(modelo)
    if Min.shape[0] == nspec:
        Ms = Min
        n_comp = Ms.shape[1]
    elif Min.shape[1] == nspec:
        Ms = Min.T
        n_comp = Ms.shape[1]
    else:
        raise ValueError(f"modelo incompatible con nspec={nspec}, shape={Min.shape}")
    return Ms, n_comp

def _jac_varpro(C, A, dC_all, use_projector=True, rcond=1e-10):
    """
    C      : (m × s)
    A      : (s × nw)
    dC_all : (m × s × p)
    Devuelve J (p × m*nw) con la forma proyectada (I-P) dC A
    """
    m, s = C.shape
    nw    = A.shape[1]
    p     = dC_all.shape[2]

    if use_projector:
        P = _projector(C, rcond=rcond)
        IminusP = np.eye(m) - P

    Js = []
    for q in range(p):
        dCq = dC_all[:, :, q]     # (m × s)
        Vq  = dCq @ A             # (m × nw)
        if use_projector:
            Vq = IminusP @ Vq
        Js.append(Vq.reshape(m * nw))
    J = np.stack(Js, axis=0)      # (p × m*nw)
    return J

# --- al final del archivo, después de definir la función ---
try:
    if USE_JAX:
        # jit de JAX con argumento estático
        from jax import jit as _jit
        _jac_varpro = _jit(_jac_varpro, static_argnames=("use_projector",))
except Exception:
    # si no hay JAX o falla el wrap, seguimos sin jit (modo NumPy)
    pass


def _build_dC_all(Co, Ms, nas, param_idx):
    """
    Co      : (m × nspec) todas las especies
    Ms      : (nspec × n_comp)
    nas     : lista de NO absorbentes
    param_idx: columnas de especies (en nspec) que están parametrizadas
    """
    nspec = Co.shape[1]
    abs_idx = [j for j in range(nspec) if j not in nas]  # especies absorbentes
    rows = []
    for i in range(Co.shape[0]):
        # dCspec/dlog10K en TODAS las especies (nspec × p)
        dC_dlog = _sens(Co[i], Ms, param_idx=param_idx)
        # Sólo absorbentes:
        rows.append(dC_dlog[abs_idx, :])   # (n_abs × p)
    return _stack_rows(rows)               # (m × n_abs × p)

def _percent_errors_from_cov(k, Cov_log10K):
    SE_log10K = _onp.sqrt(_onp.clip(_onp.diag(_as_onp(Cov_log10K)), 0.0, _onp.inf))
    # use local percent_error_log10K
    percK, SE_K, _ = percent_error_log10K(_as_onp(k), SE_log10K)
    return _onp.array(percK), _onp.array(SE_K), _onp.array(SE_log10K)

def pinv_psd_eigh(A, xp=_onp, rcond=1e-10, ridge=0.0):
    """
    Pseudo-inversa para matrices simetricas/hermiticas PSD via eigh con truncamiento.
    """
    A = xp.asarray(A)
    A = 0.5 * (A + A.T.conj())
    if ridge is not None and float(ridge) > 0.0:
        A = A + float(ridge) * xp.eye(A.shape[0], dtype=A.dtype)
    w, V = xp.linalg.eigh(A)
    wmax = xp.max(w)
    thr = rcond * wmax
    w_inv = xp.where(w > thr, 1.0 / w, 0.0)
    return (V * w_inv) @ V.T.conj()

def compute_errors_spectro_varpro(k, res, Y, modelo, nas, rcond=1e-10, use_projector=True, param_names=None):
    """
    Cálculo robusto (variable projection) para espectroscopía.
    k       : (p,)
    res     : solver con .concentraciones(k) -> (C_abs, Co_all)
    Y       : (m × nw)  datos (filas: puntos de titulación; cols: longitudes de onda)
    modelo  : M (nspec×n_comp) o (n_comp×nspec)
    nas     : lista[int] índices NO absorbentes
    """
    C, Co = res.concentraciones(k)    # C: (m × n_abs), Co: (m × nspec)
    nspec = Co.shape[1]
    Ms, n_comp = _normalize_modelo(modelo, nspec)

    # Mapear parámetros a columnas de especies (por defecto, complejos):
    p = len(k)
    param_idx = list(range(n_comp, nspec))
    if len(param_idx) != p:
        if p <= nspec:
            param_idx = list(range(nspec - p, nspec))
        else:
            raise ValueError(f"p={p} > nspec={nspec}")

    # Ajuste lineal y residuo
    A   = _pinv_backend(C, rcond=rcond) @ Y.T     # (n_abs × nw)
    R   = (C @ A - Y.T)                           # (m × nw)
    r   = R.reshape(-1)
    dof = max(r.size - p, 1)
    s2  = float((r @ r) / dof)

    # Jacobiano proyectado
    dC_all = _build_dC_all(Co, Ms, nas, param_idx)          # (m × n_abs × p)
    J = _jac_varpro(C, A, dC_all, use_projector=use_projector, rcond=rcond)

    # Identifiability
    JJT = J @ J.T
    Cov_log10K = s2 * _pinv_backend(JJT, rcond=rcond)
    
    # Filter param_names if provided
    param_names_free = None
    if param_names is not None:
        param_names_free = list(param_names) # Assuming all are free in spectro currently or handled by calling code
    
    diag = build_identifiability_report(JJT, _as_onp(Cov_log10K), param_names=param_names_free)
    
    SE_log10K = _onp.sqrt(_onp.clip(_onp.diag(_as_onp(Cov_log10K)), 0.0, _onp.inf))
    pm = percent_metrics_from_log10K(_as_onp(k), SE_log10K)

    percK      = pm["perc_linear"]      # %Error (K) por delta method (simétrico)
    SE_K       = pm["SE_K"]         
    perc_hi    = pm["perc_hi"]          # % asimétrico (arriba)
    perc_lo    = pm["perc_lo"]      # % asimétrico (abajo)

    RMS = float(np.sqrt(np.mean(R * R)))
    return {
        "percK": percK, "SE_K": SE_K, "SE_log10K": SE_log10K,
        "Cov_log10K": _as_onp(Cov_log10K), "RMS": RMS, "s2": s2,
        "A": _as_onp(A), "J": _as_onp(J), "yfit": _as_onp((C @ A).T),
        "r": _as_onp(r), "dof": int(dof), "nobs": int(r.size),
        "stability_diag": diag
    }


def compute_errors_nmr_varpro(
    k,
    res,
    dq,
    H,
    modelo,
    nas,
    rcond=1e-10,
    use_projector=True,
    mask=None,
    fixed_mask=None,
    ridge=1e-8,
    rcond_cov=None, ridge_cov=0.0, debug=False,
    param_names=None,
):
    """
    NMR version with missing data support.
    If 'mask' is None, uses all finite dq values.
    If 'mask' is bool (m x nP), ignores unobserved rows per column.
    
    Now supports 'fixed_mask' to correctly account for degrees of freedom.
    """
    C, Co = res.concentraciones(k)    # C: (m × n_abs), Co: (m × nspec)
    nspec = Co.shape[1]
    Ms, n_comp = _normalize_modelo(modelo, nspec)

    k = _as_onp(k)
    p_total = len(k)
    
    if fixed_mask is None:
        fixed_mask = _onp.zeros(p_total, dtype=bool)
    else:
        fixed_mask = _onp.asarray(fixed_mask, dtype=bool)
        
    free_idx = _onp.where(~fixed_mask)[0]
    p_free = free_idx.size

    if p_free == 0:
        import warnings
        warnings.warn("All parameters are fixed (p_free=0). SE/log10K and %error will be zero by definition.")

    param_idx = list(range(n_comp, nspec))
    if len(param_idx) != p_total:
        if p_total <= nspec:
            param_idx = list(range(nspec - p_total, nspec))
        else:
            raise ValueError(f"p={p_total} > nspec={nspec}")

    # Xi calculation is now handled per-signal or per-block to avoid global inf/nan
    # when some parent concentrations are zero (e.g. at the start of titration).
    H = _as_onp(H)
    dq = _as_onp(dq)
    m, n_abs = C.shape

    if mask is None:
        # usar todo como observado; el filtrado real lo hace Hj_all>1e-12 y finitud
        mask = _onp.isfinite(dq)

    # ---------- CAMINO CON MÁSCARA (ignora huecos por columna) ----------
    mask = _onp.asarray(mask, dtype=bool)
    nP = dq.shape[1]

    coef = _onp.zeros((n_abs, nP), dtype=float)
    R_blocks = []   # lista de residuales por columna, sólo filas observadas
    J_cols   = []   # lista de J (p × m_j) por columna

    # Precompute dC_all_abs (m × n_abs × p)
    dC_all_abs = _build_dC_all(Co, Ms, nas, param_idx)

    for j in range(nP):
        mj_obs = mask[:, j]
        # Signal-specific parent concentration
        Hj_all = H[:, j] if H.ndim == 2 else H
        
        # Point is valid for signal j if:
        # 1. It is observed (mask)
        # 2. Parent concentration > threshold
        mj = mj_obs & (_onp.abs(Hj_all) > 1e-12)
        
        if not mj.any():
            continue

        # Calculate Xi_m LOCALLY to avoid global inf
        Xi_m = C[mj, :] / Hj_all[mj, None]      # (m_j × n_abs)
        y    = dq[mj, j]                        # (m_j,)

        # Finiteness guard
        if not _onp.isfinite(Xi_m).all() or not _onp.isfinite(y).all():
            import warnings
            warnings.warn(f"Signal {j}: Non-finite values in basis or data. Skipping.")
            continue

        # coeficiente de la señal j usando SOLO filas observadas
        try:
            delta, _, _, _ = _onp.linalg.lstsq(Xi_m, y, rcond=rcond)
        except _onp.linalg.LinAlgError:
            if ridge is not None and float(ridge) > 0.0:
                XtX = Xi_m.T @ Xi_m
                delta = _onp.linalg.solve(XtX + float(ridge) * _onp.eye(XtX.shape[0]), Xi_m.T @ y)
            else:
                delta = _onp.linalg.pinv(Xi_m, rcond=rcond) @ y

        Aj = delta.reshape(-1, 1)
        coef[:, j] = Aj[:, 0]

        # residuales de la señal j
        yfit = (Xi_m @ Aj).ravel()              # (m_j,)
        R_blocks.append(yfit - y)

        # Jacobiano para la señal j: dXi en filas observadas
        dXi_m = (dC_all_abs[mj, :, :] / Hj_all[mj].reshape(-1, 1, 1))   # (m_j × n_abs × p)
        Jj = _jac_varpro(Xi_m, Aj, dXi_m, use_projector=use_projector, rcond=rcond)  # (p × m_j)
        J_cols.append(Jj)

    if not J_cols:
        raise ValueError("No hay datos observados válidos para calcular errores (padre concentracion ~0 o máscara vacía).")

    r  = _onp.concatenate([_as_onp(rb) for rb in R_blocks], axis=0)   # (M_eff,)
    J_full  = _onp.concatenate([_as_onp(Jc) for Jc in J_cols], axis=1) # (p_total × M_eff)
    
    nobs = r.size
    dof = nobs - p_free
    dof_capped = max(dof, 1)
    s2  = float((r @ r) / dof_capped)

    if p_free == 0:
        SE_log10K_full = _onp.zeros(p_total)
        Cov_free = _onp.zeros((0, 0))
        stability_diag = {
            "cond_jjt": 1.0, 
            "status": "fixed", 
            "stability_indicator": {
                "status": "excellent",
                "label": "Fixed",
                "icon": "✅",
                "cond": 1.0
            },
            "diag_summary": "Stability: ✅ Fixed (all parameters are fixed)", 
            "diag_full": ""
        }
    else:
        if rcond_cov is None:
            rcond_cov = rcond
        J = J_full[free_idx, :]
        JJT = J @ J.T
        JJT = 0.5 * (JJT + JJT.T.conj())

        invJJT = pinv_psd_eigh(JJT, xp=_onp, rcond=rcond_cov, ridge=ridge_cov)
        Cov_free = s2 * invJJT

        # --- Stability Diagnostics ---
        param_names_free = None
        if param_names is not None:
            param_names_free = [param_names[i] for i in free_idx]
            
        stability_diag = build_identifiability_report(JJT, Cov_free, param_names=param_names_free)

        if debug:
            print(stability_diag["diag_full"])

        SE_log10K_free = _onp.sqrt(_onp.clip(_onp.diag(_as_onp(Cov_free)), 0.0, _onp.inf))
        
        SE_log10K_full = _onp.zeros(p_total)
        SE_log10K_full[free_idx] = SE_log10K_free
    
    pm = percent_metrics_from_log10K(k, SE_log10K_full)

    return {
        "percK": pm["perc_linear"], "SE_K": pm["SE_K"], "SE_log10K": SE_log10K_full,
        "Cov_log10K_free": _as_onp(Cov_free), "rms": float(_onp.sqrt(_onp.mean(r**2))), "covfit": s2,
        "coef": _as_onp(coef), "xi": None, "J": _as_onp(J_full),
        "dof": dof, "nobs": nobs, "residuals_vec": _as_onp(r), "r": _as_onp(r),
        "stability_diag": stability_diag
    }


# === Bootstrap utilities ===

def wild_multipliers(rng, n, kind="rademacher"):
    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0.")
    kind = str(kind or "rademacher").strip().lower()
    if kind.startswith("mam"):
        sqrt5 = _onp.sqrt(5.0)
        a = (1.0 - sqrt5) / 2.0
        b = (1.0 + sqrt5) / 2.0
        p = b / sqrt5
        u = rng.random(n)
        return _onp.where(u < p, a, b)
    # default: Rademacher (+/-1)
    return rng.choice(_onp.array([-1.0, 1.0]), size=n)


def _wild_weights(rng, shape, wild="rademacher"):
    size = int(_onp.prod(shape))
    if size == 0:
        return _onp.empty(shape, dtype=float)
    v = wild_multipliers(rng, size, kind=wild)
    return _onp.asarray(v, dtype=float).reshape(shape)


class BootstrapCancelled(RuntimeError):
    pass


def bootstrap_full_refit(
    theta_hat,
    make_data_star_fn,
    refit_fn,
    B,
    seed=None,
    wild="rademacher",
    max_iter=30,
    tol=1e-8,
    fail_policy="skip",
    progress_cb=None,
    cancel_cb=None,
):
    """
    Full-refit bootstrap: regenerate dataset via wild residuals and refit each replicate.
    Returns dict with samples (n_success x p), percentiles, and corr (or None if n_success < 2).
    """
    theta_hat = _onp.asarray(theta_hat, dtype=float).ravel()
    B = int(B)
    if B <= 0:
        raise ValueError("B must be > 0.")
    if fail_policy not in ("skip", "stop"):
        raise ValueError("fail_policy must be 'skip' or 'stop'.")

    rng = _onp.random.default_rng(seed)
    samples_list = []
    n_fail = 0

    for b in range(B):
        if cancel_cb is not None and cancel_cb():
            raise BootstrapCancelled("Bootstrap cancelled.")
        data_star = make_data_star_fn(rng, wild=wild)
        theta0 = theta_hat.copy()
        theta_star, ok, info = refit_fn(data_star, theta0, max_iter=max_iter, tol=tol)
        if ok:
            samples_list.append(_onp.asarray(theta_star, dtype=float).ravel())
        else:
            n_fail += 1
            if fail_policy == "stop":
                raise RuntimeError(f"Bootstrap full-refit failed at replicate {b + 1}: {info}")
        if progress_cb is not None:
            progress_cb(
                {
                    "current": b + 1,
                    "total": B,
                    "n_success": len(samples_list),
                    "n_fail": n_fail,
                }
            )

    samples = (
        _onp.vstack(samples_list)
        if samples_list
        else _onp.zeros((0, theta_hat.size), dtype=float)
    )
    n_success = samples.shape[0]

    if n_success > 0:
        median = _onp.median(samples, axis=0)
        p2_5 = _onp.percentile(samples, 2.5, axis=0)
        p97_5 = _onp.percentile(samples, 97.5, axis=0)
        p16 = _onp.percentile(samples, 16.0, axis=0)
        p84 = _onp.percentile(samples, 84.0, axis=0)
    else:
        median = p2_5 = p97_5 = p16 = p84 = None

    corr = None
    if n_success >= 2:
        corr = _onp.corrcoef(samples, rowvar=False)
        corr = _onp.nan_to_num(corr, nan=0.0)
        if corr.ndim == 2:
            for i in range(corr.shape[0]):
                corr[i, i] = 1.0

    return {
        "samples": samples,
        "n_success": int(n_success),
        "n_fail": int(n_fail),
        "median": median,
        "p2_5": p2_5,
        "p97_5": p97_5,
        "p16": p16,
        "p84": p84,
        "corr": corr,
    }


def bootstrap_linearized_wild(
    k_hat,
    J,
    r,
    B,
    seed=None,
    wild="rademacher",
    rcond=1e-10,
    ridge=0.0,
):
    """
    Fast bootstrap via linearized covariance and wild residuals.
    Returns dict with samples (B x p).
    """
    k_hat = _onp.asarray(k_hat, dtype=float).ravel()
    J = _onp.asarray(J, dtype=float)
    r = _onp.asarray(r, dtype=float).ravel()
    B = int(B)
    if B <= 0:
        raise ValueError("B must be > 0.")
    if J.shape[1] != r.size:
        raise ValueError(f"J and r size mismatch: J has {J.shape[1]} cols, r has {r.size} elems.")

    rng = _onp.random.default_rng(seed)
    G = J @ J.T
    G_inv = pinv_psd_eigh(G, xp=_onp, rcond=rcond, ridge=ridge)

    samples = _onp.zeros((B, k_hat.size), dtype=float)
    for b in range(B):
        v = _wild_weights(rng, r.size, wild=wild)
        eps = r * v
        g = J @ eps
        dk = G_inv @ g
        samples[b, :] = k_hat + dk

    return {
        "samples": samples,
    }


def bootstrap_one_step_spectro(
    k_hat,
    res,
    Y,
    modelo,
    nas,
    B,
    seed=None,
    wild="rademacher",
    lam=1e-3,
    rcond=1e-10,
    use_projector=True,
):
    """
    One-step LM bootstrap (spectroscopy): dataset perturbation + recompute J,r at k_hat + 1 LM step.
    """
    k_hat = _onp.asarray(k_hat, dtype=float).ravel()
    B = int(B)
    if B <= 0:
        raise ValueError("B must be > 0.")

    metrics0 = compute_errors_spectro_varpro(
        k_hat, res, Y, modelo, nas, rcond=rcond, use_projector=use_projector
    )
    yfit = _onp.asarray(metrics0["yfit"], dtype=float)
    Y_obs = _onp.asarray(Y, dtype=float)
    if yfit.shape != Y_obs.shape:
        raise ValueError(f"yfit shape {yfit.shape} does not match Y {Y_obs.shape}.")

    R = yfit - Y_obs
    rng = _onp.random.default_rng(seed)
    samples = _onp.zeros((B, k_hat.size), dtype=float)

    for b in range(B):
        v = _wild_weights(rng, R.shape, wild=wild)
        Y_star = yfit - R * v

        metrics_star = compute_errors_spectro_varpro(
            k_hat, res, Y_star, modelo, nas, rcond=rcond, use_projector=use_projector
        )
        J = _onp.asarray(metrics_star["J"], dtype=float)
        r_star = _onp.asarray(metrics_star["r"], dtype=float).ravel()

        G = J @ J.T + float(lam) * _onp.eye(J.shape[0])
        try:
            dk = -_onp.linalg.solve(G, J @ r_star)
        except _onp.linalg.LinAlgError:
            dk = -_onp.linalg.pinv(G, rcond=rcond) @ (J @ r_star)
        samples[b, :] = k_hat + dk

    return {
        "samples": samples,
    }


def bootstrap_one_step_nmr(
    k_hat,
    res,
    dq,
    dq_fit,
    D_cols,
    modelo,
    nas,
    B,
    seed=None,
    wild="rademacher",
    lam=1e-3,
    rcond=1e-10,
    use_projector=True,
    mask=None,
    fixed_mask=None,
    rcond_cov=None,
    ridge=1e-8,
    ridge_cov=0.0,
):
    """
    One-step LM bootstrap (NMR): dataset perturbation + recompute J,r at k_hat + 1 LM step.
    """
    k_hat = _onp.asarray(k_hat, dtype=float).ravel()
    B = int(B)
    if B <= 0:
        raise ValueError("B must be > 0.")

    dq_obs = _onp.asarray(dq, dtype=float)
    dq_fit = _onp.asarray(dq_fit, dtype=float)
    if dq_obs.shape != dq_fit.shape:
        raise ValueError(f"dq_fit shape {dq_fit.shape} does not match dq {dq_obs.shape}.")

    if mask is None:
        mask = _onp.isfinite(dq_obs)
    mask = _onp.asarray(mask, dtype=bool)

    if fixed_mask is None:
        fixed_mask = _onp.zeros(k_hat.size, dtype=bool)
    else:
        fixed_mask = _onp.asarray(fixed_mask, dtype=bool)
    free_idx = _onp.where(~fixed_mask)[0]

    R = dq_fit - dq_obs
    rng = _onp.random.default_rng(seed)
    samples = _onp.zeros((B, k_hat.size), dtype=float)

    for b in range(B):
        v = _wild_weights(rng, R.shape, wild=wild)
        dq_star = dq_fit - R * v
        dq_star[~mask] = dq_obs[~mask]

        metrics_star = compute_errors_nmr_varpro(
            k_hat,
            res,
            dq_star,
            D_cols,
            modelo,
            nas,
            mask=mask,
            fixed_mask=fixed_mask,
            rcond=rcond,
            rcond_cov=rcond_cov if rcond_cov is not None else rcond,
            ridge=ridge,
            ridge_cov=ridge_cov,
            use_projector=use_projector,
        )

        if free_idx.size == 0:
            samples[b, :] = k_hat
            continue

        J_full = _onp.asarray(metrics_star["J"], dtype=float)
        r_star = _onp.asarray(metrics_star["r"], dtype=float).ravel()
        J = J_full[free_idx, :]

        G = J @ J.T + float(lam) * _onp.eye(J.shape[0])
        try:
            dk = -_onp.linalg.solve(G, J @ r_star)
        except _onp.linalg.LinAlgError:
            dk = -_onp.linalg.pinv(G, rcond=rcond) @ (J @ r_star)

        k1 = k_hat.copy()
        k1[free_idx] = k1[free_idx] + dk
        samples[b, :] = k1

    return {
        "samples": samples,
    }
