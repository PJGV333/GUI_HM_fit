# errors.py
import numpy as _onp
from .np_backend import xp as np, jit, jacrev, vmap, lax
from ..solvers.lm_conc import pinv_cs  # complex-step-safe

# ============================================================================
# DEEP DIAGNOSTICS: Análisis de identificabilidad y correlación
# ============================================================================

def analyze_identifiability(Hess, param_names=None):
    """
    Diagnóstico profundo de la matriz Hessiana (o J.T @ J).
    Imprime condición, autovalores y advertencias de singularidad.
    """
    print("\n--- DIAGNÓSTICO DE IDENTIFICABILIDAD (Hessiana aprox) ---")
    
    # 1. Análisis de Autovalores (Espectro)
    try:
        # eigh es para matrices simétricas (Hermíticas), más estable
        vals, vecs = _onp.linalg.eigh(Hess)
    except Exception as e:
        print(f"Error calculando autovalores: {e}")
        return
        
    vals = _onp.sort(vals)  # Menor a mayor
    print(f"Autovalores (min -> max):\n{vals}")
    
    # 2. Número de Condición
    # Evitar división por cero si hay autovalores negativos/cero numéricos
    max_eig = _onp.max(_onp.abs(vals))
    min_pos_vals = vals[vals > 1e-15]
    min_eig = _onp.min(_onp.abs(min_pos_vals)) if min_pos_vals.size > 0 else 1e-30
    cond_num = max_eig / min_eig
    
    print(f"Condición (kappa): {cond_num:.2e}")
    
    if cond_num > 1e12:
        print("CRÍTICO: La matriz es numéricamente singular. El sistema está mal condicionado.")
    elif cond_num > 1e8:
        print("ADVERTENCIA: Condicionamiento pobre. Los errores serán grandes.")
        
    # 3. Análisis de vectores propios para el modo más débil
    # El autovector asociado al autovalor más pequeño apunta en la dirección de mayor incertidumbre
    if param_names and len(param_names) == len(vals):
        weakest_mode = vecs[:, 0]  # Asociado al menor autovalor
        print("Modo más débil (combinación lineal de params con mayor error):")
        comps = [f"{param_names[i]}: {weakest_mode[i]:.2f}" for i in range(len(weakest_mode))]
        print(", ".join(comps))
        
    print("---------------------------------------------------------\n")


def print_correlation_matrix(Cov, param_names=None):
    """
    Calcula e imprime la matriz de correlación a partir de la covarianza.
    Ayuda a ver si dos parámetros están 'peleando' (corr -> 1 o -1).
    """
    print("\n--- MATRIZ DE CORRELACIÓN ---")
    d = _onp.sqrt(_onp.diag(Cov))
    d[d == 0] = 1e-10  # Evitar div/0
    outer_d = _onp.outer(d, d)
    Corr = Cov / outer_d
    
    # Limitar rango para visualización limpia
    Corr = _onp.clip(Corr, -1.0, 1.0)
    
    rows, cols = Corr.shape
    print("      " + "  ".join([f"{i:5d}" for i in range(cols)]))
    for i in range(rows):
        row_str = f"{i:4d} |"
        for j in range(cols):
            val = Corr[i, j]
            # Resaltar correlaciones altas
            mark = "*" if abs(val) > 0.95 and i != j else " "
            row_str += f"{val:6.2f}{mark}"
        print(row_str)
    
    if param_names:
        # Listar pares con correlación peligrosa
        print("\nPares altamente correlacionados (|r| > 0.95):")
        found = False
        for i in range(rows):
            for j in range(i+1, cols):
                if abs(Corr[i, j]) > 0.95:
                    print(f"  {param_names[i]} <--> {param_names[j]} : {Corr[i, j]:.4f}")
                    found = True
        if not found:
            print("  Ninguno detectado.")
    print("-----------------------------\n")

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
    if ridge and ridge > 0.0:
        A = A + ridge * xp.eye(A.shape[0], dtype=A.dtype)
    w, V = xp.linalg.eigh(A)
    wmax = xp.max(w)
    thr = rcond * wmax
    w_inv = xp.where(w > thr, 1.0 / w, 0.0)
    return (V * w_inv) @ V.T.conj()

def compute_errors_spectro_varpro(k, res, Y, modelo, nas, rcond=1e-10, use_projector=True):
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

    # Covarianza y errores
    Cov_log10K = s2 * _pinv_backend(J @ J.T, rcond=rcond)
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
        "A": _as_onp(A), "J": _as_onp(J), "yfit": _as_onp((C @ A).T)
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
    rcond_cov=None,
    ridge_cov=0.0,
    debug=False,
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
            if ridge and ridge > 0.0:
                XtX = Xi_m.T @ Xi_m
                delta = _onp.linalg.solve(XtX + ridge*_onp.eye(XtX.shape[0]), Xi_m.T @ y)
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
    else:
        if rcond_cov is None:
            rcond_cov = rcond
        J = J_full[free_idx, :]
        JJT = J @ J.T
        JJT = 0.5 * (JJT + JJT.T.conj())

        if debug:
            sv = _onp.linalg.svd(JJT, compute_uv=False)
            if sv.size:
                cond = float(sv[0] / sv[-1]) if sv[-1] != 0 else float("inf")
                rank_eff = int(_onp.sum(sv > sv[0] * rcond_cov))
            else:
                cond = float("inf")
                rank_eff = 0
            print(
                f"[NMR errors] dof={dof} p_free={p_free} cond(JJT)={cond:.3e} "
                f"rank_eff={rank_eff}/{p_free} rcond_cov={rcond_cov} ridge_cov={ridge_cov}"
            )

        # --- INICIO BLOQUE DIAGNÓSTICO (antes de inversión) ---
        if debug:
            analyze_identifiability(JJT, param_names=None)
        # --- FIN BLOQUE DIAGNÓSTICO ---

        invJJT = pinv_psd_eigh(JJT, xp=_onp, rcond=rcond_cov, ridge=ridge_cov)
        Cov_free = s2 * invJJT

        # --- INICIO BLOQUE DIAGNÓSTICO POST-COVARIANZA ---
        if debug:
            print_correlation_matrix(Cov_free, param_names=None)
        # --- FIN BLOQUE DIAGNÓSTICO ---

        SE_log10K_free = _onp.sqrt(_onp.clip(_onp.diag(_as_onp(Cov_free)), 0.0, _onp.inf))
        
        SE_log10K_full = _onp.zeros(p_total)
        SE_log10K_full[free_idx] = SE_log10K_free
    
    pm = percent_metrics_from_log10K(k, SE_log10K_full)

    return {
        "percK": pm["perc_linear"], "SE_K": pm["SE_K"], "SE_log10K": SE_log10K_full,
        "Cov_log10K_free": _as_onp(Cov_free), "rms": float(_onp.sqrt(_onp.mean(r**2))), "covfit": s2,
        "coef": _as_onp(coef), "xi": None, "J": _as_onp(J_full),
        "dof": dof, "nobs": nobs, "residuals_vec": _as_onp(r)
    }
