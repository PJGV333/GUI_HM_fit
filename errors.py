# errors.py
import numpy as _onp
from np_backend import xp as np, jit, jacrev, vmap, lax
from LM_conc_algoritm import pinv_cs  # complex-step-safe :contentReference[oaicite:4]{index=4}

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

    # dCspec/dlnK = diag(c) * (I + M * du/dlnK)  -> (nspec, nspec)
    dCspec_dlnK = (c[:, None]) * (np.eye(nspec) + Ms @ du_dlnK)

    # Pasar a log10: d/d(log10 K) = ln(10) * d/d(ln K)
    dCspec_dlog10K = np.log(10.0) * dCspec_dlnK

    # Seleccionar columnas si sólo un subconjunto de especies está parametrizado
    if param_idx is not None:
        dCspec_dlog10K = dCspec_dlog10K[:, param_idx]  # (nspec, p)

    return dCspec_dlog10K


# --- errors.py (añadir al final) ---
from np_backend import xp as np, USE_JAX, jit
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
    from errors import percent_error_log10K  # tu función existente
    percK, SE_K, _ = percent_error_log10K(_as_onp(k), SE_log10K)
    return _onp.array(percK), _onp.array(SE_K), _onp.array(SE_log10K)

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

def compute_errors_nmr_varpro(k, res, dq, H, modelo, nas, rcond=1e-10, use_projector=True, mask=None):
    """
    Versión NMR con soporte de datos faltantes.
    Si 'mask' es None, usa el camino denso (compatibilidad).
    Si 'mask' es bool (m × nP), ignora filas NO observadas por columna (estilo HypNMR).
    """
    C, Co = res.concentraciones(k)    # C: (m × n_abs), Co: (m × nspec)
    nspec = Co.shape[1]
    Ms, n_comp = _normalize_modelo(modelo, nspec)

    p = len(k)
    param_idx = list(range(n_comp, nspec))
    if len(param_idx) != p:
        if p <= nspec:
            param_idx = list(range(nspec - p, nspec))
        else:
            raise ValueError(f"p={p} > nspec={nspec}")

    Xi = C / H[:, None]                      # (m × n_abs)
    m, n_abs = Xi.shape
    dq = _as_onp(dq)

    # ---------- CAMINO SIN MÁSCARA (compatibilidad) ----------
    if mask is None:
        coef = _pinv_backend(Xi, rcond=rcond) @ dq  # (n_abs × nP)
        Yfit = (coef.T @ Xi.T).T                    # (m × nP)
        R    = (Yfit - dq)                          # (m × nP)
        r    = R.reshape(-1)
        dof  = max(r.size - p, 1)
        s2   = float((r @ r) / dof)

        # Jacobiano proyectado (denso)
        dC_all_abs = _build_dC_all(Co, Ms, nas, param_idx)      # (m × n_abs × p)
        dXi_all = dC_all_abs / H.reshape(-1, 1, 1)              # (m × n_abs × p)
        J = _jac_varpro(Xi, coef, dXi_all, use_projector=use_projector, rcond=rcond)

        Cov_log10K = s2 * _pinv_backend(J @ J.T, rcond=rcond)
        SE_log10K = _onp.sqrt(_onp.clip(_onp.diag(_as_onp(Cov_log10K)), 0.0, _onp.inf))
        pm = percent_metrics_from_log10K(_as_onp(k), SE_log10K)

        percK = pm["perc_linear"]; SE_K = pm["SE_K"]
        rms = float(np.sqrt(np.mean(R * R)))
        return {
            "percK": percK, "SE_K": SE_K, "SE_log10K": SE_log10K,
            "Cov_log10K": _as_onp(Cov_log10K), "rms": rms, "covfit": s2,
            "coef": _as_onp(coef), "xi": _as_onp(Xi), "J": _as_onp(J),
        }

    # ---------- CAMINO CON MÁSCARA (ignora huecos por columna) ----------
    mask = _onp.asarray(mask, dtype=bool)
    nP = dq.shape[1]

    coef = _onp.zeros((n_abs, nP), dtype=float)
    R_blocks = []   # lista de residuales por columna, sólo filas observadas
    J_cols   = []   # lista de J (p × m_j) por columna

    # Precompute dC_all_abs (m × n_abs × p)
    dC_all_abs = _build_dC_all(Co, Ms, nas, param_idx)

    for j in range(nP):
        mj = mask[:, j]
        if not mj.any():
            continue
        Xi_m = Xi[mj, :]                        # (m_j × n_abs)
        y    = dq[mj, j]                         # (m_j,)

        # coeficiente de la señal j usando SOLO filas observadas
        Aj = _pinv_backend(Xi_m, rcond=rcond) @ y[:, None]   # (n_abs × 1)
        coef[:, j] = Aj[:, 0]

        # residuales de la señal j
        yfit = (Xi_m @ Aj).ravel()              # (m_j,)
        R_blocks.append(yfit - y)

        # Jacobiano para la señal j: dXi en filas observadas
        dXi_m = (dC_all_abs[mj, :, :] / H[mj].reshape(-1, 1, 1))   # (m_j × n_abs × p)
        Jj = _jac_varpro(Xi_m, Aj, dXi_m, use_projector=use_projector, rcond=rcond)  # (p × m_j)
        J_cols.append(Jj)

    if not J_cols:
        raise ValueError("No hay datos observados para calcular errores (máscara vacía).")

    r  = _onp.concatenate([_as_onp(rb) for rb in R_blocks], axis=0)   # (M_eff,)
    J  = _onp.concatenate([_as_onp(Jc) for Jc in J_cols], axis=1)     # (p × M_eff)
    dof = max(r.size - p, 1)
    s2  = float((r @ r) / dof)

    Cov_log10K = s2 * _pinv_backend(J @ J.T, rcond=rcond)
    SE_log10K  = _onp.sqrt(_onp.clip(_onp.diag(_as_onp(Cov_log10K)), 0.0, _onp.inf))
    pm = percent_metrics_from_log10K(_as_onp(k), SE_log10K)

    percK = pm["perc_linear"]; SE_K = pm["SE_K"]
    # RMS sobre observados únicamente
    RMS = float(_onp.sqrt(_onp.mean(_onp.concatenate(R_blocks)**2)))

    return {
        "percK": percK, "SE_K": SE_K, "SE_log10K": SE_log10K,
        "Cov_log10K": _as_onp(Cov_log10K), "rms": RMS, "covfit": s2,
        "coef": _as_onp(coef), "xi": _as_onp(Xi), "J": _as_onp(J),
    }
