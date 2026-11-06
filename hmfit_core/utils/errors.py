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

def _jax_array_type():
    """jax.numpy.ndarray if JAX is importable, else None. Resolved once: the previous
    per-call ``import jax.numpy`` cost ~25 % of a one-step bootstrap when JAX is absent."""
    global _JAX_ARRAY_TYPE
    if _JAX_ARRAY_TYPE is _UNRESOLVED:
        try:
            import jax.numpy as jnp
            _JAX_ARRAY_TYPE = jnp.ndarray
        except Exception:
            _JAX_ARRAY_TYPE = None
    return _JAX_ARRAY_TYPE


_UNRESOLVED = object()
_JAX_ARRAY_TYPE = _UNRESOLVED


def _as_onp(x):
    jax_type = _jax_array_type()
    if jax_type is not None and isinstance(x, jax_type):
        return _onp.array(x)
    return _onp.array(x)


class _FixedPointConcentrations:
    """Wrap a speciation solver so that ``concentraciones(k)`` is solved once per distinct
    ``k``. The one-step bootstraps re-evaluate the metrics at the same k_hat for every
    replicate (only the data change), so the speciation, ~50 % of their time, is reused.
    The solver is deterministic and stateless between calls, hence results are identical;
    copies are returned so callers cannot alter the cached arrays."""

    def __init__(self, solver):
        self._solver = solver
        self._key = None
        self._value = None

    def concentraciones(self, k):
        key = _onp.asarray(k, dtype=float).tobytes()
        if key != self._key:
            self._value = self._solver.concentraciones(k)
            self._key = key
        return tuple(_onp.array(v, copy=True) if isinstance(v, _onp.ndarray) else v for v in self._value)

    def __getattr__(self, name):
        return getattr(self._solver, name)

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

    # V: (m × p × nw)
    V = np.tensordot(dC_all, A, axes=(1, 0))
    if use_projector:
        V = (IminusP @ V.reshape(m, p * nw)).reshape(m, p, nw)

    # J: (p × m*nw)
    J = V.transpose(1, 0, 2).reshape(p, m * nw)
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


def _weighted_column_lstsq(C, YT, W, rcond=1e-10):
    """
    Per-wavelength weighted least-squares projection.

    C  : (m, s)    concentration/design matrix (rows = titration points)
    YT : (m, nw)   observed data matrix (columns = wavelengths)
    W  : (m, nw)   residual weights (the data term is sum_i W[i,j]^2 (C a - y_j)^2)

    W holds residual multipliers (the data term is sum_i (W[i,j] r_ij)^2), so
    each column system has its rows scaled by W[:, j] and a plain
    least-squares problem is solved; this is the exact weighted solution and
    reduces to the unweighted projection when the column of W is constant
    (positive scaling cancels in the argmin). Zero-weight rows drop out of the
    column problem entirely.

    Returns (A, n_linear_per_lambda):
      A (s, nw) coefficients; n_linear_per_lambda[j] = rank of the effective
      design for wavelength j (used for the residual degrees of freedom).
    """
    C = _onp.asarray(C, dtype=float)
    YT = _onp.asarray(YT, dtype=float)
    W = _onp.asarray(W, dtype=float)
    m, nw = YT.shape
    s = C.shape[1]
    A = _onp.zeros((s, nw), dtype=float)
    n_lin = _onp.zeros(nw, dtype=int)
    for j in range(nw):
        f = W[:, j]
        valid = f > 0.0
        if not _onp.any(valid):
            continue
        Cs = C[valid] * f[valid, None]
        ys = YT[valid, j] * f[valid]
        A[:, j] = _onp.linalg.lstsq(Cs, ys, rcond=rcond)[0]
        n_lin[j] = int(_onp.linalg.matrix_rank(Cs, tol=None))
    return np.asarray(A), n_lin


def _jac_varpro_weighted(C, A, dC_all, W, rcond=1e-10):
    """
    Variable-projection Jacobian consistent with per-wavelength weighted columns.

    Same sign/shape convention as ``_jac_varpro`` (J is (p, m*nw) over the
    row-major flattening of the residual matrix, J = +(I - P_j) applied to the
    scaled column derivative). W holds residual multipliers; for each
    wavelength j (with f = W[:, j]):
        r_j = f * (C a_j - y_j)   (the exact weighted residual),
        J_j = (I - P_j) diag(f) dC a_j,   P_j = projector of col(diag(f) C),
    so J_j = d r_j / d theta up to sign. For constant f this equals the
    unweighted projector result multiplied by f, i.e. identical to the flat
    ``_jac_varpro(...) * W.ravel()`` convention.
    """
    C = _onp.asarray(C, dtype=float)
    A = _onp.asarray(A, dtype=float)
    dC_all = _onp.asarray(dC_all, dtype=float)
    W = _onp.asarray(W, dtype=float)
    m, nw = W.shape
    p = dC_all.shape[2]
    J = _onp.zeros((p, m * nw), dtype=float)
    for j in range(nw):
        f = W[:, j]
        valid = f > 0.0
        if not _onp.any(valid):
            continue
        idx = _onp.flatnonzero(valid)
        V = _onp.einsum("isp,s->ip", dC_all[idx], A[:, j])  # (nv, p)
        V = V * f[idx, None]
        Cs = C[idx] * f[idx, None]
        P = Cs @ _onp.linalg.pinv(Cs, rcond=rcond)
        Vp = V - P @ V
        J[:, idx * nw + j] = Vp.T
    return np.asarray(J)


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


def _coerce_spectro_group_map(group_map, n_abs):
    n_abs = int(max(n_abs, 0))
    if n_abs == 0:
        return _onp.zeros((0, 0), dtype=float)
    if group_map is None:
        return _onp.eye(n_abs, dtype=float)
    arr = _onp.asarray(group_map, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"group_map must be 2D, got shape={arr.shape}")
    if arr.shape[0] != n_abs:
        raise ValueError(f"group_map row mismatch: expected {n_abs}, got {arr.shape[0]}")
    if arr.shape[1] <= 0:
        raise ValueError("group_map must contain at least one group.")
    return arr

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

def compute_errors_spectro_varpro(
    k,
    res,
    Y,
    modelo,
    nas,
    rcond=1e-10,
    use_projector=True,
    param_names=None,
    weights=None,
    group_map=None,
    param_transform=None,
    count_linear_dof=True,
    fixed_mask=None,
):
    """
    Cálculo robusto (variable projection) para espectroscopía.

    fixed_mask : optional (p,) bool. Fixed parameters are excluded from the covariance
                 (SE = 0 by definition) and from the degrees of freedom.
    k       : (p,)
    res     : solver con .concentraciones(k) -> (C_abs, Co_all)
    Y       : (m × nw)  datos (filas: puntos de titulación; cols: longitudes de onda)
    modelo  : M (nspec×n_comp) o (n_comp×nspec)
    nas     : lista[int] índices NO absorbentes
    weights : None, vector (nw,) o matriz (m,nw) para ponderar residuo/Jacobiano.
    """
    C, Co = res.concentraciones(k)    # C: (m × n_abs), Co: (m × nspec)
    nspec = Co.shape[1]
    Ms, n_comp = _normalize_modelo(modelo, nspec)

    p = len(k)
    param_transform_onp = None
    base_param_idx = list(range(n_comp, nspec))
    if param_transform is not None:
        param_transform_onp = _onp.asarray(param_transform, dtype=float)
        if param_transform_onp.ndim != 2:
            raise ValueError(
                f"param_transform must be 2D, got shape={param_transform_onp.shape}"
            )
        if param_transform_onp.shape[0] != len(base_param_idx):
            raise ValueError(
                "param_transform row mismatch: "
                f"expected {len(base_param_idx)}, got {param_transform_onp.shape[0]}"
            )
        if param_transform_onp.shape[1] != p:
            raise ValueError(
                "param_transform column mismatch: "
                f"expected {p}, got {param_transform_onp.shape[1]}"
            )
        param_idx = base_param_idx
    else:
        param_idx = base_param_idx
        if len(param_idx) != p:
            if p <= nspec:
                param_idx = list(range(nspec - p, nspec))
            else:
                raise ValueError(f"p={p} > nspec={nspec}")

    G_map_onp = _coerce_spectro_group_map(group_map, C.shape[1])
    G_map = np.asarray(G_map_onp, dtype=float)
    C_eff = np.asarray(_as_onp(C) @ G_map_onp, dtype=float)

    # Ajuste lineal y residuo (se resuelve abajo, una vez conocidos los pesos;
    # aquí solo se necesitan las formas para construir la matriz de pesos).
    m_obs, nw = int(Y.shape[1]), int(Y.shape[0])   # Y: (nw × m) -> R: (m × nw)

    # Pesos espectrales (por lambda) o por punto.
    W_onp = _onp.ones((m_obs, nw), dtype=float)
    if weights is not None:
        w_in = _onp.asarray(weights, dtype=float)
        w_in = _onp.nan_to_num(_onp.abs(w_in), nan=0.0, posinf=0.0, neginf=0.0)
        if w_in.ndim == 1 and w_in.size == nw:
            W_onp = _onp.broadcast_to(w_in.reshape(1, nw), (m_obs, nw)).copy()
        elif w_in.shape == (m_obs, nw):
            W_onp = w_in
        else:
            try:
                W_onp = _onp.broadcast_to(w_in, (m_obs, nw)).copy()
            except ValueError:
                W_onp = _onp.ones((m_obs, nw), dtype=float)
    W = np.asarray(W_onp, dtype=float)
    # Phase 2: when the weights vary along the titration points (per-point
    # matrix m x nw), the linear projection must be the weighted one: for each
    # wavelength the rows are scaled by the residual multiplier W (the square
    # root of the corresponding statistical weight) and solved column by column.
    # With weights constant per wavelength, weighted and unweighted projections
    # coincide (positive column scaling
    # cancels in the argmin), so the fast global-pinv path is kept.
    per_point_weights = bool(np.any(W != W[0:1, :]))
    if per_point_weights:
        A_group, n_lin_per_lambda = _weighted_column_lstsq(
            np.asarray(C_eff, dtype=float), np.asarray(Y, dtype=float).T, W, rcond=rcond
        )
    else:
        A_group = _pinv_backend(C_eff, rcond=rcond) @ Y.T      # (n_group × nw)
        n_lin_per_lambda = None
    A_species = G_map @ A_group                            # (n_abs × nw)
    R   = (C_eff @ A_group - Y.T)                          # (m × nw)
    R_w = R * W
    r = R_w.reshape(-1)
    # Linear parameters (pure spectra, rank(C_eff) per wavelength) also consume dof.
    # Only observations with positive weight are data; a wavelength with zero weight
    # everywhere contributes neither observations nor fitted linear coefficients.
    used = W > 0.0
    n_obs_used = int(np.count_nonzero(used))
    n_wl_used = int(np.count_nonzero(np.any(used, axis=0)))
    if per_point_weights:
        # Per-wavelength rank of the effective design restricted to the rows
        # with positive weight (row scaling does not change rank).
        n_linear = int(np.sum(np.asarray(n_lin_per_lambda)))
    else:
        try:
            rank_c = int(_onp.linalg.matrix_rank(_as_onp(C_eff), tol=None))
        except Exception:
            rank_c = int(C_eff.shape[1])
        n_linear = int(rank_c * n_wl_used)

    fixed = _onp.zeros(p, dtype=bool)
    if fixed_mask is not None:
        fm = _onp.asarray(fixed_mask, dtype=bool).ravel()
        if fm.size == p:
            fixed = fm
        elif fm.size != 0:
            raise ValueError(f"fixed_mask has {fm.size} entries for {p} parameters.")
    free_idx = _onp.where(~fixed)[0]

    # Jacobiano proyectado
    dC_all = _build_dC_all(Co, Ms, nas, param_idx)
    if param_transform_onp is not None:
        dC_all = np.tensordot(dC_all, np.asarray(param_transform_onp, dtype=float), axes=(2, 0))
    dC_eff = np.transpose(np.tensordot(dC_all, G_map, axes=(1, 0)), (0, 2, 1))
    if per_point_weights:
        J = _jac_varpro_weighted(C_eff, A_group, dC_eff, W, rcond=rcond)
    else:
        J_raw = _jac_varpro(C_eff, A_group, dC_eff, use_projector=use_projector, rcond=rcond)
        W_flat = W.reshape(-1)
        J = J_raw * W_flat[None, :]

    # Covariance of the free parameters (fixed ones: exactly zero, by definition)
    J_onp = _as_onp(J)
    J_free = J_onp[free_idx, :]
    lin = linearized_covariance(
        J_free.T, float(_as_onp(r) @ _as_onp(r)), n_obs_used,
        n_linear if count_linear_dof else 0,
    )
    dof = lin["dof"]
    s2 = lin["s2"]
    Cov_log10K = _onp.zeros((p, p), dtype=float)
    Cov_log10K[_onp.ix_(free_idx, free_idx)] = lin["cov"]

    param_names_free = None
    if param_names is not None:
        names_all = list(param_names)
        param_names_free = [names_all[i] for i in free_idx] if len(names_all) == p else None
    JJT = J_free @ J_free.T
    if free_idx.size:
        diag = build_identifiability_report(JJT, _onp.nan_to_num(lin["cov"]), param_names=param_names_free)
    else:
        diag = {"status": "warn", "diag_summary": "All parameters are fixed; nothing estimated."}
    if lin["notes"]:
        diag = dict(diag)
        diag["notes"] = list(lin["notes"])

    d = _onp.diag(Cov_log10K)
    SE_log10K = _onp.where(_onp.isfinite(d), _onp.sqrt(_onp.clip(d, 0.0, _onp.inf)), _onp.nan)
    pm = percent_metrics_from_log10K(_as_onp(k), SE_log10K)

    percK      = pm["perc_linear"]      # %Error (K) por delta method (simétrico)
    SE_K       = pm["SE_K"]         
    perc_hi    = pm["perc_hi"]          # % asimétrico (arriba)
    perc_lo    = pm["perc_lo"]      # % asimétrico (abajo)

    RMS = float(np.sqrt(np.mean(R_w * R_w)))
    RMS_raw = float(np.sqrt(np.mean(R * R)))
    return {
        "percK": percK, "SE_K": SE_K, "SE_log10K": SE_log10K,
        "Cov_log10K": _as_onp(Cov_log10K), "RMS": RMS, "s2": s2,
        "A": _as_onp(A_group),
        "A_species": _as_onp(A_species),
        "group_map": _as_onp(G_map),
        "J": _as_onp(J), "yfit": _as_onp((C_eff @ A_group).T),
        "r": _as_onp(r), "dof": int(dof), "nobs": n_obs_used, "n_linear": int(n_linear),
        "RMS_raw": RMS_raw,
        "stability_diag": diag,
        "fixed_mask": fixed, "free_idx": free_idx,
        "nonidentifiable": _onp.isin(_onp.arange(p), free_idx[lin["nonidentifiable"]]),
        "notes": list(lin["notes"]), "variance_valid": bool(lin["valid"]),
    }


def compute_errors_nmr_varpro(
    k,
    res,
    dq,
    H,
    modelo,
    nas,
    stoichiometry=None,
    rcond=1e-10,
    use_projector=True,
    mask=None,
    fixed_mask=None,
    weights=None,  # NEW argument: (nP,) or (m, nP)
    ridge=1e-8,
    rcond_cov=None, ridge_cov=0.0, debug=False,
    param_names=None,
    count_linear_dof=True,
):
    """
    NMR version with missing data support and Weighted Covariance (IRLS).
    If 'mask' is None, uses all finite dq values.
    If 'mask' is bool (m x nP), ignores unobserved rows per column.

    Now supports 'fixed_mask' to correctly account for degrees of freedom.
    Now supports 'weights' for IRLS (Signal-based weighting).
    Optional 'stoichiometry' enables per-signal fractions for Xi (HypNMR-like).
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
    
    # Process weights
    # If weights is None, use 1.0
    # weights can be shape (nP,) -> per signal
    # or (m, nP) -> per point
    w_mat = _onp.ones_like(dq)
    if weights is not None:
        weights = _onp.asarray(weights, dtype=float)
        if weights.shape == (nP,):
            w_mat = w_mat * weights[None, :]
        elif weights.shape == dq.shape:
            w_mat = weights
        else:
            # Fallback or error? Assuming broadcastable or scalar
            try:
                w_mat = w_mat * weights
            except:
                pass

    coef = _onp.zeros((n_abs, nP), dtype=float)
    R_blocks = []   # lista de residuales por columna, sólo filas observadas
    J_cols   = []   # lista de J (p × m_j) por columna
    W_blocks = []   # lista de pesos (sqrt) correspondientes a R_blocks
    n_linear = 0    # coeficientes lineales efectivamente estimados (desplazamientos límite)
    row_blocks = [] # índice de punto de titulación de cada residual (para bootstrap por bloques)

    S = None
    if stoichiometry is not None:
        try:
            S = _onp.asarray(stoichiometry, dtype=float)
            if S.size == 0 or S.ndim != 2:
                S = None
            else:
                if S.shape == (nP, n_abs):
                    S = S.T
                if S.shape == (nspec, nP):
                    nas_idx = []
                    try:
                        if nas is not None:
                            nas_idx = [i for i in list(nas) if 0 <= int(i) < nspec]
                    except Exception:
                        nas_idx = []
                    if nas_idx:
                        S = _onp.delete(S, nas_idx, axis=0)
                if S.shape != (n_abs, nP):
                    S = None
        except Exception:
            S = None

    # Precompute dC_all_abs (m × n_abs × p)
    dC_all_abs = _build_dC_all(Co, Ms, nas, param_idx)

    for j in range(nP):
        mj_obs = mask[:, j]
        if S is not None:
            S_j = S[:, j]
            D_all = C @ S_j
            mj = mj_obs & _onp.isfinite(D_all) & (_onp.abs(D_all) > 1e-12)
        else:
            # Signal-specific parent concentration
            Hj_all = H[:, j] if H.ndim == 2 else H
            # Point is valid for signal j if:
            # 1. It is observed (mask)
            # 2. Parent concentration > threshold
            mj = mj_obs & (_onp.abs(Hj_all) > 1e-12)
        
        if not mj.any():
            continue

        # Calculate Xi_m LOCALLY to avoid global inf
        if S is not None:
            D = D_all[mj]
            Xi_m = (C[mj, :] * S_j[None, :]) / D[:, None]  # (m_j × n_abs)
        else:
            Xi_m = C[mj, :] / Hj_all[mj, None]      # (m_j × n_abs)
        y    = dq[mj, j]                        # (m_j,)
        w_j  = w_mat[mj, j]                     # (m_j,)
        sq_w = _onp.sqrt(w_j)                   # (m_j,)

        # Finiteness guard
        if not _onp.isfinite(Xi_m).all() or not _onp.isfinite(y).all():
            import warnings
            warnings.warn(f"Signal {j}: Non-finite values in basis or data. Skipping.")
            continue

        # Weighted Least Squares for coeffs:
        # min || W^(1/2) * (Xi * beta - y) ||^2
        # equivalent to OLS on (sq_w * Xi, sq_w * y)
        Xi_w = Xi_m * sq_w[:, None]
        y_w  = y * sq_w
        
        try:
            delta, _, rank_j, _ = _onp.linalg.lstsq(Xi_w, y_w, rcond=rcond)
            n_linear += int(rank_j)
        except _onp.linalg.LinAlgError:
            n_linear += int(Xi_w.shape[1])
            if ridge is not None and float(ridge) > 0.0:
                XtX = Xi_w.T @ Xi_w
                delta = _onp.linalg.solve(XtX + float(ridge) * _onp.eye(XtX.shape[0]), Xi_w.T @ y_w)
            else:
                delta = _onp.linalg.pinv(Xi_w, rcond=rcond) @ y_w

        Aj = delta.reshape(-1, 1)
        coef[:, j] = Aj[:, 0]

        # residuales de la señal j (original scale)
        yfit = (Xi_m @ Aj).ravel()              # (m_j,)
        # Note: 'r' for covariance calculation uses weighted residuals?
        # Standard approach: Cov = inv(J.T * W * J) * sigma_sq
        # Where sigma_sq is reduced chi-squared generally. 
        # But 'R_blocks' is returned as 'residuals_vec'. Often users want raw residuals.
        # We will store RAW residuals in R_blocks for output, 
        # but apply weights for J and singular value calculation if needed.
        R_blocks.append(yfit - y)
        W_blocks.append(sq_w)
        row_blocks.append(_onp.where(mj)[0])

        # Jacobiano para la señal j: dXi en filas observadas
        if S is not None:
            dC_m = dC_all_abs[mj, :, :]
            dD_m = _onp.tensordot(dC_m, S_j, axes=(1, 0))
            dXi_m = (dC_m * S_j[None, :, None]) / D[:, None, None]
            dXi_m = dXi_m - Xi_m[:, :, None] * dD_m[:, None, :] / D[:, None, None]
        else:
            dXi_m = (dC_all_abs[mj, :, :] / Hj_all[mj].reshape(-1, 1, 1))   # (m_j × n_abs × p)
        Jj = _jac_varpro(Xi_m, Aj, dXi_m, use_projector=use_projector, rcond=rcond)  # (p × m_j)
        J_cols.append(Jj)

    if not J_cols:
        raise ValueError("No hay datos observados válidos para calcular errores (padre concentracion ~0 o máscara vacía).")

    r_raw   = _onp.concatenate([_as_onp(rb) for rb in R_blocks], axis=0)       # (M_eff,)
    J_full  = _onp.concatenate([_as_onp(Jc) for Jc in J_cols], axis=1)         # (p_total × M_eff)
    w_full  = _onp.concatenate([_as_onp(wb) for wb in W_blocks], axis=0)       # (M_eff,)
    
    # Weighted Jacobian for Covariance: J_w = J * W^(1/2) (broadcast over columns M_eff)
    # J_full is (p x M_eff). w_full is (M_eff,)
    J_w = J_full * w_full[None, :]

    # Observations actually used: finite, unmasked and with positive weight.
    nobs = int(_onp.count_nonzero(w_full > 0.0))
    chi2 = float(_onp.sum((r_raw * w_full) ** 2))
    # Degrees of freedom: the limiting shifts / linear coefficients solved by variable
    # projection are fitted parameters too (n_linear = sum of the effective ranks of the
    # per-signal design matrices). No silent clamp: dof <= 0 -> variance undefined (NaN).
    nonid_full = _onp.zeros(p_total, dtype=bool)
    notes = []

    if p_free == 0:
        dof = nobs - (int(n_linear) if count_linear_dof else 0)
        s2 = chi2 / dof if dof > 0 else float("nan")
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
        # Use Weighted Jacobian for Fisher Info Matrix
        J = J_w[free_idx, :]   # (p_free x M_eff)
        JJT = J @ J.T          # (p_free x p_free)
        JJT = 0.5 * (JJT + JJT.T.conj())

        lin = linearized_covariance(J.T, chi2, nobs, int(n_linear) if count_linear_dof else 0)
        dof = lin["dof"]
        s2 = lin["s2"]
        Cov_free = lin["cov"]
        notes.extend(lin["notes"])
        nonid_full[free_idx] = lin["nonidentifiable"]

        # --- Stability Diagnostics ---
        param_names_free = None
        if param_names is not None:
            param_names_free = [param_names[i] for i in free_idx]
            
        stability_diag = build_identifiability_report(JJT, _onp.nan_to_num(Cov_free), param_names=param_names_free)
        if notes:
            stability_diag = dict(stability_diag)
            stability_diag["notes"] = list(notes)

        if debug:
            print(stability_diag["diag_full"])

        d_free = _onp.diag(_as_onp(Cov_free))
        SE_log10K_free = _onp.where(_onp.isfinite(d_free), _onp.sqrt(_onp.clip(d_free, 0.0, _onp.inf)), _onp.nan)

        SE_log10K_full = _onp.zeros(p_total)
        SE_log10K_full[free_idx] = SE_log10K_free
    
    pm = percent_metrics_from_log10K(k, SE_log10K_full)

    return {
        "percK": pm["perc_linear"], "SE_K": pm["SE_K"], "SE_log10K": SE_log10K_full,
        "Cov_log10K_free": _as_onp(Cov_free), "rms": float(_onp.sqrt(_onp.mean(r_raw**2))), "covfit": s2,
        "coef": _as_onp(coef), "xi": None, "J": _as_onp(J_full),
        "dof": dof, "nobs": nobs, "n_linear": int(n_linear),
        "row_index": _onp.concatenate(row_blocks) if row_blocks else _onp.zeros(0, dtype=int),
        "residuals_vec": _as_onp(r_raw), "r": _as_onp(r_raw),
        "stability_diag": stability_diag,
        "fixed_mask": _onp.asarray(fixed_mask, dtype=bool), "free_idx": free_idx,
        "nonidentifiable": nonid_full, "notes": notes,
        "variance_valid": bool(_onp.isfinite(s2)),
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


# --- Statistical helpers shared by every module -----------------------------

def linearized_covariance(J_free, sse, n_obs, n_linear=0, *, rank_tol=None):
    """
    Linearized covariance of the *free* nonlinear parameters, s^2 (J^T J)^-1.

    J_free  : (n_obs, p_free) Jacobian of the (weighted) residual vector w.r.t. the free
              parameters (rows = observations).
    sse     : weighted residual sum of squares.
    n_linear: number of linear coefficients estimated by variable projection.

    Returns a dict with
      dof, s2           : n_obs - p_free - n_linear and SSE/dof. If dof <= 0 the variance
                          cannot be estimated: s2 = NaN and ``valid`` is False (no silent
                          clamp of dof to 1).
      cov               : (p_free, p_free) covariance; NaN rows/cols for non-identifiable
                          parameters, i.e. those with a non-negligible component in the
                          numerical null space of the column-scaled Jacobian.
      rank, nonidentifiable (bool mask), notes (list of str), valid (bool).
    A standard error of zero is therefore never used to mean "not identifiable".
    """
    J = _onp.asarray(J_free, dtype=float)
    if J.ndim != 2:
        raise ValueError("J_free must be 2-D (n_obs, p_free).")
    n_obs = int(n_obs)
    p = int(J.shape[1])
    n_linear = int(n_linear or 0)
    notes = []
    dof = n_obs - p - n_linear
    if dof <= 0:
        s2 = float("nan")
        notes.append(
            f"Insufficient degrees of freedom (n_obs={n_obs}, nonlinear={p}, linear={n_linear}, "
            f"dof={dof}): the residual variance and the uncertainties cannot be estimated."
        )
    else:
        s2 = float(sse) / float(dof)
    cov = _onp.full((p, p), _onp.nan, dtype=float)
    nonid = _onp.zeros(p, dtype=bool)
    rank = 0
    if p == 0:
        return {"cov": _onp.zeros((0, 0)), "dof": dof, "s2": s2, "rank": 0,
                "nonidentifiable": nonid, "notes": notes, "valid": dof > 0}
    if not _onp.all(_onp.isfinite(J)):
        notes.append("Jacobian contains non-finite values; covariance not available.")
        return {"cov": cov, "dof": dof, "s2": s2, "rank": 0,
                "nonidentifiable": _onp.ones(p, dtype=bool), "notes": notes, "valid": False}
    norms = _onp.linalg.norm(J, axis=0)
    zero_col = norms <= 0.0
    scale = _onp.where(zero_col, 1.0, norms)
    Js = J / scale[None, :]
    U, sv, Vt = _onp.linalg.svd(Js, full_matrices=False)
    if rank_tol is None:
        rank_tol = max(J.shape) * _onp.finfo(float).eps * (sv[0] if sv.size else 0.0)
    keep = sv > rank_tol
    rank = int(_onp.count_nonzero(keep))
    if rank < p:
        V_null = Vt[~keep, :].T  # (p, p-rank)
        nonid = (_onp.linalg.norm(V_null, axis=1) > 1e-6) | zero_col
        names = ", ".join(str(i + 1) for i in _onp.where(nonid)[0])
        notes.append(
            f"Jacobian is rank deficient (rank {rank} < {p} free parameters): parameter(s) "
            f"#{names} are not identifiable from these data; their uncertainty is undefined (NaN)."
        )
    inv = Vt[keep, :].T @ _onp.diag(1.0 / sv[keep] ** 2) @ Vt[keep, :]
    cov_s = s2 * inv
    cov = cov_s / scale[:, None] / scale[None, :]
    cov[nonid, :] = _onp.nan
    cov[:, nonid] = _onp.nan
    return {"cov": cov, "dof": dof, "s2": s2, "rank": rank, "nonidentifiable": nonid,
            "notes": notes, "valid": bool(dof > 0 and rank > 0)}


def t_quantile(dof, prob=0.975):
    """Student-t quantile for ``dof`` degrees of freedom (normal if dof is huge/invalid)."""
    try:
        dof = float(dof)
    except Exception:
        dof = float("inf")
    if not _onp.isfinite(dof) or dof <= 0:
        dof = float("inf")
    try:
        from scipy.stats import t as _t, norm as _norm
        if _onp.isinf(dof):
            return float(_norm.ppf(prob))
        return float(_t.ppf(prob, dof))
    except Exception:  # pragma: no cover - scipy is a hard dependency
        return 1.959963984540054 if prob == 0.975 else 1.0


def nonidentifiable_parameters(J_rows_params):
    """Boolean mask of parameters lying in the numerical null space of J (rows = params)."""
    J = _onp.asarray(J_rows_params, dtype=float)
    if J.ndim != 2 or J.shape[0] == 0:
        return _onp.zeros(J.shape[0] if J.ndim == 2 else 0, dtype=bool)
    return linearized_covariance(J.T, 0.0, J.shape[1] + J.shape[0] + 1)["nonidentifiable"]


def lm_damped_step(J, r, lam=1e-3, rcond=1e-10):
    """
    One Levenberg-Marquardt step with *scale-invariant* (Marquardt) damping.

    J : (p, m) Jacobian (rows = parameters), r : (m,) residual vector.
    Solves (G + lam*diag(G)) dk = -J r with G = J J^T.

    A plain ``lam*I`` damping depends on the units of the parameters and of the
    signal: when ``diag(G)`` is small (e.g. rate constants in M^-1 s^-1 or weak
    absorbances) it dominates G and shrinks every bootstrap step towards zero,
    which silently underestimates the uncertainty by orders of magnitude.
    """
    J = _onp.asarray(J, dtype=float)
    r = _onp.asarray(r, dtype=float).ravel()
    G = J @ J.T
    d = _onp.diag(G).copy()
    d_floor = 1e-15 * (float(_onp.max(d)) if d.size and _onp.max(d) > 0 else 1.0)
    d = _onp.maximum(d, d_floor)
    A = G + float(lam) * _onp.diag(d)
    g = J @ r
    try:
        return -_onp.linalg.solve(A, g)
    except _onp.linalg.LinAlgError:
        return -_onp.linalg.pinv(A, rcond=rcond) @ g


def sample_correlation(samples):
    """
    Correlation matrix of bootstrap samples. Columns with zero spread (fixed parameters)
    or non-finite values (non-identifiable) get NaN correlations (undefined), diagonal 1
    for the others. Always 2-D, also for a single parameter.
    """
    X = _onp.atleast_2d(_onp.asarray(samples, dtype=float))
    p = X.shape[1]
    corr = _onp.full((p, p), _onp.nan)
    if X.shape[0] < 2:
        return corr
    ok = _onp.all(_onp.isfinite(X), axis=0) & (_onp.std(X, axis=0) > 0)
    idx = _onp.where(ok)[0]
    if idx.size:
        sub = _onp.atleast_2d(_onp.corrcoef(X[:, idx], rowvar=False))
        corr[_onp.ix_(idx, idx)] = sub
        corr[idx, idx] = 1.0
    return corr


def correlation_from_covariance(cov, undefined=None):
    """
    Correlation matrix from a covariance matrix under the HM fit convention: rows and
    columns whose variance is zero (fixed parameters) or non-finite (non-identifiable),
    or that are flagged in ``undefined``, are NaN; the diagonal of the rest is 1.
    """
    C = _onp.atleast_2d(_onp.asarray(cov, dtype=float))
    p = C.shape[0]
    var = _onp.diag(C).copy() if p else _onp.zeros(0)
    ok = _onp.isfinite(var) & (var > 0)
    if undefined is not None:
        ok &= ~_onp.asarray(undefined, dtype=bool).reshape(-1)
    corr = _onp.full((p, p), _onp.nan)
    idx = _onp.where(ok)[0]
    if idx.size:
        sd = _onp.sqrt(var[idx])
        sub = C[_onp.ix_(idx, idx)] / _onp.outer(sd, sd)
        corr[_onp.ix_(idx, idx)] = _onp.clip(sub, -1.0, 1.0)
        corr[idx, idx] = 1.0
    return corr


def enforce_uncertainty_convention(estimate, se, ci_lo, ci_hi, corr, *, fixed_mask=None,
                                   nonidentifiable=None, dof=None, extra_ci=()):
    """
    Apply the reporting convention shared by every technique and error method:

    * fixed parameter: SE = 0, every interval bound = estimate, correlations NaN;
    * non-identifiable parameter: SE and intervals NaN, correlations NaN;
    * dof <= 0: SE and intervals of every free parameter NaN, all correlations NaN.

    Returns new arrays ``(se, ci_lo, ci_hi, corr, extra_ci, notes)``; ``extra_ci`` are
    additional interval arrays (e.g. 16/84 %) treated like ``ci_lo``/``ci_hi``.
    """
    est = _onp.asarray(estimate, dtype=float).reshape(-1)
    p = est.size
    se = _onp.array(se, dtype=float).reshape(-1)
    lo = _onp.array(ci_lo, dtype=float).reshape(-1)
    hi = _onp.array(ci_hi, dtype=float).reshape(-1)
    extra = [None if a is None else _onp.array(a, dtype=float).reshape(-1) for a in extra_ci]
    corr = _onp.array(corr, dtype=float) if corr is not None else _onp.full((p, p), _onp.nan)
    corr = _onp.atleast_2d(corr)
    if corr.shape != (p, p):
        corr = _onp.full((p, p), _onp.nan)
    fixed = _onp.zeros(p, dtype=bool) if fixed_mask is None else _onp.asarray(fixed_mask, dtype=bool).reshape(-1)
    if fixed.size != p:
        fixed = _onp.zeros(p, dtype=bool)
    nonid = _onp.zeros(p, dtype=bool) if nonidentifiable is None else _onp.asarray(nonidentifiable, dtype=bool).reshape(-1)
    if nonid.size != p:
        nonid = _onp.zeros(p, dtype=bool)
    nonid &= ~fixed
    notes = []
    undefined = nonid.copy()
    try:
        dof_bad = dof is not None and float(dof) <= 0
    except (TypeError, ValueError):
        dof_bad = False
    if dof_bad:
        undefined |= ~fixed
        notes.append(
            f"Insufficient degrees of freedom (dof={dof}): uncertainties and correlations are undefined (NaN)."
        )
    for arr in [se, lo, hi] + [a for a in extra if a is not None]:
        arr[undefined] = _onp.nan
    se[fixed] = 0.0
    for arr in [lo, hi] + [a for a in extra if a is not None]:
        arr[fixed] = est[fixed]
    bad = fixed | undefined
    corr[bad, :] = _onp.nan
    corr[:, bad] = _onp.nan
    if dof_bad:
        corr[:, :] = _onp.nan
    good = _onp.where(~bad)[0]
    if not dof_bad:
        corr[good, good] = 1.0
    return se, lo, hi, corr, extra, notes


PARAM_STATUS_NOTES = {
    "fixed": "fixed parameter (not estimated; SE = 0)",
    "nonidentifiable": "not identifiable from these data (uncertainty undefined)",
    "dof": "insufficient degrees of freedom (dof <= 0; uncertainty undefined)",
    "failed": "the error analysis could not compute this value",
}


def parameter_status_codes(se, *, fixed_mask=None, nonidentifiable=None, dof=None):
    """
    Per-parameter reason code for the reported uncertainty: "" (defined), "fixed",
    "nonidentifiable", "dof" (dof <= 0) or "failed" (undefined for any other reason).
    Used by the GUI to explain undefined values and by exports for the "Note" column.
    """
    se = _onp.asarray(se, dtype=float).reshape(-1)
    p = se.size
    fixed = _onp.zeros(p, bool) if fixed_mask is None else _onp.asarray(fixed_mask, bool).reshape(-1)
    nonid = _onp.zeros(p, bool) if nonidentifiable is None else _onp.asarray(nonidentifiable, bool).reshape(-1)
    fixed = fixed if fixed.size == p else _onp.zeros(p, bool)
    nonid = nonid if nonid.size == p else _onp.zeros(p, bool)
    try:
        dof_bad = dof is not None and float(dof) <= 0
    except (TypeError, ValueError):
        dof_bad = False
    codes = []
    for i in range(p):
        if fixed[i]:
            codes.append("fixed")
        elif nonid[i]:
            codes.append("nonidentifiable")
        elif dof_bad:
            codes.append("dof")
        elif not _onp.isfinite(se[i]):
            codes.append("failed")
        else:
            codes.append("")
    return codes


def hc1_rescale_factor(n_obs, n_params_total):
    """
    Global HC1-type degrees-of-freedom correction sqrt(n / (n - p_total)).

    Raw least-squares residuals are on average smaller than the errors because the fitted
    parameters (nonlinear *and* linear) absorb part of the noise: E[sum r^2] = (n - p) s^2.
    Multiplying every residual by sqrt(n/(n-p)) restores the average variance before wild
    or residual resampling. It is a single global factor: it does NOT correct the
    leverage of individual observations (that would be HC2, r_i / sqrt(1 - h_ii), or HC3,
    r_i / (1 - h_ii), which require the diagonal of the hat matrix).
    Returns 1.0 when n - p_total <= 0 (no correction is possible).
    """
    n = int(n_obs)
    p = int(max(n_params_total, 0))
    if n <= 0 or n - p <= 0:
        return 1.0
    return float(_onp.sqrt(n / float(n - p)))


# Backwards-compatible alias (the factor was previously called a "leverage" correction,
# which is inaccurate: it is a global HC1-type correction).
leverage_rescale_factor = hc1_rescale_factor


def resampling_notes(n_blocks, block="row", n_params_free=None):
    """
    Statistical caveats of the wild bootstrap given the number of independent units.

    n_blocks: independent resampling units (titration points / spectra for block="row";
    individual residuals for block="element").
    """
    notes = []
    n_blocks = int(n_blocks)
    if str(block).lower().startswith("row"):
        if n_blocks < 20:
            notes.append(
                f"Only {n_blocks} independent blocks (titration points / spectra): with a "
                f"Rademacher wild bootstrap there are at most 2^{n_blocks} distinct resamples "
                "and the bootstrap variance is imprecise and can be biased low. Treat the "
                "intervals as indicative and compare with the analytic estimate."
            )
        if n_params_free is not None and n_blocks <= 2 * int(n_params_free):
            notes.append(
                f"The number of blocks ({n_blocks}) is small relative to the free parameters "
                f"({int(n_params_free)})."
            )
    else:
        notes.append(
            "Element-wise resampling assumes independent errors between channels of the "
            "same spectrum / titration point; correlated errors (baseline drift, dilution) "
            "make it optimistic."
        )
    return notes


def block_wild_multipliers(rng, shape, kind="rademacher", block="row"):
    """
    Wild multipliers for a residual matrix.

    block="row"     -> one multiplier per row (titration point / spectrum / time),
                       shared across all channels. Preserves correlation between
                       wavelengths or NMR signals measured in the same sample.
    block="element" -> independent multiplier per element (assumes i.i.d. noise).
    """
    shape = tuple(int(s) for s in (shape if isinstance(shape, (tuple, list)) else (shape,)))
    if len(shape) <= 1 or str(block or "row").lower().startswith("elem"):
        return _wild_weights(rng, shape, wild=kind)
    v = wild_multipliers(rng, shape[0], kind=kind).reshape((shape[0],) + (1,) * (len(shape) - 1))
    return _onp.broadcast_to(v, shape).astype(float)


def bootstrap_quality_notes(n_success, n_fail, B, max_fail_fraction=0.05, min_success=30):
    """Human readable warnings about the reliability of a bootstrap run."""
    notes = []
    B = max(int(B), 1)
    frac = float(n_fail) / B
    if n_fail and frac > max_fail_fraction:
        notes.append(
            f"{n_fail}/{B} replicates ({100*frac:.1f}%) failed and were discarded; "
            "failed replicates are usually the extreme ones, so the reported intervals may be too narrow."
        )
    if n_success < min_success:
        notes.append(
            f"Only {n_success} successful replicates; percentile intervals (2.5/97.5) need "
            "several hundred replicates to be stable. Treat them as indicative."
        )
    return notes


def make_wild_data_star_fn(fitted, observed, *, point_axis=0, block="row",
                           n_params_total=0, mask=None, kind_default="rademacher"):
    """
    Build ``make_data_star_fn(rng, wild=...)`` for full-refit bootstraps.

    fitted/observed : arrays of identical shape.
    point_axis      : axis that indexes titration points / spectra / times.
    block           : "row" (one multiplier per point, keeps channel correlation) or "element".
    n_params_total  : nonlinear + linear fitted parameters, for the global HC1-type correction.
    mask            : optional boolean array; unobserved entries are copied from ``observed``.
    """
    fitted = _onp.asarray(fitted, dtype=float)
    observed = _onp.asarray(observed, dtype=float)
    if fitted.shape != observed.shape:
        raise ValueError(f"fitted shape {fitted.shape} does not match observed {observed.shape}.")
    resid = observed - fitted
    valid = _onp.isfinite(resid) if mask is None else (_onp.asarray(mask, dtype=bool) & _onp.isfinite(resid))
    resid = _onp.where(valid, resid, 0.0)
    resid = resid * hc1_rescale_factor(int(_onp.count_nonzero(valid)), n_params_total)
    moved = _onp.moveaxis(resid, point_axis, 0) if resid.ndim > 1 else resid

    def _fn(rng, wild=kind_default):
        v = block_wild_multipliers(rng, moved.shape, kind=wild, block=block)
        r_star = _onp.moveaxis(moved * v, 0, point_axis) if resid.ndim > 1 else moved * v
        out = fitted + r_star
        return _onp.where(valid, out, observed)

    return _fn


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
    gc_collect=False,
    gc_every=1,
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
    gc_every = int(gc_every)
    if gc_every <= 0:
        raise ValueError("gc_every must be >= 1.")

    rng = _onp.random.default_rng(seed)
    samples_list = []
    n_fail = 0
    fail_info = []
    failure_reasons = {}
    n_fallback = 0
    fallback_methods = set()

    for b in range(B):
        if cancel_cb is not None and cancel_cb():
            raise BootstrapCancelled("Bootstrap cancelled.")
        data_star = make_data_star_fn(rng, wild=wild)
        theta0 = theta_hat.copy()
        theta_star, ok, info = refit_fn(data_star, theta0, max_iter=max_iter, tol=tol)
        theta_star = _onp.asarray(theta_star, dtype=float).ravel()
        if ok and (theta_star.size != theta_hat.size or not _onp.all(_onp.isfinite(theta_star))):
            ok = False
            info = {"failure_reason": "nonfinite_result", "error": "Invalid refit parameters"}
        if ok:
            samples_list.append(theta_star)
            if isinstance(info, dict):
                used = info.get("optimizer_used")
                requested = info.get("optimizer_requested")
                fallback_used = bool(info.get("fallback_used"))
                if fallback_used or (used and requested and used != requested):
                    n_fallback += 1
                    if used:
                        fallback_methods.add(str(used))
        else:
            n_fail += 1
            reason = str(info.get("failure_reason") or "unspecified") if isinstance(info, dict) else "unspecified"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            if info is not None and len(fail_info) < 3:
                fail_info.append(info)
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
        if gc_collect and ((b + 1) % gc_every == 0):
            data_star = None
            theta0 = None
            theta_star = None
            info = None
            import gc
            gc.collect()

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
        corr = sample_correlation(samples)

    return {
        "samples": samples,
        "n_attempted": int(n_success + n_fail),
        "n_success": int(n_success),
        "n_fail": int(n_fail),
        "failure_reasons": failure_reasons,
        "median": median,
        "p2_5": p2_5,
        "p97_5": p97_5,
        "p16": p16,
        "p84": p84,
        "corr": corr,
        "fail_info": fail_info,
        "quality_notes": bootstrap_quality_notes(n_success, n_fail, B),
        "n_fallback": int(n_fallback),
        "fallback_methods": sorted(fallback_methods),
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
    groups=None,
    n_linear=0,
    leverage_correction=True,
):
    """
    Fast bootstrap via linearized covariance and wild residuals.

    groups : optional (m,) integer labels (e.g. titration-point / spectrum index of each
             residual). When given, one wild multiplier is drawn per group so residuals of
             the same spectrum keep their mutual correlation (block wild bootstrap).
    n_linear : number of linear (variable-projection) parameters, used together with p
             for the global HC1-type sqrt(n/(n-p_total)) correction of the residuals.
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
    if leverage_correction:
        r = r * hc1_rescale_factor(r.size, k_hat.size + int(n_linear or 0))
    inv = None
    n_groups = 0
    if groups is not None:
        groups = _onp.asarray(groups).ravel()
        if groups.size != r.size:
            raise ValueError("groups must have one label per residual.")
        _, inv = _onp.unique(groups, return_inverse=True)
        n_groups = int(inv.max()) + 1 if inv.size else 0

    samples = _onp.zeros((B, k_hat.size), dtype=float)
    for b in range(B):
        if inv is not None:
            v = wild_multipliers(rng, n_groups, kind=wild)[inv]
        else:
            v = _wild_weights(rng, r.size, wild=wild)
        eps = r * v
        g = J @ eps
        dk = G_inv @ g
        samples[b, :] = k_hat + dk

    # The pseudo-inverse gives zero movement along null-space directions, which would
    # look like a (spuriously small) uncertainty. Non-identifiable parameters -> NaN.
    nonid = nonidentifiable_parameters(J)
    samples[:, nonid] = _onp.nan
    return {
        "samples": samples,
        "nonidentifiable": nonid,
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
    weights=None,
    group_map=None,
    param_transform=None,
    wild_block="row",
    leverage_correction=True,
    fixed_mask=None,
):
    """
    One-step LM bootstrap (spectroscopy): dataset perturbation + recompute J,r at k_hat + 1 LM step.

    Y is (n_lambda x m_points); with ``wild_block="row"`` one multiplier is drawn per
    titration point (column of Y) so the spectral correlation of each spectrum is kept.
    """
    k_hat = _onp.asarray(k_hat, dtype=float).ravel()
    B = int(B)
    if B <= 0:
        raise ValueError("B must be > 0.")

    res = _FixedPointConcentrations(res)  # same k_hat for every replicate
    metrics0 = compute_errors_spectro_varpro(
        k_hat,
        res,
        Y,
        modelo,
        nas,
        rcond=rcond,
        use_projector=use_projector,
        weights=weights,
        group_map=group_map,
        param_transform=param_transform,
        fixed_mask=fixed_mask,
    )
    free_idx = _onp.asarray(metrics0["free_idx"], dtype=int)
    yfit = _onp.asarray(metrics0["yfit"], dtype=float)
    Y_obs = _onp.asarray(Y, dtype=float)
    if yfit.shape != Y_obs.shape:
        raise ValueError(f"yfit shape {yfit.shape} does not match Y {Y_obs.shape}.")

    R = yfit - Y_obs
    if leverage_correction:
        n_lin = int(metrics0.get("n_linear", 0) or 0)
        R = R * hc1_rescale_factor(int(metrics0["nobs"]), free_idx.size + n_lin)
    rng = _onp.random.default_rng(seed)
    samples = _onp.tile(k_hat, (B, 1)).astype(float)

    for b in range(B):
        # rows of R.T are titration points
        v = block_wild_multipliers(rng, R.T.shape, kind=wild, block=wild_block).T
        Y_star = yfit - R * v

        metrics_star = compute_errors_spectro_varpro(
            k_hat,
            res,
            Y_star,
            modelo,
            nas,
            rcond=rcond,
            use_projector=use_projector,
            weights=weights,
            group_map=group_map,
            param_transform=param_transform,
        )
        if free_idx.size == 0:
            continue
        J = _onp.asarray(metrics_star["J"], dtype=float)[free_idx, :]
        r_star = _onp.asarray(metrics_star["r"], dtype=float).ravel()

        # fixed parameters never move
        samples[b, free_idx] = k_hat[free_idx] + lm_damped_step(J, r_star, lam=lam, rcond=rcond)

    nonid = _onp.asarray(metrics0["nonidentifiable"], dtype=bool)
    samples[:, nonid] = _onp.nan
    return {
        "samples": samples,
        "nonidentifiable": nonid,
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
    stoichiometry=None,
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
    wild_block="row",
    leverage_correction=True,
):
    """
    One-step LM bootstrap (NMR): dataset perturbation + recompute J,r at k_hat + 1 LM step.

    dq is (m_points x n_signals); ``wild_block="row"`` shares the multiplier among all
    signals of the same titration point.
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

    res = _FixedPointConcentrations(res)  # same k_hat for every replicate
    metrics0 = compute_errors_nmr_varpro(
        k_hat, res, dq_obs, D_cols, modelo, nas, stoichiometry=stoichiometry,
        mask=mask, fixed_mask=fixed_mask, rcond=rcond,
        rcond_cov=rcond_cov if rcond_cov is not None else rcond,
        ridge=ridge, ridge_cov=ridge_cov, use_projector=use_projector,
    )
    R = dq_fit - dq_obs
    R = _onp.where(mask, R, 0.0)
    if leverage_correction:
        n_obs = int(_onp.count_nonzero(mask))
        n_lin = int(metrics0.get("n_linear", 0) or 0)
        R = R * hc1_rescale_factor(n_obs, free_idx.size + n_lin)
    rng = _onp.random.default_rng(seed)
    samples = _onp.zeros((B, k_hat.size), dtype=float)

    for b in range(B):
        v = block_wild_multipliers(rng, R.shape, kind=wild, block=wild_block)
        dq_star = dq_fit - R * v
        dq_star[~mask] = dq_obs[~mask]

        metrics_star = compute_errors_nmr_varpro(
            k_hat,
            res,
            dq_star,
            D_cols,
            modelo,
            nas,
            stoichiometry=stoichiometry,
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

        dk = lm_damped_step(J, r_star, lam=lam, rcond=rcond)

        k1 = k_hat.copy()
        k1[free_idx] = k1[free_idx] + dk
        samples[b, :] = k1

    if free_idx.size:
        J_original = _onp.asarray(metrics0["J"], dtype=float)[free_idx, :]
        nonid_free = nonidentifiable_parameters(J_original)
        samples[:, free_idx[nonid_free]] = _onp.nan
    return {
        "samples": samples,
    }
