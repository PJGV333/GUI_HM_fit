# errors.py
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
