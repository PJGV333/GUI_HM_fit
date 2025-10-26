# errors.py
import numpy as np
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