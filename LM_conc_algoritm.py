from np_backend import xp as np, jit, jacrev, vmap, lax
from scipy.linalg import cho_factor, cho_solve, LinAlgError

# ---------- pseudo-inversa complex-step-safe (sin conjugado) ----------
import numpy as np

def pinv_cs(A, rcond=1e-12):
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


class LevenbergMarquardt:
    """
    LM manual en espacio logarítmico u = ln(c):
      c = exp(u) >= 0
      c_spec = K * exp(modelo^T @ u)
      MB(u)   = modelo @ c_spec
      J_u[k,h] = sum_j ν_{kj} ν_{hj} c_spec_j
      Δu = (J_u + λI)^+ d,  u <- u + Re(Δu)
    * Complex-step-safe: K puede ser complejo y se propaga sin conjugados.
    * Devuelve (C, c_calculada) como NR.
    """
    def __init__(self, C_T, modelo, nas, model_sett):
        self.C_T = np.array(C_T)
        self.modelo = np.array(modelo, dtype=float)   # (n_comp, nspec)
        self.nas = nas
        self.model_sett = model_sett

    # ----- transformaciones de K -----
    def non_coop(self, K):
        K = np.asarray(K)
        if K.size < 3:
            return np.cumsum(K)
        K_0 = np.array([K[2] - np.log10(4)], dtype=K.dtype)
        K_1 = np.concatenate((K, K_0))
        return np.cumsum(K_1)

    def step_by_step(self, K):
        return np.cumsum(K)

    def _prepare_K_numeric(self, K):
        K = np.asarray(K)
        n_comp = self.C_T.shape[1]
        pre_ko = np.zeros(n_comp, dtype=K.dtype)
        K = np.concatenate((pre_ko, K))
        if self.model_sett == "Free":
            K = 10.0 ** K
        elif self.model_sett == "Step by step":
            K = 10.0 ** self.step_by_step(K)
        elif self.model_sett == "Non-cooperative":
            K = 10.0 ** self.non_coop(K)
        else:
            K = 10.0 ** K
        dtype = np.complex128 if np.iscomplexobj(K) else np.float64
        return np.asarray(K, dtype=dtype)
    
    # ----- helpers -----
    def _c_spec_from_u(self, u, K, modelo):
        """
        u: (n_comp,) real
        K: (nspec,) float o complex
        modelo: (n_comp, nspec)
        """
        mt_u = modelo.T @ u                  # (nspec,)
        c_spec = K * np.exp(mt_u)            # (nspec,)
        return c_spec

    def _mass_balance(self, c_spec, modelo):
        # (n_comp,), no conj
        return modelo @ c_spec

    # ----- LM sobre u = ln(c) -----
    def concentraciones(self, K, max_iter=200, tol=1e-10,
                    lam0=1e-2, lam_up=10.0, lam_down=0.2,
                    max_step=2.0, max_backtrack=8):
        """
        LM robusto con fallback a Newton amortiguado (búsqueda de línea).
        Válido para RMN y Espectroscopía. Misma interfaz que el original.
        """
        import numpy as np
        from errors import pinv_cs  # wrapper a np.linalg.pinv
    
        # --- datos base ---
        ctot = np.array(self.C_T, dtype=float)           # (n_reac, n_comp)
        n_reac, n_comp = ctot.shape
    
        K = self._prepare_K_numeric(K)
        modelo = np.asarray(self.modelo, dtype=float)    # (n_comp, nspec)  ← tu convención
        mt = modelo.T                                    # (nspec, n_comp)
        nspec = len(K)
    
        ridge = 1e-12
        c_calculada = np.zeros((n_reac, nspec), dtype=float)
    
        # utilidades (¡ojo a la orientación!)
        def mass_balance(c_spec):
            # modelo: (n_comp, nspec)  ×  c_spec: (nspec,)  →  (n_comp,)
            return modelo @ c_spec
    
        def jac_u(c_spec):
            # J = M^T diag(c_spec) M con M = mt (nspec × n_comp)
            # mt: (nspec, n_comp)  →  J: (n_comp, n_comp)
            return mt.T @ (c_spec[:, None] * mt)
    
        def c_from(u):
            # _c_spec_from_u espera 'modelo' con forma (n_comp, nspec)
            return self._c_spec_from_u(u, K, modelo)
    
        for i in range(n_reac):
            # u inicial
            c0 = np.clip(ctot[i, :n_comp], 1e-18, None)
            u = np.log(c0)
    
            # residuo inicial
            c_spec = c_from(u)
            d = ctot[i] - mass_balance(c_spec)           # (n_comp,)
            prev = float(np.linalg.norm(d))
    
            lam = float(lam0)
            it = 0
            while it < max_iter:
                # Jacobiano y sistema LM
                J = jac_u(c_spec)                        # (n_comp, n_comp)
                J_d = J + (lam + ridge) * np.eye(n_comp)
    
                # Precondicionamiento Jacobi
                diagJ = np.clip(np.diag(J_d), 1e-30, None)
                P = 1.0 / np.sqrt(diagJ)
                PJP = (P[:, None] * J_d) * P[None, :]
                Pd  = P * d
    
                # Resolver (PJP)Δz = Pd ; Δu = PΔz
                try:
                    delta_z = np.linalg.solve(PJP, Pd)
                except np.linalg.LinAlgError:
                    delta_z = pinv_cs(PJP) @ Pd
                delta_u = P * delta_z
    
                # limitar paso
                ndu = float(np.linalg.norm(delta_u))
                if ndu > max_step:
                    delta_u *= (max_step / max(ndu, 1e-18))
    
                # backtracking (Armijo) sobre el paso actual
                alpha = 1.0
                ok = False
                for _ in range(max_backtrack):
                    u_trial = u + alpha * delta_u
                    c_trial = c_from(u_trial)
                    d_trial = ctot[i] - mass_balance(c_trial)
                    cur = float(np.linalg.norm(d_trial))
                    if cur < prev:
                        ok = True
                        break
                    alpha *= 0.5
    
                if ok:
                    # aceptar
                    u = u_trial
                    c_spec = c_trial
                    d = d_trial
                    prev = cur
                    lam = max(lam * lam_down, 1e-12)
                    if prev < tol:
                        break
                    it += 1
                    continue
                else:
                    # FAILOVER: Newton amortiguado con la misma búsqueda de línea
                    PJ = (P[:, None] * J) * P[None, :]
                    try:
                        delta_zN = np.linalg.solve(PJ + ridge*np.eye(n_comp), Pd)
                    except np.linalg.LinAlgError:
                        delta_zN = pinv_cs(PJ + ridge*np.eye(n_comp)) @ Pd
                    delta_uN = P * delta_zN
    
                    nduN = float(np.linalg.norm(delta_uN))
                    if nduN > max_step:
                        delta_uN *= (max_step / max(nduN, 1e-18))
    
                    alpha = 1.0
                    accepted_newton = False
                    for _ in range(max_backtrack):
                        u_trial = u + alpha * delta_uN
                        c_trial = c_from(u_trial)
                        d_trial = ctot[i] - mass_balance(c_trial)
                        cur = float(np.linalg.norm(d_trial))
                        if cur < prev:
                            accepted_newton = True
                            break
                        alpha *= 0.5
    
                    if accepted_newton:
                        u = u_trial
                        c_spec = c_trial
                        d = d_trial
                        prev = cur
                        lam = min(lam * lam_up, 1e12)  # tras Newton, amortigua más
                        if prev < tol:
                            break
                        it += 1
                        continue
                    else:
                        # reinicio seguro de u para evitar atascos numéricos
                        u = np.log(np.clip(ctot[i, :n_comp], 1e-16, None))
                        c_spec = c_from(u)
                        d = ctot[i] - mass_balance(c_spec)
                        prev = float(np.linalg.norm(d))
                        lam = lam0 * 10.0
                        it += 1
                        continue
    
            c_calculada[i] = c_spec
    
        C = np.delete(c_calculada, self.nas, axis=1)
        return C, c_calculada
    
            