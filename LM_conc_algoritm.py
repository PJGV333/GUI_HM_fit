import numpy as np

# ---------- pseudo-inversa complex-step-safe (sin conjugado) ----------
def pinv_cs(A, rcond=1e-12):
    A = np.asarray(A)
    if not np.iscomplexobj(A):
        return np.linalg.pinv(A, rcond=rcond)
    m, n = A.shape
    Ar = np.block([[A.real, -A.imag],
                   [A.imag,  A.real]])           # (2m x 2n)
    Pr = np.linalg.pinv(Ar, rcond=rcond)         # (2n x 2m)
    X = Pr[:n,    :m]
    Y = Pr[n:2*n, :m]
    return X + 1j*Y

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
                         lam0=1e-2, lam_up=10.0, lam_down=0.2):
        ctot = np.array(self.C_T, dtype=float)
        n_reac, n_comp = ctot.shape

        K = self._prepare_K_numeric(K)
        modelo = np.asarray(self.modelo, dtype=float)     # (n_comp, nspec)
        mt = modelo.T
        nspec = len(K)

        dtype_out = K.dtype
        c_calculada = np.zeros((n_reac, nspec), dtype=dtype_out)

        lam = lam0
        for i in range(n_reac):
            # u0 = ln(c_guess) con clip suave
            c0 = np.clip(ctot[i, :n_comp], 1e-18, None)
            u = np.log(c0)                                # (n_comp,)

            # residuo inicial
            c_spec = self._c_spec_from_u(u, K, modelo)
            d = ctot[i] - self._mass_balance(c_spec, modelo)
            prev = np.linalg.norm(d)

            it = 0
            while it < max_iter:
                # Jacobiano en u: J_u[k,h] = sum_j ν_{kj} ν_{hj} c_spec_j
                # Usamos mt = modelo^T (nspec x n_comp)
                # Construcción explícita (tamaños pequeños)
                J = np.empty((n_comp, n_comp), dtype=dtype_out)
                for k in range(n_comp):
                    for h in range(n_comp):
                        J[k, h] = np.sum(mt[:, k] * mt[:, h] * c_spec)

                J_d = J + lam * np.eye(n_comp, dtype=dtype_out)
                delta_u = pinv_cs(J_d) @ d                 # (n_comp,)

                # actualiza u (solo la parte real), recalcula
                u_trial = u + np.real(delta_u)
                c_spec_trial = self._c_spec_from_u(u_trial, K, modelo)
                d_trial = ctot[i] - self._mass_balance(c_spec_trial, modelo)
                cur = np.linalg.norm(d_trial)

                if cur < prev:
                    u = u_trial
                    c_spec = c_spec_trial
                    d = d_trial
                    prev = cur
                    lam = max(lam * lam_down, 1e-12)
                    if prev < tol:
                        break
                else:
                    lam = min(lam * lam_up, 1e12)
                it += 1

            c_calculada[i] = c_spec

        C = np.delete(c_calculada, self.nas, axis=1)
        return C, c_calculada
