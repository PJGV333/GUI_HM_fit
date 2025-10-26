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

        # dentro de LevenbergMarquardt.concentraciones(...)

        lam = lam0
        max_backtrack = 5
        max_step = 5.0  # tamaño máximo de ||Δu|| para evitar saltos enormes

        for i in range(n_reac):
            c0 = np.clip(ctot[i, :n_comp], 1e-18, None)
            u = np.log(c0)  # (n_comp,)

            c_spec = self._c_spec_from_u(u, K, modelo)
            d = ctot[i] - self._mass_balance(c_spec, modelo)
            prev = np.linalg.norm(d)

            it = 0
            while it < max_iter:
                # Hessiano Gauss–Newton en u (simétrico PSD):
                # H[k,h] = sum_j ν_{kj} ν_{hj} c_spec_j
                H = np.empty((n_comp, n_comp), dtype=dtype_out)
                for k_ in range(n_comp):
                    vk = mt[:, k_]
                    for h_ in range(n_comp):
                        vh = mt[:, h_]
                        H[k_, h_] = np.sum(vk * vh * c_spec)

                # LM clásico: H_λ = H + λ diag(H)  (no λI)
                diagH = np.real(np.diag(H))
                # ridge adaptativo por mala condición
                alpha = 1e-12 * np.maximum(1.0, np.max(diagH))
                D = np.diag(diagH + alpha)
                H_d = H + lam * D

                # Δu = H_λ^+ d  (complex-step safe)
                delta_u = pinv_cs(H_d) @ d

                # control de paso
                du_norm = np.linalg.norm(np.real(delta_u))
                if du_norm > max_step:
                    delta_u = delta_u * (max_step / du_norm)

                # backtracking si no mejora
                accepted = False
                u_trial = u
                lam_local = lam
                for _ in range(max_backtrack):
                    u_trial = u + np.real(delta_u)
                    c_spec_trial = self._c_spec_from_u(u_trial, K, modelo)
                    d_trial = ctot[i] - self._mass_balance(c_spec_trial, modelo)
                    cur = np.linalg.norm(d_trial)
                    if cur < prev:
                        accepted = True
                        break
                    # si no mejora, sube λ y vuelve a intentar con paso más chico
                    lam_local = min(lam_local * lam_up, 1e12)
                    H_d = H + lam_local * D
                    delta_u = pinv_cs(H_d) @ d
                    du_norm = np.linalg.norm(np.real(delta_u))
                    if du_norm > max_step:
                        delta_u = delta_u * (max_step / du_norm)

                if accepted:
                    u = u_trial
                    c_spec = c_spec_trial
                    d = d_trial
                    prev = cur
                    lam = max(lam_local * lam_down, 1e-12)
                    if prev < tol:
                        break
                else:
                    # Fallback: paso tipo Newton (resolver H Δu = d) si LM no acepta
                    try:
                        delta_u_nr = pinv_cs(H) @ d
                        u_trial = u + np.real(delta_u_nr)
                        c_spec_trial = self._c_spec_from_u(u_trial, K, modelo)
                        d_trial = ctot[i] - self._mass_balance(c_spec_trial, modelo)
                        cur = np.linalg.norm(d_trial)
                        if cur < prev:
                            u, c_spec, d, prev = u_trial, c_spec_trial, d_trial, cur
                            lam = min(lam * lam_down, 1e-3)
                            if prev < tol:
                                break
                        else:
                            lam = min(lam * lam_up, 1e12)
                    except Exception:
                        lam = min(lam * lam_up, 1e12)

                it += 1

            c_calculada[i] = c_spec


        C = np.delete(c_calculada, self.nas, axis=1)
        return C, c_calculada