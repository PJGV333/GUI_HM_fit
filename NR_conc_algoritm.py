from np_backend import xp as np, jit, jacrev, vmap, lax


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

class NewtonRaphson:
    def __init__(self, C_T, modelo, nas, model_sett):
        self.C_T = C_T
        self.modelo = modelo
        self.nas = nas
        self.model_sett = model_sett

    def non_coop(self, K):
            K_0 = np.array([K[2] - np.log10(4)])
            K_1 = np.concatenate((K, K_0))
            K_2 = np.cumsum(K_1)
            return K_2

    def step_by_step(self, K):
        K_2 = np.cumsum(K)
        return K_2

    #def concentraciones(self, K, max_iter=1000, tol=1e-10):
    #    ctot = np.array(self.C_T)
    #    n_reacciones, n_componentes = ctot.shape
    #    pre_ko = np.zeros(n_componentes)
    #    K = np.concatenate((pre_ko, K))
#
    #    if self.model_sett == "Free":
    #        K = 10**K
    #    elif self.model_sett == "Step by step":
    #        K = self.step_by_step(K)
    #        K = 10**K
    #    elif self.model_sett == "Non-cooperative":
    #        K = self.non_coop(K)
    #        K = 10**K
#
    #    nspec = len(K)
#
    #    def calcular_concentraciones(ctot_i, c_guess):
    #        c_spec = np.prod(np.power(np.tile(c_guess, (nspec, 1)).T, self.modelo), axis=0) * K
    #        c_tot_cal = np.sum(self.modelo * np.tile(c_spec, (n_componentes, 1)), axis=1)
    #        d = ctot_i - c_tot_cal
#
    #        J = np.empty((n_componentes, n_componentes))
    #        for j in range(n_componentes):
    #            for h in range(n_componentes):
    #                J[j, h] = np.sum(self.modelo.T[:, j] * self.modelo.T[:, h] * c_spec)
#
    #        delta_c = d @ np.linalg.pinv(J) @ np.diagflat(c_guess)
#
    #        c_guess += delta_c
    #        return c_guess, np.linalg.norm(d), c_spec
#
    #    c_calculada = np.zeros((n_reacciones, nspec))
    #    for i in range(n_reacciones):
    #        c_guess = np.ones(n_componentes) * 1e-10
    #        # Asignación dinámica para c_guess basada en el número de componentes
    #        c_guess[:n_componentes] = ctot[i, :n_componentes]
    #        dif = tol + 1
    #        it = 0
    #        while dif > tol and it < max_iter:
    #            c_guess, delta_c_norm_sq, c_spec = calcular_concentraciones(ctot[i], c_guess)
    #            dif = delta_c_norm_sq
    #            it += 1
    #        c_calculada[i] = c_spec
#
    #    C = np.delete(c_calculada, self.nas, axis=1)
    #    return C, c_calculada


# =============================================================================
#     def concentraciones(self, K, max_iter=1000, tol=1e-10):
#         ctot = np.array(self.C_T)
#         n_reacciones, n_componentes = ctot.shape
# 
#         # <<< nuevo: elegir dtype según K >>>
#         dtype = np.complex128 if np.iscomplexobj(K) else float
# 
#         pre_ko = np.zeros(n_componentes, dtype=dtype)
#         K = np.concatenate((pre_ko, K))
# 
#         if self.model_sett == "Free":
#             K = 10**K
#         elif self.model_sett == "Step by step":
#             K = self.step_by_step(K); K = 10**K
#         elif self.model_sett == "Non-cooperative":
#             K = self.non_coop(K); K = 10**K
# 
#         nspec = len(K)
# 
#         def calcular_concentraciones(ctot_i, c_guess):
#             # Asegura que c_guess sea del dtype correcto
#             c_guess = np.asarray(c_guess, dtype=dtype)
# 
#             c_spec = np.prod(np.power(np.tile(c_guess, (nspec, 1)).T, self.modelo), axis=0) * K
#             c_tot_cal = np.sum(self.modelo * np.tile(c_spec, (n_componentes, 1)), axis=1)
#             d = ctot_i - c_tot_cal
# 
#             # Jacobiano en el dtype correcto
#             J = np.empty((n_componentes, n_componentes), dtype=dtype)
#             for j in range(n_componentes):
#                 for h in range(n_componentes):
#                     J[j, h] = np.sum(self.modelo.T[:, j] * self.modelo.T[:, h] * c_spec)
# 
#             delta_c = d @ pinv_cs(J) @ np.diagflat(c_guess)
#             c_guess += delta_c
#             return c_guess, np.linalg.norm(d), c_spec
# 
#         c_calculada = np.zeros((n_reacciones, nspec), dtype=dtype)
#         for i in range(n_reacciones):
#             c_guess = np.ones(n_componentes, dtype=dtype) * 1e-10
#             c_guess[:n_componentes] = ctot[i, :n_componentes]
#             dif = tol + 1; it = 0
#             while dif > tol and it < max_iter:
#                 c_guess, delta_c_norm_sq, c_spec = calcular_concentraciones(ctot[i], c_guess)
#                 dif = delta_c_norm_sq; it += 1
#             c_calculada[i] = c_spec
# 
#         C = np.delete(c_calculada, self.nas, axis=1)
#         return C, c_calculada
# 
# =============================================================================

    def concentraciones(self, K, max_iter=1000, tol=1e-10):
        
        ctot = np.array(self.C_T)
        n_reacciones, n_componentes = ctot.shape
    
        # elegir dtype según K
        dtype = np.complex128 if np.iscomplexobj(K) else float
    
        pre_ko = np.zeros(n_componentes, dtype=dtype)
        K = np.concatenate((pre_ko, K))
    
        if self.model_sett == "Free":
            K = 10**K
        elif self.model_sett == "Step by step":
            K = self.step_by_step(K); K = 10**K
        elif self.model_sett == "Non-cooperative":
            K = self.non_coop(K); K = 10**K
    
        nspec = len(K)
    
        # fijar referencias a M y M^T una sola vez
        modelo = np.asarray(self.modelo, dtype=float)      # (n_componentes, nspec)
        mt = modelo.T                                      # (nspec, n_componentes)
    
        def calcular_concentraciones(ctot_i, c_guess):
            # Asegura que c_guess sea del dtype correcto
            c_guess = np.asarray(c_guess, dtype=dtype)
    
            # especies: ∏_k c_k^{ν_{k,j}} * K_j
            c_spec = np.prod(np.power(np.tile(c_guess, (nspec, 1)).T, modelo), axis=0) * K
    
            # balance de masa: M @ c_spec
            c_tot_cal = modelo @ c_spec
    
            d = ctot_i - c_tot_cal
    
            # --- Jacobiano vectorizado (reemplaza el doble for) ---
            # J = M^T diag(c_spec) M, con M = mt (nspec × n_componentes)
            J = mt.T @ (c_spec[:, None] * mt)             # (n_componentes, n_componentes)
    
            # Paso NR en c (tu forma original): delta_c = (d @ pinv(J)) .* c_guess
            delta_u = d @ pinv_cs(J)                      # (n_componentes,)
            delta_c = delta_u * c_guess                   # evita np.diagflat(c_guess)
    
            c_guess += delta_c
            return c_guess, np.linalg.norm(d), c_spec
    
        c_calculada = np.zeros((n_reacciones, nspec), dtype=dtype)
        for i in range(n_reacciones):
            c_guess = np.ones(n_componentes, dtype=dtype) * 1e-10
            c_guess[:n_componentes] = ctot[i, :n_componentes]
            dif = tol + 1; it = 0
            while dif > tol and it < max_iter:
                c_guess, delta_c_norm_sq, c_spec = calcular_concentraciones(ctot[i], c_guess)
                dif = delta_c_norm_sq; it += 1
            c_calculada[i] = c_spec
    
        C = np.delete(c_calculada, self.nas, axis=1)
        return C, c_calculada
