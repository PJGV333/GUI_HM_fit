# LM_conc_algoritm.py
# ---------------------------------------------------------
# Levenberg–Marquardt para resolver el balance de masa (c de componentes)
# Internamente usa NumPy (onp) para mutabilidad; al salir convierte al backend (np).

import numpy as onp
from np_backend import xp as np  # backend conmutable (JAX/NumPy)

# ---------------------------------------------------------
# Pseudoinversa robusta en NumPy real (fallback si el sistema es mal condicionado)
def pinv_cs(A, rcond=1e-12):
    A = onp.asarray(A)
    try:
        return onp.linalg.pinv(A, rcond=rcond)
    except onp.linalg.LinAlgError:
        ATA = A.T @ A + (rcond if onp.isscalar(rcond) else 1e-12) * onp.eye(A.shape[1], dtype=A.dtype)
        return onp.linalg.solve(ATA, A.T)
# ---------------------------------------------------------


class LevenbergMarquardt:
    """
    Interfaz esperada por la GUI:
        res = LevenbergMarquardt(C_T, modelo, nas, model_sett, ...)

    Donde:
      - C_T      : (n_reacciones, n_componentes), totales por corrida
      - modelo   : (n_componentes, nspec), matriz estequiométrica
      - nas      : índices de especies no absorbentes (columnas a eliminar)
      - model_sett : 'Free' | 'Step by step' | 'Non-cooperative' (convención existente)
    """

    def __init__(self, C_T, modelo, nas, model_sett,
                 tol=1e-10, max_iter=200,
                 lam0=1e-2, lam_up=10.0, lam_down=0.2,
                 max_step=2.0, max_backtrack=8):
        self.C_T = onp.asarray(C_T, dtype=float)            # (n_reac, n_comp)
        self.modelo = onp.asarray(modelo, dtype=float)      # (n_comp, nspec)
        self.nas = onp.asarray(nas if nas is not None else [], dtype=int)
        #self.model_sett = model_sett
        self.model_sett = model_sett or "Free"

        # shapes
        self.n_componentes, self.nspec = self.modelo.shape
        self.n_reacciones = self.C_T.shape[0]
        self.mt = self.modelo.T

        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.lam0 = float(lam0)
        self.lam_up = float(lam_up)
        self.lam_down = float(lam_down)
        self.max_step = float(max_step)
        self.max_backtrack = int(max_backtrack)

        self.n_reac, self.n_comp = self.C_T.shape
        self.nspec = self.modelo.shape[1]
        self.mt = self.modelo.T  # (nspec, n_comp)

    # ----- transformaciones K (siguiendo tu convención actual) -----
    def step_by_step(self, K_log):
        return onp.cumsum(onp.asarray(K_log, dtype=float))

    def non_coop(self, K_log):
        """
        No cooperativo 1:2 colapsado:
        - Si llega solo K1, genera internamente [K1, K1 - log10(4)] en log10 y luego acumula (β).
        - Si llegan K1,K2 “crudos”, aplica corrección al segundo paso y acumula.
        """
        K_log = onp.asarray(K_log, dtype=float)
        start = self.n_componentes
        n_complex = K_log.size - start

        if n_complex == 1 and (self.nspec - self.n_componentes) == 2:
            K1 = K_log[start]
            K_log = onp.concatenate((K_log, onp.array([K1 - onp.log10(4.0)], dtype=float)))
        elif n_complex >= 2:
            K_log[start + 1] -= onp.log10(4.0)

        return onp.cumsum(K_log)


    def _prepare_K_numeric(self, k_in):
        """
        Free / Step by step: entran n_complex constantes (log10).
        Non-cooperative: puede entrar n_complex - 1 (colapsado) o n_complex.
        Hace pre-padding de 0's para componentes y devuelve tamaño nspec.
        """
        k_in = onp.asarray(k_in, dtype=float).ravel()
        n_comp    = self.n_componentes
        nspec     = self.nspec
        n_complex = nspec - n_comp

        pre_ko   = onp.zeros(n_comp, dtype=float)       # log10(1)=0 para componentes
        K_log_in = onp.concatenate((pre_ko, k_in))      # aún en log10

        ms = self.model_sett
        if ms == "Free":
            if k_in.size != n_complex:
                raise ValueError(f"Para 'Free' se esperan {n_complex} constantes, llegaron {k_in.size}.")
            K_log_eff = K_log_in

        elif ms == "Step by step":
            if k_in.size != n_complex:
                raise ValueError(f"Para 'Step by step' se esperan {n_complex} constantes, llegaron {k_in.size}.")
            K_log_eff = self.step_by_step(K_log_in)

        elif ms in ("Non-cooperative", "Statistical"):
            if not (k_in.size == n_complex - 1 or k_in.size == n_complex):
                raise ValueError(
                    f"Para 'Non-cooperative' se esperan {n_complex-1} o {n_complex} constantes, "
                    f"llegaron {k_in.size}."
                )
            K_log_eff = self.non_coop(K_log_in)

        else:
            K_log_eff = K_log_in  # fallback

        K_num = onp.power(10.0, K_log_eff)
        return onp.asarray(K_num, dtype=onp.result_type(K_num.dtype, onp.float64))


    # ----- helpers -----
    def _c_spec_from_u(self, u, K):
        # c_spec = K * exp(M^T @ u)
        mt_u = self.mt @ u
        return K * onp.exp(mt_u)

    def _mass_balance(self, c_spec):
        # modelo: (n_comp, nspec) × c_spec: (nspec,) → (n_comp,)
        return self.modelo @ c_spec

    def _jac_u(self, c_spec):
        # J = M @ diag(c_spec) @ M^T ; forma eficiente:
        # mt: (nspec, n_comp)
        return self.mt.T @ (c_spec[:, None] * self.mt)  # (n_comp, n_comp)

    # ---------- LM principal (NumPy real) ----------
    def concentraciones(self, K, tol=None, max_iter=None,
                        lam0=None, lam_up=None, lam_down=None,
                        max_step=None, max_backtrack=None):
        # Overwrites opcionales
        tol = self.tol if tol is None else float(tol)
        max_iter = self.max_iter if max_iter is None else int(max_iter)
        lam0 = self.lam0 if lam0 is None else float(lam0)
        lam_up = self.lam_up if lam_up is None else float(lam_up)
        lam_down = self.lam_down if lam_down is None else float(lam_down)
        max_step = self.max_step if max_step is None else float(max_step)
        max_backtrack = self.max_backtrack if max_backtrack is None else int(max_backtrack)

        ridge = 1e-12
        c_calculada = onp.zeros((self.n_reac, self.nspec), dtype=float)

        K_num = self._prepare_K_numeric(K)

        for i in range(self.n_reac):
            # u = ln(c) inicial
            c0 = onp.clip(self.C_T[i, :self.n_comp], 1e-18, None)
            u = onp.log(c0)

            # estado inicial
            c_spec = self._c_spec_from_u(u, K_num)
            d = self.C_T[i] - self._mass_balance(c_spec)
            prev = float(onp.linalg.norm(d, ord=2))
            lam = float(lam0)

            it = 0
            while prev > tol and it < max_iter:
                J = self._jac_u(c_spec)                         # (n_comp, n_comp)
                Jd = J + (lam + ridge) * onp.eye(self.n_comp)   # damping

                # Precondicionamiento Jacobi
                diagJ = onp.clip(onp.diag(Jd), 1e-30, None)
                P = 1.0 / onp.sqrt(diagJ)
                PJP = (P[:, None] * Jd) * P[None, :]
                Pd  = P * d

                # Resolver Δz y llevar a Δu
                try:
                    delta_z = onp.linalg.solve(PJP, Pd)
                except onp.linalg.LinAlgError:
                    delta_z = pinv_cs(PJP) @ Pd
                delta_u = P * delta_z

                # Limitar paso
                ndu = float(onp.linalg.norm(delta_u))
                if ndu > max_step:
                    delta_u *= (max_step / max(ndu, 1e-18))

                # Backtracking (Armijo)
                alpha = 1.0
                accepted = False
                for _ in range(max_backtrack):
                    u_trial = u + alpha * delta_u
                    c_trial = self._c_spec_from_u(u_trial, K_num)
                    d_trial = self.C_T[i] - self._mass_balance(c_trial)
                    cur = float(onp.linalg.norm(d_trial, ord=2))
                    if cur < prev:
                        accepted = True
                        break
                    alpha *= 0.5

                if accepted:
                    u = u_trial
                    c_spec = c_trial
                    d = d_trial
                    prev = cur
                    lam = max(lam * lam_down, 1e-12)
                    it += 1
                    continue

                # Fallback: Newton amortiguado (mismo precondicionamiento)
                PJ = (P[:, None] * J) * P[None, :]
                try:
                    delta_zN = onp.linalg.solve(PJ + ridge * onp.eye(self.n_comp), Pd)
                except onp.linalg.LinAlgError:
                    delta_zN = pinv_cs(PJ + ridge * onp.eye(self.n_comp)) @ Pd
                delta_uN = P * delta_zN

                nduN = float(onp.linalg.norm(delta_uN))
                if nduN > max_step:
                    delta_uN *= (max_step / max(nduN, 1e-18))

                alpha = 1.0
                acceptedN = False
                for _ in range(max_backtrack):
                    u_trial = u + alpha * delta_uN
                    c_trial = self._c_spec_from_u(u_trial, K_num)
                    d_trial = self.C_T[i] - self._mass_balance(c_trial)
                    cur = float(onp.linalg.norm(d_trial, ord=2))
                    if cur < prev:
                        acceptedN = True
                        break
                    alpha *= 0.5

                if acceptedN:
                    u = u_trial
                    c_spec = c_trial
                    d = d_trial
                    prev = cur
                    lam = min(lam * lam_up, 1e12)
                    it += 1
                else:
                    # Reinicio seguro para evitar estancamientos
                    u = onp.log(onp.clip(self.C_T[i, :self.n_comp], 1e-16, None))
                    c_spec = self._c_spec_from_u(u, K_num)
                    d = self.C_T[i] - self._mass_balance(c_spec)
                    prev = float(onp.linalg.norm(d, ord=2))
                    lam = lam0 * 10.0
                    it += 1

            c_calculada[i, :] = c_spec

        # Eliminar especies no absorbentes (NumPy real)
        if self.nas.size > 0:
            C_np = onp.delete(c_calculada, self.nas, axis=1)
        else:
            C_np = c_calculada

        # Regresar en el backend (JAX/NumPy)
        return np.asarray(C_np, dtype=float), np.asarray(c_calculada, dtype=float)
