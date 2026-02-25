# SPDX-License-Identifier: GPL-3.0-or-later
# LM_conc_algoritm.py
# ---------------------------------------------------------
# Levenberg–Marquardt para resolver el balance de masa (c de componentes)
# Internamente usa NumPy (onp) para mutabilidad; al salir convierte al backend (np).

import numpy as onp
from typing import Optional
from ..utils.np_backend import xp as np
from ..utils.noncoop_utils import infer_noncoop_series
from ..graph.hmgraph import HMGraph

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
                 graph: Optional[HMGraph] = None,
                 tol=1e-10, max_iter=200,
                 lam0=1e-2, lam_up=10.0, lam_down=0.2,
                 max_step=2.0, max_backtrack=8):
        self.graph = graph
        self._graph_revision = None

        self.C_T = onp.asarray(C_T, dtype=float)            # (n_reac, n_comp)

        if self.graph is not None:
            modelo_g, nas_g = self.graph.compile()
            self._graph_revision = self.graph.revision
            self.modelo = onp.asarray(modelo_g, dtype=float)
            self.nas = onp.asarray(nas_g if nas is None else nas, dtype=int)
        else:
            self.modelo = onp.asarray(modelo, dtype=float)  # (n_comp, nspec)
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

    def _recompile_graph_if_needed(self) -> None:
        if self.graph is None:
            return
        rev = self.graph.revision
        if self._graph_revision != rev:
            modelo_g, nas_g = self.graph.compile()
            self.modelo = onp.asarray(modelo_g, dtype=float)
            self.nas = onp.asarray(nas_g, dtype=int)
            self.n_componentes, self.nspec = self.modelo.shape
            self.mt = self.modelo.T
            self._graph_revision = rev

    # ----- transformaciones K (siguiendo tu convención actual) -----
    def step_by_step(self, K_log):
        return onp.cumsum(onp.asarray(K_log, dtype=float))

    def non_coop(self, K_log):
        """
        Modo estadístico / no cooperativo para una serie 1:N (o N:1) con 2 componentes.

        Entrada:
          - K_log: vector en log10 que incluye ceros para componentes al inicio.
            En modo Non-cooperative se espera que el usuario provea SOLO log10(K1).

        Salida:
          - log10(beta) por especie (tamaño nspec), con asignación robusta aunque
            el orden de columnas de complejos no sea HG, HG2, ...
        """
        K_log = onp.asarray(K_log, dtype=float).ravel()
        n_comp = self.n_componentes
        nspec = self.nspec
        if K_log.size < n_comp + 1:
            return onp.cumsum(K_log)

        N, j_per_complex, complex_cols = infer_noncoop_series(self.modelo)

        logK1 = float(K_log[n_comp])
        js = onp.arange(1, N + 1, dtype=float)
        logK_by_j = logK1 + onp.log10((N - js + 1.0) / (js * float(N)))
        logBeta_by_j = onp.cumsum(logK_by_j)

        logBeta_species = onp.zeros(nspec, dtype=float)
        for col, j in zip(complex_cols.tolist(), j_per_complex.tolist()):
            logBeta_species[int(col)] = float(logBeta_by_j[int(j) - 1])
        return logBeta_species


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
            if not (k_in.size == 1 or k_in.size == n_complex):
                raise ValueError(
                    f"Para 'Non-cooperative' se esperan 1 o {n_complex} constantes, "
                    f"llegaron {k_in.size}."
                )
            K_log_eff = self.non_coop(K_log_in) if k_in.size == 1 else self.step_by_step(K_log_in)

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
        self._recompile_graph_if_needed()
        # Overwrites opcionales
        tol = self.tol if tol is None else float(tol)
        max_iter = self.max_iter if max_iter is None else int(max_iter)
        lam0 = self.lam0 if lam0 is None else float(lam0)
        lam_up = self.lam_up if lam_up is None else float(lam_up)
        lam_down = self.lam_down if lam_down is None else float(lam_down)
        max_step = self.max_step if max_step is None else float(max_step)
        max_backtrack = self.max_backtrack if max_backtrack is None else int(max_backtrack)

        ridge = 1e-12
        eye_n = onp.eye(self.n_comp, dtype=float)
        c_calculada = onp.zeros((self.n_reac, self.nspec), dtype=float)

        K_num = self._prepare_K_numeric(K)

        # Buffers preasignados para reducir allocaciones en el while.
        u = onp.empty(self.n_comp, dtype=float)
        c0 = onp.empty(self.n_comp, dtype=float)
        mt_u = onp.empty(self.nspec, dtype=float)
        c_spec = onp.empty(self.nspec, dtype=float)
        d = onp.empty(self.n_comp, dtype=float)
        mass_balance = onp.empty(self.n_comp, dtype=float)
        j_weighted = onp.empty((self.nspec, self.n_comp), dtype=float)
        J = onp.empty((self.n_comp, self.n_comp), dtype=float)
        Jd = onp.empty((self.n_comp, self.n_comp), dtype=float)
        diagJ = onp.empty(self.n_comp, dtype=float)
        P = onp.empty(self.n_comp, dtype=float)
        PJP = onp.empty((self.n_comp, self.n_comp), dtype=float)
        Pd = onp.empty(self.n_comp, dtype=float)
        delta_u = onp.empty(self.n_comp, dtype=float)
        PJ = onp.empty((self.n_comp, self.n_comp), dtype=float)
        c_trial = onp.empty(self.nspec, dtype=float)
        d_trial = onp.empty(self.n_comp, dtype=float)
        u_trial = onp.empty(self.n_comp, dtype=float)

        def eval_state(u_vec: onp.ndarray, c_spec_out: onp.ndarray, d_out: onp.ndarray, ctot_row: onp.ndarray) -> None:
            mt_u[:] = self.mt @ u_vec
            onp.exp(mt_u, out=mt_u)
            onp.multiply(K_num, mt_u, out=c_spec_out)
            mass_balance[:] = self.modelo @ c_spec_out
            d_out[:] = ctot_row
            d_out -= mass_balance

        def fill_jacobian(c_spec_vec: onp.ndarray) -> None:
            onp.multiply(c_spec_vec[:, None], self.mt, out=j_weighted)
            J[:] = self.mt.T @ j_weighted

        for i in range(self.n_reac):
            ctot_row = self.C_T[i, :self.n_comp]
            # u = ln(c) inicial
            onp.clip(ctot_row, 1e-18, None, out=c0)
            onp.log(c0, out=u)

            # estado inicial
            eval_state(u, c_spec, d, ctot_row)
            prev = float(onp.linalg.norm(d, ord=2))
            lam = float(lam0)

            it = 0
            while prev > tol and it < max_iter:
                fill_jacobian(c_spec)                            # (n_comp, n_comp)
                Jd[:] = J
                Jd += (lam + ridge) * eye_n                     # damping

                # Precondicionamiento Jacobi
                diagJ[:] = Jd.diagonal()
                onp.clip(diagJ, 1e-30, None, out=diagJ)
                onp.sqrt(diagJ, out=P)
                P[:] = 1.0 / P
                onp.multiply(P[:, None], Jd, out=PJP)
                PJP *= P[None, :]
                onp.multiply(P, d, out=Pd)

                # Resolver Δz y llevar a Δu
                try:
                    delta_z = onp.linalg.solve(PJP, Pd)
                except onp.linalg.LinAlgError:
                    delta_z = pinv_cs(PJP) @ Pd
                onp.multiply(P, delta_z, out=delta_u)

                # Limitar paso
                ndu = float(onp.linalg.norm(delta_u))
                if ndu > max_step:
                    delta_u *= (max_step / max(ndu, 1e-18))

                # Backtracking (Armijo)
                alpha = 1.0
                accepted = False
                for _ in range(max_backtrack):
                    u_trial[:] = u
                    u_trial += alpha * delta_u
                    eval_state(u_trial, c_trial, d_trial, ctot_row)
                    cur = float(onp.linalg.norm(d_trial, ord=2))
                    if cur < prev:
                        accepted = True
                        break
                    alpha *= 0.5

                if accepted:
                    u[:] = u_trial
                    c_spec[:] = c_trial
                    d[:] = d_trial
                    prev = cur
                    lam = max(lam * lam_down, 1e-12)
                    it += 1
                    continue

                # Fallback: Newton amortiguado (mismo precondicionamiento)
                onp.multiply(P[:, None], J, out=PJ)
                PJ *= P[None, :]
                Jd[:] = PJ
                Jd += ridge * eye_n
                try:
                    delta_zN = onp.linalg.solve(Jd, Pd)
                except onp.linalg.LinAlgError:
                    delta_zN = pinv_cs(Jd) @ Pd
                onp.multiply(P, delta_zN, out=delta_u)

                nduN = float(onp.linalg.norm(delta_u))
                if nduN > max_step:
                    delta_u *= (max_step / max(nduN, 1e-18))

                alpha = 1.0
                acceptedN = False
                for _ in range(max_backtrack):
                    u_trial[:] = u
                    u_trial += alpha * delta_u
                    eval_state(u_trial, c_trial, d_trial, ctot_row)
                    cur = float(onp.linalg.norm(d_trial, ord=2))
                    if cur < prev:
                        acceptedN = True
                        break
                    alpha *= 0.5

                if acceptedN:
                    u[:] = u_trial
                    c_spec[:] = c_trial
                    d[:] = d_trial
                    prev = cur
                    lam = min(lam * lam_up, 1e12)
                    it += 1
                else:
                    # Reinicio seguro para evitar estancamientos
                    onp.clip(ctot_row, 1e-16, None, out=c0)
                    onp.log(c0, out=u)
                    eval_state(u, c_spec, d, ctot_row)
                    prev = float(onp.linalg.norm(d, ord=2))
                    lam = lam0 * 10.0
                    it += 1

            c_calculada[i, :] = c_spec

        if self.graph is not None:
            self.graph.set_last_solution(c_calculada)

        # Eliminar especies no absorbentes (NumPy real)
        if self.nas.size > 0:
            C_np = onp.delete(c_calculada, self.nas, axis=1)
        else:
            C_np = c_calculada

        # Regresar en el backend (JAX/NumPy)
        return np.asarray(C_np, dtype=float), np.asarray(c_calculada, dtype=float)
