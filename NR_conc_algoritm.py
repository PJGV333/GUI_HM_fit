# NR_conc_algoritm.py
# ---------------------------------------------
# Newton–Raphson para balance de masa con:
#  - Paso positivo (c >= 0)
#  - Búsqueda de línea (Armijo) y paso máximo
#  - Cálculo interno en NumPy real (onp), salida en backend (np)

import numpy as onp
from np_backend import xp as np  # backend conmutable (JAX/NumPy)
from noncoop_utils import infer_noncoop_series

# ------------------------------------------------------------------
def pinv_cs(A, rcond=1e-12):
    A = onp.asarray(A)
    try:
        return onp.linalg.pinv(A, rcond=rcond)
    except onp.linalg.LinAlgError:
        ATA = A.T @ A + (rcond if onp.isscalar(rcond) else 1e-12) * onp.eye(A.shape[1], dtype=A.dtype)
        return onp.linalg.solve(ATA, A.T)
# ------------------------------------------------------------------


class NewtonRaphson:
    """
    Llamada esperada por la GUI:
        res = NewtonRaphson(ctot, modelo, nas, model_sett, ...)
    Donde:
      - ctot   : (n_reacciones, n_componentes)
      - modelo : (n_componentes, nspec)
      - nas    : índices de especies no-absorbentes (columnas a eliminar)
      - model_sett : 'Free' | 'Step by step' | 'Non-cooperative'
    """

    def __init__(self, ctot, modelo, nas=None, model_sett=None,
                 tol=1e-10, max_iter=200, k_is_log10=True,
                 max_step=2.0, max_backtrack=8):

        self.modelo = onp.asarray(modelo, dtype=float)
        self.ctot   = onp.asarray(ctot,   dtype=float)
        self.nas    = onp.asarray(nas if nas is not None else [], dtype=int)
        self.model_sett = model_sett or "Free"

        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.k_is_log10 = bool(k_is_log10)  # la GUI manda log10(K); lo mantenemos por compatibilidad

        self.max_step = float(max_step)
        self.max_backtrack = int(max_backtrack)

        # shapes
        self.n_componentes, self.nspec = self.modelo.shape
        self.n_reacciones = self.ctot.shape[0]

        self.mt = self.modelo.T  # (nspec, n_componentes)

    # ----- transformaciones de K según configuración -----

    def step_by_step(self, K_log):
        # cumsum en log10 → β acumuladas
        return onp.cumsum(onp.asarray(K_log, dtype=float))

    def non_coop(self, K_log):
        """
        Modo estadístico / no cooperativo para una serie 1:N (o N:1) con 2 componentes.

        Entrada:
          - K_log: vector en log10 que incluye ceros para componentes al inicio.
            En modo Non-cooperative se espera que el usuario provea SOLO log10(K1)
            (primer paso macroscópico).

        Salida:
          - log10(beta) por especie (tamaño nspec), con asignación robusta aunque
            el orden de columnas de complejos no sea HG, HG2, ...
        """
        K_log = onp.asarray(K_log, dtype=float).ravel()
        n_comp = self.n_componentes
        nspec = self.nspec
        if K_log.size < n_comp + 1:
            return onp.cumsum(K_log)

        # Infer N and mapping j -> complex column
        N, j_per_complex, complex_cols = infer_noncoop_series(self.modelo)

        logK1 = float(K_log[n_comp])
        js = onp.arange(1, N + 1, dtype=float)
        # logK_j = logK1 + log10((N-j+1)/(j*N))
        logK_by_j = logK1 + onp.log10((N - js + 1.0) / (js * float(N)))
        logBeta_by_j = onp.cumsum(logK_by_j)

        logBeta_species = onp.zeros(nspec, dtype=float)
        for col, j in zip(complex_cols.tolist(), j_per_complex.tolist()):
            logBeta_species[int(col)] = float(logBeta_by_j[int(j) - 1])
        return logBeta_species

    def _prepare_K_numeric(self, k_in):
        """
        Aplica model_sett y regresa K numérico (no log10), con dtype promovido.
        La GUI envía K en log10 para los complejos:
          - Free / Step by step → n_complex valores
          - Non-cooperative     → n_complex - 1 (p.ej. 1:2 → solo K1)
        """
        k_in = onp.asarray(k_in, dtype=float).ravel()
        n_comp   = self.n_componentes
        nspec    = self.nspec
        n_complex = nspec - n_comp

        # Pre-padding de ceros (componentes) en log10
        pre_ko = onp.zeros(n_comp, dtype=float)
        K_log_full = onp.concatenate((pre_ko, k_in))  # aún en log10

        ms = self.model_sett
        if ms == "Free":
            # Deben venir todos los complejos
            if k_in.size != n_complex:
                raise ValueError(f"Para 'Free' se esperan {n_complex} constantes, llegaron {k_in.size}.")
            K_log_eff = K_log_full
        elif ms == "Step by step":
            if k_in.size != n_complex:
                raise ValueError(f"Para 'Step by step' se esperan {n_complex} constantes, llegaron {k_in.size}.")
            K_log_eff = self.step_by_step(K_log_full)
        elif ms in ("Non-cooperative", "Statistical"):
            # Non-cooperative: por defecto se espera SOLO K1 (un parámetro).
            # Por compatibilidad, aceptamos también n_complex (y se trata como step-by-step).
            if not (k_in.size == 1 or k_in.size == n_complex):
                raise ValueError(
                    f"Para 'Non-cooperative' se esperan 1 o {n_complex} constantes, "
                    f"llegaron {k_in.size}."
                )
            K_log_eff = self.non_coop(K_log_full) if k_in.size == 1 else self.step_by_step(K_log_full)
        else:
            # fallback conservador
            K_log_eff = K_log_full

        # A numérico
        K_num = onp.power(10.0, K_log_eff)

        # dtype consistente
        dtype = onp.result_type(K_num.dtype, onp.float64)
        return onp.asarray(K_num, dtype=dtype)

    # ------------------ Núcleo NR por corrida ------------------

    def _solve_single_nr(self, ctot_row, K_num):
        """
        Resuelve c (componentes) por NR amortiguado para una corrida.
        Retorna (c_final, ||res||, c_spec_final).
        """
        n_comp = self.n_componentes
        M  = self.modelo
        MT = self.mt

        # inicial positivo
        c = onp.maximum(ctot_row[:n_comp].astype(float).copy(), 1e-14)

        def especies_from_c(c_vec):
            # c_spec = K * exp(M^T log(c))
            log_c = onp.log(onp.clip(c_vec, 1e-300, onp.inf))
            return onp.exp(MT @ log_c) * K_num

        def residuo(c_vec):
            c_spec = especies_from_c(c_vec)
            return ctot_row - (M @ c_spec), c_spec  # (n_comp,), (nspec,)

        def jacobian(c_vec, c_spec):
            inv_c = 1.0 / onp.clip(c_vec, 1e-300, onp.inf)
            term = M * c_spec
            J = onp.empty((n_comp, n_comp), dtype=float)
            for h in range(n_comp):
                col_h = term[h, :] * inv_c[h]
                J[:, h] = M @ col_h
            return J

        # estado inicial
        d, c_spec = residuo(c)
        prev = float(onp.linalg.norm(d, ord=2))
        it = 0
        ridge = 1e-12

        while prev > self.tol and it < self.max_iter:
            J = jacobian(c, c_spec)

            try:
                delta = onp.linalg.solve(J, d)
            except onp.linalg.LinAlgError:
                delta = pinv_cs(J) @ d

            ndelta = float(onp.linalg.norm(delta, ord=2))
            if ndelta > self.max_step:
                delta *= (self.max_step / max(ndelta, 1e-18))

            alpha = 1.0
            accepted = False
            for _ in range(self.max_backtrack):
                cand = onp.maximum(c + alpha * delta, 1e-14)
                d_trial, c_spec_trial = residuo(cand)
                cur = float(onp.linalg.norm(d_trial, ord=2))
                if cur < prev:
                    accepted = True
                    break
                alpha *= 0.5

            if accepted:
                c = cand
                d = d_trial
                c_spec = c_spec_trial
                prev = cur
                it += 1
                continue

            # Gauss–Newton regularizado
            JTJ = J.T @ J + ridge * onp.eye(n_comp)
            g   = J.T @ d
            try:
                delta_gn = onp.linalg.solve(JTJ, g)
            except onp.linalg.LinAlgError:
                delta_gn = pinv_cs(JTJ) @ g

            ndelta = float(onp.linalg.norm(delta_gn, ord=2))
            if ndelta > self.max_step:
                delta_gn *= (self.max_step / max(ndelta, 1e-18))

            alpha = 1.0
            accepted = False
            for _ in range(self.max_backtrack):
                cand = onp.maximum(c + alpha * delta_gn, 1e-14)
                d_trial, c_spec_trial = residuo(cand)
                cur = float(onp.linalg.norm(d_trial, ord=2))
                if cur < prev:
                    accepted = True
                    break
                alpha *= 0.5

            if accepted:
                c = cand
                d = d_trial
                c_spec = c_spec_trial
                prev = cur
                it += 1
            else:
                c = onp.maximum(c + 0.1 * delta_gn, 1e-14)
                d, c_spec = residuo(c)
                prev = float(onp.linalg.norm(d, ord=2))
                it += 1

        return c, prev, c_spec

    # ------------------ API llamado por la GUI ------------------

    def concentraciones(self, k):
        """
        Devuelve:
          C            : (n_reacciones, nspec - len(nas))  (solo absorbentes)
          c_calculada  : (n_reacciones, nspec)             (todas las especies)
        """
        K_num = self._prepare_K_numeric(k)

        # chequeo defensivo
        if K_num.size != self.nspec:
            raise ValueError(f"K_num tiene tamaño {K_num.size}, pero nspec={self.nspec}.")

        nspec = self.nspec
        c_calculada = onp.zeros((self.n_reacciones, nspec), dtype=float)

        for i in range(self.n_reacciones):
            _, _, c_spec_i = self._solve_single_nr(self.ctot[i, :], K_num)
            c_calculada[i, :] = c_spec_i

        # eliminar no-absorbentes en NumPy real
        if self.nas.size > 0:
            C_np = onp.delete(c_calculada, self.nas, axis=1)
        else:
            C_np = c_calculada

        # salida en backend
        return np.asarray(C_np, dtype=float), np.asarray(c_calculada, dtype=float)
