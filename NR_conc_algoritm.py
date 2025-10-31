# NR_conc_algoritm.py
# ---------------------------------------------
# Newton–Raphson para balance de masa con:
#  - Paso positivo (c >= 0)
#  - Búsqueda de línea (Armijo) y paso máximo
#  - Cálculo interno en NumPy real (onp), salida en backend (np)
import numpy as onp
from np_backend import xp as np  # backend conmutable (JAX/NumPy)

# ------------------------------------------------------------------
# Pseudoinversa robusta en NumPy real (fallback)
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
      - model_sett : string (p.ej., 'Free') — aquí no afecta al NR
    """

    def __init__(self, ctot, modelo, nas=None, model_sett=None,
                 tol=1e-10, max_iter=200, k_is_log10=True,
                 max_step=2.0, max_backtrack=8):
        self.modelo = onp.asarray(modelo, dtype=float)
        self.ctot   = onp.asarray(ctot,   dtype=float)
        self.nas    = onp.asarray(nas if nas is not None else [], dtype=int)
        _ = model_sett  # no usado aquí

        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.k_is_log10 = bool(k_is_log10)

        self.max_step = float(max_step)          # norma máxima del paso Δc
        self.max_backtrack = int(max_backtrack)  # reintentos de backtracking

        # shapes
        self.n_componentes, self.nspec = self.modelo.shape
        self.n_reacciones = self.ctot.shape[0]

        self.mt = self.modelo.T  # (nspec, n_componentes)

    # ---------- núcleo NR con búsqueda de línea (NumPy real) ----------
    def _solve_single_nr(self, ctot_i, K):
        """
        Resuelve c (componentes) por NR amortiguado para una corrida.
        Retorna (c_final, ||res||, c_spec_final).
        """
        n_comp = self.n_componentes
        M  = self.modelo
        MT = self.mt

        # inicial positivo
        c = onp.maximum(ctot_i[:n_comp].astype(float).copy(), 1e-14)

        def especies_from_c(c_vec):
            # c_spec = K * exp(M^T log(c))
            log_c = onp.log(onp.clip(c_vec, 1e-300, onp.inf))
            return onp.exp(MT @ log_c) * K

        def residuo(c_vec):
            c_spec = especies_from_c(c_vec)
            return self.ctot_row - (M @ c_spec), c_spec  # (n_comp,), (nspec,)

        def jacobian(c_vec, c_spec):
            # J_{j,h} = sum_p M_{j,p} * v_{h,p} * c_spec_p / c_h
            inv_c = 1.0 / onp.clip(c_vec, 1e-300, onp.inf)
            term = M * c_spec  # (n_comp, nspec), broadcasting por filas (v_{h,p}*c_spec_p)
            J = onp.empty((n_comp, n_comp), dtype=float)
            for h in range(n_comp):
                col_h = term[h, :] * inv_c[h]      # (nspec,)
                J[:, h] = M @ col_h                # (n_comp,)
            return J

        # estado inicial
        self.ctot_row = ctot_i
        d, c_spec = residuo(c)
        prev = float(onp.linalg.norm(d, ord=2))
        it = 0
        ridge = 1e-12

        while prev > self.tol and it < self.max_iter:
            J = jacobian(c, c_spec)

            # Resolver J Δ = d (o pseudo-inversa si mal condicionado)
            try:
                delta = onp.linalg.solve(J, d)
            except onp.linalg.LinAlgError:
                delta = pinv_cs(J) @ d

            # limitar tamaño de paso
            ndelta = float(onp.linalg.norm(delta, ord=2))
            if ndelta > self.max_step:
                delta *= (self.max_step / max(ndelta, 1e-18))

            # backtracking (Armijo simple): aceptar si ||r|| baja
            alpha = 1.0
            accepted = False
            for _ in range(self.max_backtrack):
                cand = onp.maximum(c + alpha * delta, 1e-14)  # mantiene positividad
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

            # fallback: paso de Gauss–Newton “regularizado” (J^T J + μI) Δ = J^T d
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
                # si nada mejora, reduce agresivamente el paso
                c = onp.maximum(c + 0.1 * delta_gn, 1e-14)
                d, c_spec = residuo(c)
                prev = float(onp.linalg.norm(d, ord=2))
                it += 1

        return c, prev, c_spec

    # ------------------ API llamado por la GUI ------------------
    def concentraciones(self, k):
        """
        Devuelve:
          C  : (n_reacciones, nspec - len(nas))  (solo absorbentes)
          c_calculada : (n_reacciones, nspec)    (todas las especies)
        """
        # Construye K con tamaño nspec (1.0 para componentes si k no los trae)
        k = onp.asarray(k, dtype=float).ravel()
        kval = onp.power(10.0, k) if self.k_is_log10 else k

        n_comp = self.n_componentes
        nspec  = self.nspec

        if kval.size == nspec:
            K = kval.astype(float)
        elif kval.size == (nspec - n_comp):
            K = onp.ones(nspec, dtype=float)
            K[n_comp:] = kval
        else:
            raise ValueError(
                f"Longitud de k incompatible: len(k)={kval.size}, "
                f"se esperaba {nspec} o {nspec - n_comp} (nspec={nspec}, n_comp={n_comp})."
            )

        c_calculada = onp.zeros((self.n_reacciones, nspec), dtype=float)

        for i in range(self.n_reacciones):
            c_i, _, c_spec_i = self._solve_single_nr(self.ctot[i, :], K)
            c_calculada[i, :] = c_spec_i

        # eliminar no-absorbentes en NumPy real
        if self.nas.size > 0:
            C_np = onp.delete(c_calculada, self.nas, axis=1)
        else:
            C_np = c_calculada

        # salida en backend
        return np.asarray(C_np, dtype=float), np.asarray(c_calculada, dtype=float)
