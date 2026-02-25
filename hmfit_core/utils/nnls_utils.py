from .np_backend import xp as np, jacrev, jit, lax, USE_JAX
# from jax import lax # Removed direct import

@jit
def _nnls_step(C, Y, A, ridge, step):
    # grad = Cᵀ(CA − Y) + λA
    R = C @ A - Y                          # (m, nw)
    grad = C.T @ R + ridge * A             # (s, nw)
    A_next = np.maximum(A - step * grad, 0.)
    return A_next

def solve_A_nnls_pgd2(C, Y, ridge=0.0, max_iters=300):
    """
    NNLS por descenso de gradiente proyectado (PGD) para todas las columnas de Y.
    C: (m, s), Y: (m, nw)  ->  A: (s, nw) con A >= 0
    ridge: λ >= 0 opcional para estabilidad; por defecto 0.
    """
    C = np.asarray(C)
    Y = np.asarray(Y)

    # Paso = 1 / (||C||₂² + λ)
    # ||C||₂ = mayor valor singular; en JAX funciona con rectangulares.
    #smax = np.linalg.norm(C, ord=2)
    s = np.linalg.svd(C, compute_uv=False)
    smax = s.max()
    L = smax * smax + ridge
    step = 1.0 / (L + 1e-12)

    # Inicialización: no negativa y razonable
    A0 = np.maximum(np.linalg.pinv(C) @ Y, 0.0).astype(C.dtype)

    def body(i, A):
        return _nnls_step(C, Y, A, ridge, step)

    A = lax.fori_loop(0, max_iters, body, A0)
    return A


##############################################################
# Versión con penalización suave para A<0
##############################################################
@jit
def _penalty_step(C, Y, A, ridge, mu, step):
    # R = CA − Y
    R = C @ A - Y                        # (m, nw)
    A_neg = np.minimum(A, 0.0)          # parte negativa (subgradiente hinge^2)
    # ∇ = Cᵀ(CA − Y) + λA + μ·min(A,0)
    grad = C.T @ R + ridge * A + mu * A_neg
    return A - step * grad

@jit
def _penalty_smooth_step(C, Y, A, ridge, mu, alpha_smooth, smooth_matrix, step):
    # R = CA − Y
    R = C @ A - Y
    A_neg = np.minimum(A, 0.0)
    smooth_term = A @ smooth_matrix
    grad = C.T @ R + ridge * A + mu * A_neg + alpha_smooth * smooth_term
    return A - step * grad


def solve_A_nnls_pgd(
    C,
    Y,
    ridge=0.0,
    mu=1e-2,
    max_iters=300,
    lower_bound=None,
    alpha_smooth=0.0,
    smooth_matrix=None,
):
    """
    Minimiza ½||CA−Y||² + ½·ridge||A||² + ½·mu||min(A,0)||²
    (sin proyección dura; permite A<0 pero lo penaliza).
    C: (m,s), Y: (m,nw) -> A: (s,nw)
    lower_bound: límite inferior suave opcional (p.ej. -δ).
    """
    C = np.asarray(C)
    Y = np.asarray(Y)

    # Paso estable: 1/(||C||² + ridge + mu + alpha*||L||)
    svals = np.linalg.svd(C, compute_uv=False)
    lips = (svals.max() ** 2) + ridge + mu

    alpha = float(alpha_smooth)
    Lmat = None
    if alpha > 0.0 and smooth_matrix is not None:
        Lcand = np.asarray(smooth_matrix, dtype=C.dtype)
        if Lcand.ndim == 2 and Lcand.shape[0] == Lcand.shape[1] == Y.shape[1]:
            svals_l = np.linalg.svd(Lcand, compute_uv=False)
            lips = lips + alpha * svals_l.max()
            Lmat = Lcand
        else:
            alpha = 0.0
    else:
        alpha = 0.0

    step = 1.0 / (lips + 1e-12)

    # Warm start (puede ser negativo; la penalización lo corrige)
    A0 = (np.linalg.pinv(C) @ Y).astype(C.dtype)

    if lower_bound is None:
        if alpha > 0.0 and Lmat is not None:
            def body(i, A):
                return _penalty_smooth_step(C, Y, A, ridge, mu, alpha, Lmat, step)
            A = lax.fori_loop(0, max_iters, body, A0)
        else:
            def body(i, A):
                return _penalty_step(C, Y, A, ridge, mu, step)
            A = lax.fori_loop(0, max_iters, body, A0)
    else:
        lb = np.asarray(lower_bound, dtype=A0.dtype)
        if lb.ndim == 1:
            lb = lb[:, None]
        if alpha > 0.0 and Lmat is not None:
            def body(i, A):
                A_next = _penalty_smooth_step(C, Y, A, ridge, mu, alpha, Lmat, step)
                return np.maximum(A_next, lb)
            A = lax.fori_loop(0, max_iters, body, A0)
        else:
            def body(i, A):
                A_next = _penalty_step(C, Y, A, ridge, mu, step)
                return np.maximum(A_next, lb)
            A = lax.fori_loop(0, max_iters, body, A0)
    return A


def residuals(k, Y, conc_fn):
    # conc_fn(k) -> (C, extra)  donde C es (m, s)
    C, *_ = conc_fn(k)
    #A = np.linalg.pinv(C) @ Y.T           # coeficientes por mínimos cuadrados
    A = solve_A_nnls_pgd(C, Y.T, ridge=0.0, max_iters=300)
    r = C @ A - Y.T                       # residuos por columna
    return r.ravel()

if USE_JAX:
    jacobian_r = jit(jacrev(lambda kk, Y, conc_fn: residuals(kk, Y, conc_fn), argnums=0))
else:
    jacobian_r = None
