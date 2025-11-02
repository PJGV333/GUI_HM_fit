from np_backend import xp as np, jacrev, jit
from jax import lax

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

def solve_A_nnls_pgd(C, Y, ridge=0.0, mu=1e-2, max_iters=300):
    """
    Minimiza ½||CA−Y||² + ½·ridge||A||² + ½·mu||min(A,0)||²
    (sin proyección dura; permite A<0 pero lo penaliza).
    C: (m,s), Y: (m,nw) -> A: (s,nw)
    """
    C = np.asarray(C)
    Y = np.asarray(Y)

    # Paso estable: 1/(||C||² + ridge + mu)
    svals = np.linalg.svd(C, compute_uv=False)
    L = (svals.max() ** 2) + ridge + mu
    step = 1.0 / (L + 1e-12)

    # Warm start (puede ser negativo; la penalización lo corrige)
    A0 = (np.linalg.pinv(C) @ Y).astype(C.dtype)

    def body(i, A):
        return _penalty_step(C, Y, A, ridge, mu, step)

    A = lax.fori_loop(0, max_iters, body, A0)
    return A


def residuals(k, Y, conc_fn):
    # conc_fn(k) -> (C, extra)  donde C es (m, s)
    C, *_ = conc_fn(k)
    #A = np.linalg.pinv(C) @ Y.T           # coeficientes por mínimos cuadrados
    A = solve_A_nnls_pgd(C, Y.T, ridge=0.0, max_iters=300)
    r = C @ A - Y.T                       # residuos por columna
    return r.ravel()

jacobian_r = jit(jacrev(lambda kk, Y, conc_fn: residuals(kk, Y, conc_fn), argnums=0))