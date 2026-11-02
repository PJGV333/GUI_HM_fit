import numpy as onp

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


class WeightedLinearSolveError(RuntimeError):
    """A weighted linear optimization could not produce a reliable solution."""


def _solve_weighted_penalty_newton(C, Y, W, ridge, mu, lower_bound,
                                   alpha_smooth, smooth_matrix, max_iters):
    """Projected semismooth Newton for the weighted, penalized linear problem.

    The data Hessian, the active negative-coefficient penalty and the smoothness
    Hessian are assembled explicitly. A projected line search handles both the
    lower bound and changes in the penalty active set. Never return a result
    without checking projected first-order optimality.
    """
    C, Y, W = (onp.asarray(x, dtype=float) for x in (C, Y, W))
    s, nw = C.shape[1], Y.shape[1]
    size = s * nw
    if size > 4000:
        raise WeightedLinearSolveError("weighted Newton system exceeds the supported size")
    W2 = W * W
    H0 = onp.zeros((size, size), dtype=float)
    for j in range(nw):
        indices = onp.arange(s) * nw + j
        H0[onp.ix_(indices, indices)] = C.T @ (W2[:, j, None] * C)
    H0.flat[::size + 1] += float(ridge)
    L = None
    if alpha_smooth > 0 and smooth_matrix is not None:
        L = onp.asarray(smooth_matrix, dtype=float)
        if L.shape != (nw, nw):
            raise ValueError("smooth_matrix shape does not match the data")
        H0 += float(alpha_smooth) * onp.kron(onp.eye(s), L)
    b = (C.T @ (W2 * Y)).ravel()
    lower = onp.full(size, -onp.inf)
    if lower_bound is not None:
        lb = onp.asarray(lower_bound, dtype=float)
        if lb.ndim == 1 and lb.size == s:
            lb = lb[:, None]
        lower = onp.broadcast_to(lb, (s, nw)).ravel()
    x = onp.maximum(onp.zeros(size), lower)

    def objective(z):
        A = z.reshape(s, nw)
        return (0.5 * onp.sum((W * (C @ A - Y)) ** 2)
                + 0.5 * float(ridge) * onp.dot(z, z)
                + 0.5 * float(mu) * onp.dot(onp.minimum(z, 0), onp.minimum(z, 0))
                + (0.5 * float(alpha_smooth) * onp.sum((A @ L) * A)
                   if L is not None else 0.0))

    for _ in range(int(max_iters)):
        grad = H0 @ x - b + float(mu) * onp.minimum(x, 0)
        active = (x <= lower + 1e-12) & (grad > 0)
        projected_grad = onp.where(active, 0.0, grad)
        if onp.max(onp.abs(projected_grad), initial=0.0) <= 1e-10 * max(1.0, onp.max(onp.abs(b), initial=0.0)):
            return np.asarray(x.reshape(s, nw))
        free = onp.flatnonzero(~active)
        H = H0[onp.ix_(free, free)].copy()
        H.flat[::len(free) + 1] += float(mu) * (x[free] < 0)
        scale = max(float(onp.max(onp.abs(onp.diag(H)), initial=0.0)), 1e-20)
        direction = onp.zeros(size)
        try:
            direction[free] = onp.linalg.solve(H + onp.eye(len(free)) * scale * 1e-12,
                                               -grad[free])
        except onp.linalg.LinAlgError as exc:
            raise WeightedLinearSolveError("weighted Newton system is singular") from exc
        value = objective(x)
        step = 1.0
        for _ in range(50):
            candidate = onp.maximum(x + step * direction, lower)
            if objective(candidate) <= value + 1e-4 * onp.dot(grad, candidate - x):
                x = candidate
                break
            step *= 0.5
        else:
            raise WeightedLinearSolveError("weighted Newton line search failed")
    raise WeightedLinearSolveError("weighted Newton did not converge")


def solve_A_nnls_pgd(
    C,
    Y,
    ridge=0.0,
    mu=1e-2,
    max_iters=300,
    lower_bound=None,
    alpha_smooth=0.0,
    smooth_matrix=None,
    weights_row=None,
):
    """
    Minimiza ½||CA−Y||² + ½·ridge||A||² + ½·mu||min(A,0)||²
    (sin proyección dura; permite A<0 pero lo penaliza).
    C: (m,s), Y: (m,nw) -> A: (s,nw)
    lower_bound: límite inferior suave opcional (p.ej. -δ).
    weights_row: multiplicadores del residuo (m,nw): el término de datos es
    ½Σ_ij w_ij² (CA−Y)_ij². ridge, mu y alpha son penalizaciones absolutas;
    multiplicadores constantes distintos de uno cambian su importancia relativa.
    """
    C = np.asarray(C)
    Y = np.asarray(Y)

    Wm = None
    if weights_row is not None:
        Wm = np.asarray(weights_row, dtype=C.dtype)
        Wm = np.nan_to_num(np.abs(Wm), nan=0.0, posinf=0.0, neginf=0.0)
        if Wm.ndim == 1:
            if Wm.size == Y.shape[1]:
                Wm = np.broadcast_to(Wm.reshape(1, Y.shape[1]), (C.shape[0], Y.shape[1]))
            elif Wm.size == C.shape[0]:
                Wm = np.broadcast_to(Wm.reshape(C.shape[0], 1), (C.shape[0], Y.shape[1]))
        elif Wm.shape != (C.shape[0], Y.shape[1]):
            Wm = np.broadcast_to(Wm, (C.shape[0], Y.shape[1]))
        # Absolute penalty semantics: a constant multiplier other than one
        # still changes the data/penalty balance, even without row variation.
        if bool(np.any(Wm != 1.0)):
            return _solve_weighted_penalty_newton(
                C, Y, Wm, ridge, mu, lower_bound, alpha_smooth, smooth_matrix, max_iters
            )
        Wm = None  # unit multipliers preserve the legacy unweighted path

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
