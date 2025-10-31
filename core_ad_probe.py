from np_backend import xp as np, jacrev, jit

def residuals(k, Y, conc_fn):
    # conc_fn(k) -> (C, extra)  donde C es (m, s)
    C, *_ = conc_fn(k)
    A = np.linalg.pinv(C) @ Y.T           # coeficientes por mínimos cuadrados
    r = C @ A - Y.T                       # residuos por columna
    return r.ravel()

jacobian_r = jit(jacrev(lambda kk, Y, conc_fn: residuals(kk, Y, conc_fn), argnums=0))
