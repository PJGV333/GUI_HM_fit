"""Differentiation of separable spectral fits for fixed inner active sets.

This low-level routine handles independent wavelengths, without smoothing or
lower bounds. It does not replace the existing absorptivity solver.
"""
import numpy as np


def spectral_profile_jacobian(C, dC, Y, A, weights, *, solver_mode, mu=0.0,
                              regularization_mode='legacy'):
    """Return the derivative of the actual profiled residual.

    ``dC`` has shape (parameters, points, components). ``A`` is the output of
    the existing inner solver. Callers must use this only away from changes in
    the sign/active set and only when the inner optimum is well determined.
    """
    C, dC, Y, A = map(lambda v: np.asarray(v, dtype=float), (C, dC, Y, A))
    W = np.broadcast_to(np.asarray(weights, dtype=float), Y.shape)
    m, n = C.shape
    nw = Y.shape[1]
    if dC.ndim != 3 or dC.shape[1:] != C.shape or Y.shape[0] != m or A.shape != (n, nw):
        raise ValueError('incompatible spectral dimensions')
    if solver_mode not in ('soft_penalty', 'nnls_hard') or regularization_mode not in ('legacy', 'absolute'):
        raise ValueError('unsupported solver or regularization mode')
    if mu < 0:
        raise ValueError('negative penalty')
    # Legacy wavelength-only weights do not participate in the inner solve.
    per_point = not np.all(W == W[0:1, :])
    data = np.empty((m, nw, dC.shape[0]))
    penalty = np.zeros((n, nw, dC.shape[0]))
    for k in range(nw):
        w = W[:, k]
        inner_w = w if regularization_mode == 'absolute' or per_point else np.ones(m)
        if regularization_mode == 'absolute':
            mu_inner = mu
        elif per_point:
            nz = inner_w > 0
            mu_inner = mu * np.mean(inner_w[nz] ** 2) if np.any(nz) else 0.0
        else:
            mu_inner = mu
        a = A[:, k]
        free = a > 0 if solver_mode == 'nnls_hard' else np.ones(n, dtype=bool)
        if not np.any(free):
            data[:, k, :] = 0
            continue
        cf = C[:, free]
        af = a[free]
        neg = af < 0 if solver_mode == 'soft_penalty' else np.zeros(af.size, bool)
        M = np.vstack((inner_w[:, None] * cf, np.diag(np.sqrt(mu_inner) * neg)))
        if np.linalg.matrix_rank(M) < M.shape[1]:
            raise ValueError('inner active system is rank deficient')
        pinv = np.linalg.pinv(M)
        residual = np.concatenate((inner_w * (Y[:, k] - cf @ af), -np.sqrt(mu_inner) * neg * af))
        for p, dc in enumerate(dC):
            dcf = dc[:, free]
            dM = np.vstack((inner_w[:, None] * dcf, np.zeros((af.size, af.size))))
            da = -pinv @ (dM @ af) + pinv @ (pinv.T @ (dM.T @ residual))
            data[:, k, p] = w * (dcf @ af + cf @ da)
            if regularization_mode == 'absolute' and solver_mode == 'soft_penalty':
                penalty[free, k, p] = np.sqrt(mu) * neg * da
    result = data.reshape(m * nw, dC.shape[0])
    if regularization_mode == 'absolute' and solver_mode == 'soft_penalty':
        result = np.vstack((result, penalty.reshape(n * nw, dC.shape[0])))
    return result
