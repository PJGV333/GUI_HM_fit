"""Historical JAX kinetics experiments; not used by production fits."""


def jax_solve_linear_matrix(C, D, weights=None):
    """Project with the same relative rank cutoff as the SciPy observation model."""
    import jax.numpy as jnp
    from hmfit.kinetics.observation.linear_matrix import RANK_RCOND

    if weights is None:
        return jnp.linalg.pinv(C, rtol=RANK_RCOND) @ D
    if weights.ndim == 1 or weights.shape[1] == 1:
        w = weights.reshape(-1, 1)
        return jnp.linalg.pinv(C * w, rtol=RANK_RCOND) @ (D * w)
    return jnp.concatenate([
        jnp.linalg.pinv(C * weights[:, j:j + 1], rtol=RANK_RCOND)
        @ (D[:, j:j + 1] * weights[:, j:j + 1])
        for j in range(D.shape[1])
    ], axis=1)
