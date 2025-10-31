# np_backend.py
USE_JAX = True  # ← mañana lo podemos poner en False si algo falla

if USE_JAX:
    from jax import config
    config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as xp
    from jax import jit, jacrev, vmap, value_and_grad, lax
else:
    import numpy as xp
    def jit(f): return f
    def jacrev(f): raise NotImplementedError("jacrev no disponible con NumPy")
    def vmap(f, *a, **k): return f
    def value_and_grad(f): raise NotImplementedError("value_and_grad no disponible con NumPy")
    class _Lax:  # placeholders para mantener API
        cond = while_loop = None
    lax = _Lax()

# Conversor seguro: pandas/iterables -> array del backend
import numpy as _onp
def to_xp(x, dtype=float):
    return xp.asarray(_onp.asarray(x, dtype=dtype))