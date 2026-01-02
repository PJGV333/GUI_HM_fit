# np_backend.py
import os

# Enable JAX by default; allow override with HMFIT_USE_JAX=0/1.
_USE_JAX_ENV = os.environ.get("HMFIT_USE_JAX", "").strip().lower()
_JAX_ON = ("1", "true", "yes", "y", "on")
_JAX_OFF = ("0", "false", "no", "n", "off")
if _USE_JAX_ENV in _JAX_OFF:
    USE_JAX = False
elif _USE_JAX_ENV in _JAX_ON:
    USE_JAX = True
else:
    USE_JAX = True

if USE_JAX:
    try:
        from jax import config
        config.update("jax_enable_x64", True)
        import jax
        import jax.numpy as xp
        from jax import jit, jacrev, vmap, value_and_grad, lax
    except Exception:
        USE_JAX = False

if not USE_JAX:
    import numpy as xp
    def jit(f): return f
    def jacrev(f): raise NotImplementedError("jacrev no disponible con NumPy")
    def vmap(f, *a, **k): return f
    def value_and_grad(f): raise NotImplementedError("value_and_grad no disponible con NumPy")
    class _Lax:  # placeholders para mantener API
        cond = while_loop = fori_loop = None
        def fori_loop(self, lower, upper, body_fun, init_val):
             val = init_val
             for i in range(lower, upper):
                 val = body_fun(i, val)
             return val
    lax = _Lax()

# Conversor seguro: pandas/iterables -> array del backend
import numpy as _onp
def to_xp(x, dtype=float):
    return xp.asarray(_onp.asarray(x, dtype=dtype))
