# np_backend.py
# The equilibrium/uncertainty engine always uses NumPy.  Availability of JAX
# must not implicitly change its array types or numerical behavior.  Kinetics
# selects its experimental JAX implementation explicitly and independently.
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
