import jax
import jax.numpy as jnp
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
x = jnp.array([1.0, 2.0, 3.0])
print("JAX array:", x)
