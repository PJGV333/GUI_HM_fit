import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def test_jax_smoke() -> None:
    assert hasattr(jax, "__version__")
    _ = jax.devices()
    x = jnp.array([1.0, 2.0, 3.0])
    assert x.shape == (3,)
