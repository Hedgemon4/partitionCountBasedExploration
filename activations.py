from jaxtyping import Array, ArrayLike
import jax.numpy as jnp
import jax

### Talked to quinn about sparsity; later layers less sparse?


def fta(x: ArrayLike, centres, eta) -> Array:
    z = jnp.expand_dims(x, axis=-1)
    term1 = jax.nn.relu(centres - z)
    term2 = jax.nn.relu(z - eta - centres)
    combined = term1 + term2
    return 1.0 - fta_indicator(
        combined, eta
    )


def fta_indicator(x, eta):
    return jnp.where(x < eta, x, 0.0) + jnp.where(x > eta, 1.0, 0.0)
