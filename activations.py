from jaxtyping import Array, ArrayLike
import jax.numpy as jnp
import jax

### Talked to quinn about sparsity; later layers less sparse?


def fta(x: ArrayLike, centres, eta) -> Array:
    return 1 - fta_indicator(
        jnp.maximum(centres - x, 0) + jnp.maximum(x - eta - centres, 0), eta
    )


def fta_indicator(x, eta):
    return jnp.where(x < eta, x, 0) + jnp.where(x > eta, 1, 0)
