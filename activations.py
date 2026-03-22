from jaxtyping import Array, ArrayLike
import jax.numpy as jnp
import jax

### Talked to quinn about sparsity; later layers less sparse?


def fta(x: ArrayLike, centres, eta) -> Array:
    term1 = jnp.maximum(centres[None, :] - x[:, None], 0)
    term2 = jnp.maximum(x[:, None] - eta - centres[None, :] , 0)
    combined = term1 + term2
    return 1.0 - fta_indicator(
        combined, eta
    )


def fta_indicator(x, eta):
    return jnp.where(x <= eta, x, 0.0) + jnp.where(x > eta, 1.0, 0.0)
