from jaxtyping import Array, ArrayLike
import jax.numpy as jnp
import jax
import equinox as eqx

### Talked to quinn about sparsity; later layers less sparse?


class FTA(eqx.Module):
    centres: Array
    bound: float
    eta: float
    num_bins: int
    static_centres: bool

    ### TODO: Add linear tile!

    def __init__(self, bound, eta, static_centres=True):
        self.bound = bound
        self.eta = eta
        self.num_bins = int(self.bound * self.eta / 2)
        self.centres = self.eta * jnp.arange(self.num_bins) - self.bound
        self.static_centres = static_centres

    def __call__(self, x: Array) -> Array:
        jax.debug.print("FTA Centres: {}", self.centres)
        centres = jax.lax.stop_gradient(self.centres) if self.static_centres else self.centres

        z = jnp.expand_dims(x, axis=-1)
        term1 = jax.nn.relu(centres - z)
        term2 = jax.nn.relu(z - self.eta - centres)
        combined = term1 + term2
        return 1.0 - self.fta_indicator(combined)

    def fta_indicator(self, x):
        return jnp.where(x < self.eta, x, 0.0) + jnp.where(x > self.eta, 1.0, 0.0)


class ElephantActivation(eqx.Module):
    a: float
    h: float
    d: float

    def __init__(self, a, h, d):
        self.a = a
        self.h = h
        self.d = d

    def __call__(self, x):
        return (self.h / (1 + jnp.abs(x / self.a) ** self.d))


