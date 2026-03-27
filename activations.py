from jaxtyping import Array, ArrayLike
import jax.numpy as jnp
import jax
import equinox as eqx

from configs.activations import ActivationConfig, FTAConfig, ElephantConfig, ReLUConfig

### Talked to quinn about sparsity; later layers less sparse?


class FTA(eqx.Module):
    centres: Array
    bound: float
    eta: float
    num_bins: int
    static_centres: bool

    def __init__(self, bound, eta, static_centres=True):
        self.bound = bound
        self.eta = eta
        self.num_bins = int(self.bound * self.eta / 2)
        self.centres = self.eta * jnp.arange(self.num_bins) - self.bound
        self.static_centres = static_centres

    def __call__(self, x: Array) -> Array:
        centres = (
            jax.lax.stop_gradient(self.centres) if self.static_centres else self.centres
        )

        z = jnp.expand_dims(x, axis=-1)
        term1 = jax.nn.relu(centres - z)
        term2 = jax.nn.relu(z - self.eta - centres)
        combined = term1 + term2

        left_activation = jnp.minimum(0.0, z + self.bound)
        right_activation = jnp.maximum(0.0, z - self.bound)
        boundary_active = (z < -self.bound) | (z > self.bound)

        middle_activation = 1.0 - self.fta_indicator(combined)
        middle_activation = jnp.where(boundary_active, 0.0, middle_activation)

        final_activation = jnp.concatenate([left_activation, middle_activation, right_activation], axis=-1)

        return final_activation

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
        return self.h / (1 + jnp.abs(x / self.a) ** self.d)


class TiledElephantActivation(eqx.Module):
    a: Array
    h: Array
    d: float
    centres: Array
    static_centres: bool

    def __init__(self, a, h, d, num_features, num_bins, bin_size, static_centres=True):
        self.a = jnp.ones(num_features) * a
        self.h = jnp.ones(num_features) * h
        self.d = d
        self.centres = jnp.arange(num_bins) * bin_size - (num_bins * bin_size / 2)
        self.static_centres = static_centres
        self.num_bins = num_bins

    def __call__(self, x):
        z = jnp.expand_dims(x, axis=-1)
        centres = (
            jax.lax.stop_gradient(self.centres) if self.static_centres else self.centres
        )
        return self.h[:, None] / (1 + jnp.abs(z - centres / self.a[:, None]) ** self.d)


def make_activation(config: ActivationConfig):
    if isinstance(config, FTAConfig):
        return FTA(
            bound=config.bound, eta=config.eta, static_centres=config.static_centres
        )
    elif isinstance(config, ElephantConfig):
        return ElephantActivation(a=config.a, h=config.h, d=config.d)
    elif isinstance(config, ReLUConfig):
        return eqx.nn.Lambda(jax.nn.relu)
    raise ValueError(f"Unknown config: {type(config)}")
