import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, PyTree

### TODO: Use tyro instead of this
SEED = 1
LEARNING_RATE = 1e-3

class QNetwork(eqx.Module):
    layers: list

    def __init__(self, input_size, num_actions, key):
        key1, key2, key3 = jax.random.split(key, 3)

        # Might need to transpose for atari
        self.layers = [
            eqx.nn.Linear(in_features=input_size, out_features=128, key=key1),
            eqx.nn.LayerNorm(128),
            jax.nn.relu,
            eqx.nn.Linear(in_features=128, out_features=64, key=key2),
            eqx.nn.LayerNorm(64),
            jax.nn.relu,
            eqx.nn.Linear(in_features=64, out_features=num_actions, key=key3),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def run():
    print("Starting Run")
    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key, 2)
    model = QNetwork(input_size=4, num_actions=2, key=subkey)
    optim = optax.adam(1e-3)

if __name__ == '__main__':
    run()
