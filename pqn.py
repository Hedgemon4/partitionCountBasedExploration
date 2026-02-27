from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tyro
import gymnax
from jaxtyping import Array, Float, Int, PyTree


@dataclass
class Args:
    seed = 1
    learning_rate = 1e-3
    environment = "CartPole-v1"
    num_environments = 32


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


def make_env(environment_name, num_en):
    env, env_params = gymnax.make(environment_name)

    # vmap_step =

    return env, env_params


def run(args: Args):
    key = jax.random.PRNGKey(args.seed)

    # Environment Setup

    # Network Setup
    key, subkey = jax.random.split(key, 2)
    model = QNetwork(input_size=4, num_actions=2, key=subkey)
    optim = optax.adam(args.learning_rate)


if __name__ == "__main__":
    args = tyro.cli(Args)
    print("Starting Run")
    run(args)
