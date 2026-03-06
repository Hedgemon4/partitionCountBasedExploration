from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tyro
import gymnax
import chex
from jaxtyping import Array, Float, Int, PyTree

from exploration import epsilon_greedy


@chex.dataclass(frozen=True)
class Args:
    seed = 1
    learning_rate = 1e-3
    environment = "CartPole-v1"
    num_environments = 32
    num_steps = 32
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.1


@chex.dataclass(frozen=True)
class Transition:
    state: chex.Array
    action: chex.Array
    reward: chex.Array
    next_state: chex.Array
    next_action: chex.Array
    done: chex.Array


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


def make_env(environment_name):
    env, env_params = gymnax.make(environment_name)

    vmap_reset = lambda num_envs: lambda random_key: jax.vmap(
        env.reset, in_axes=(0, None)
    )(jax.random.split(random_key, num_envs), env_params)
    vmap_step = lambda num_envs: lambda random_key, state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(random_key, num_envs), state, action, env_params)

    return env, vmap_reset, vmap_step, env_params


def run(args: Args):
    key = jax.random.PRNGKey(args.seed)

    # Environment Setup
    env, vmap_reset, vmap_step, env_params = make_env(args.environment)

    # Network Setup
    key, subkey = jax.random.split(key, 2)
    model = QNetwork(input_size=4, num_actions=2, key=subkey)
    optim = optax.adam(args.learning_rate)

    optimizer_state = optim.init(eqx.filter(model, eqx.is_array))

    # Reset Environment
    key, subkey = jax.random.split(key, 2)
    state, env_state = vmap_reset(args.num_environments)(subkey)

    ### TODO: Environment Step
    def step(key, state, env_state, model):
        # Get actions
        q_values = jax.vmap(model)(state)
        key, subkey = jax.random.split(key, 2)
        actions = epsilon_greedy(subkey, args.epsilon_start, q_values)

        # Step Environment
        key, subkey = jax.random.split(key, 2)
        next_state, env_state, reward, done, info = vmap_step(args.num_environments)(
            subkey, env_state, actions
        )
        return next_state, env_state, reward, done, info

    key, subkey = jax.random.split(key, 2)
    next_state, _, _, _, _ = step(subkey, state, env_state, model)

    return next_state
    ### TODO: Compute Loss and Update Model


if __name__ == "__main__":
    args = tyro.cli(Args)
    print("Starting Run")
    compiled_run = jax.jit(run)
    item = compiled_run(args)
    print(item)
