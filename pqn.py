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
    total_training_steps = 5e7
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.1
    num_epochs = 2
    num_minibatches = 4
    gamma = 0.99


@chex.dataclass(frozen=True)
class Transition:
    state: chex.Array
    action: chex.Array
    reward: chex.Array
    q_value: chex.Array
    next_state: chex.Array
    next_action: chex.Array
    next_q_value: chex.Array
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


def loss(model, states, target):
    q_values = model(states)


def run(args: Args):
    key = jax.random.PRNGKey(args.seed)

    # Environment Setup
    env, vmap_reset, vmap_step, env_params = make_env(args.environment)

    # Network Setup
    key, subkey = jax.random.split(key, 2)
    model = QNetwork(input_size=4, num_actions=2, key=subkey)
    optim = optax.adam(args.learning_rate)

    optimizer_state = optim.init(eqx.filter(model, eqx.is_array))

    num_epsilon_decay_steps = (
        args.total_training_steps // args.num_environments // args.num_steps
    )

    # Epsilon Decay Setup
    epsilon_schedule = optax.linear_schedule(
        init_value=args.epsilon_start,
        end_value=args.epsilon_end,
        transition_steps=num_epsilon_decay_steps,
    )

    step_number = 0

    def train_step(key, step_number):
        epsilon = epsilon_schedule(step_number)

        ### TODO: Step Environments
        def step(carry, _):
            key, env_state, state, action, q_value = carry

            # Step Environment
            key, subkey = jax.random.split(key, 2)
            next_state, env_state, reward, done, info = vmap_step(
                args.num_environments
            )(subkey, env_state, action)

            # Get next actions
            next_q_values = jax.vmap(model)(next_state)
            key, subkey = jax.random.split(key, 2)
            next_action, next_q = epsilon_greedy(subkey, epsilon, next_q_values)

            transition = Transition(
                state=state,
                action=action,
                reward=reward,
                q_value=q_value,
                next_state=next_state,
                next_action=next_action,
                next_q_value=next_q,
                done=done,
            )

            return (key, env_state, next_state, next_action, next_q), (transition, info)

        # Reset Environment
        key, subkey = jax.random.split(key, 2)
        state, env_state = vmap_reset(args.num_environments)(subkey)

        # Get first actions
        q_values = jax.vmap(model)(state)
        key, subkey = jax.random.split(key, 2)
        action, selected_q = epsilon_greedy(subkey, args.epsilon_start, q_values)

        key, subkey = jax.random.split(key, 2)
        carry = (key, env_state, state, action, selected_q)

        final_outs, intermediate_values = jax.lax.scan(
            step, carry, None, args.num_steps
        )

        transitions, infos = intermediate_values

        # Compute Targets
        def targets(transition, gamma):
            return (
                transition.reward
                + (1 - transition.done) * gamma * transition.next_q_value
            )

        targets = jax.vmap(targets, in_axes=(0, None))(transitions, args.gamma)

        return transitions, infos

    return train_step(key, step_number)


if __name__ == "__main__":
    args = tyro.cli(Args)
    print("Starting Run")
    compiled_run = jax.jit(run)
    item1, item2 = compiled_run(args)
