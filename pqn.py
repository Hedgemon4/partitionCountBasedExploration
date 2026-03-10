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
    total_training_steps = 10
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


def loss(model, states, actions, targets):
    q_values = jax.vmap(model)(states)
    index = jnp.arange(q_values.shape[0])
    selected_q_values = q_values[index, actions]
    return jnp.mean((selected_q_values - targets) ** 2)


def run(args: Args):
    key = jax.random.PRNGKey(args.seed)

    # Environment Setup
    env, vmap_reset, vmap_step, env_params = make_env(args.environment)

    # Network Setup
    key, subkey = jax.random.split(key, 2)
    model = QNetwork(input_size=4, num_actions=2, key=subkey)
    optim = optax.adam(args.learning_rate)

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

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

    def train_step(carry, _):
        key, step_number = carry
        epsilon = epsilon_schedule(step_number)

        ### TODO: Step Environments
        def step(carry, _):
            key, env_state, state, action, q_value = carry

            # Step Environment
            key, subkey = jax.random.split(key, 2)
            next_state, env_state, reward, done, info = vmap_step(
                args.num_environments
            )(subkey, env_state, action)
            jax.debug.print("Single env info: {}", info)
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

        update_targets = jax.vmap(targets, in_axes=(0, None))(transitions, args.gamma)

        # Split network for eqx
        network_params, static = eqx.partition(model, eqx.is_array)

        def epoch(carry, _):
            rng, params, optimizer_state = carry
            next_rng, epoch_rng = jax.random.split(rng, 2)

            # Shuffle data
            def process_data(x, rng):
                x = x.reshape(-1, *x.shape[2:])
                x = jax.random.permutation(rng, x)
                return x.reshape(args.num_minibatches, -1, *x.shape[1:])

            # Using the same key will make sure data is shuffled in the same way across all fields
            minibatches = jax.tree_util.tree_map(
                lambda x: process_data(x, epoch_rng), transitions
            )
            targets = jax.tree_util.tree_map(process_data, update_targets, epoch_rng)

            # Compute the loss and update the model
            def update_model(carry, batch):
                model_params, optimizer_state = carry
                model = eqx.combine(model_params, static)
                mini_batch, targets = batch
                loss_value, grads = eqx.filter_value_and_grad(loss)(
                    model, mini_batch.state, mini_batch.action, targets
                )
                updates, optimizer_state = optim.update(
                    grads, optimizer_state, eqx.filter(model, eqx.is_array)
                )
                model = eqx.apply_updates(model, updates)
                params, _ = eqx.partition(model, eqx.is_array)
                return (params, optimizer_state), (loss_value)

            updates, metrics = jax.lax.scan(
                update_model, (params, optimizer_state), (minibatches, targets)
            )
            updated_params, updated_optimizer = updates
            jax.debug.print("Metrics: {}", metrics)
            return (next_rng, updated_params, updated_optimizer), metrics

        # Handle key split
        epoch_outs, epoch_metrics = jax.lax.scan(
            epoch, (subkey, network_params, opt_state), None, args.num_epochs
        )
        epoch_key, epoch_params, epoch_opt_state = epoch_outs
        step_number += 1

        return (epoch_key, step_number), (epoch_metrics, infos)

    # training_information, update_information = jax.lax.scan(train_step,(key, step_number), None, args.total_training_steps)
    # item1, item2 = train_step((key, step_number), None)
    item1, item2 = train_step((key, step_number), None)
    return (item1, item2)


if __name__ == "__main__":
    args = tyro.cli(Args)
    print("Starting Run")
    compiled_run = jax.jit(run)
    item1, item2 = compiled_run(args)
    metrics, info = item2
    print(info)
    print("Finished Run")
