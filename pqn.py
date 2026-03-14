import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tyro
import gymnax
import chex
from jaxtyping import Array, Float, Int, PyTree

from exploration import epsilon_greedy
from wrappers import FlattenObservationWrapper, LogWrapper


@chex.dataclass(frozen=True)
class Args:
    seed = 23
    learning_rate = 1e-4
    environment = "CartPole-v1"
    num_environments = 32
    num_steps = 64
    total_time_steps = 5e5
    epsilon_start = 1.0
    epsilon_end = 0.2
    epsilon_decay = 0.2
    num_epochs = 4
    num_minibatches = 16
    hidden_size = 256
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

    def __init__(self, input_size, num_actions, hidden_size, key):
        key1, key2, key3 = jax.random.split(key, 3)

        ### TODO: Might need to transpose for atari
        self.layers = [
            eqx.nn.Linear(in_features=input_size, out_features=hidden_size, key=key1),
            eqx.nn.LayerNorm(hidden_size),
            jax.nn.relu,
            eqx.nn.Linear(in_features=hidden_size, out_features=hidden_size, key=key2),
            eqx.nn.LayerNorm(hidden_size),
            jax.nn.relu,
            eqx.nn.Linear(in_features=hidden_size, out_features=num_actions, key=key3),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def make_env(environment_name):
    env, env_params = gymnax.make(environment_name)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
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
    return jnp.mean((selected_q_values - targets) ** 2), selected_q_values


def run(args: Args):
    key = jax.random.PRNGKey(args.seed)

    # Environment Setup
    env, vmap_reset, vmap_step, env_params = make_env(args.environment)

    # Network Setup
    key, subkey = jax.random.split(key, 2)
    initial_model = QNetwork(
        input_size=4, num_actions=2, hidden_size=args.hidden_size, key=subkey
    )
    optim = optax.adam(args.learning_rate)

    opt_state = optim.init(eqx.filter(initial_model, eqx.is_array))

    num_updates = int(args.total_time_steps // args.num_environments // args.num_steps)

    # Epsilon Decay Setup
    epsilon_schedule = optax.linear_schedule(
        init_value=args.epsilon_start,
        end_value=args.epsilon_end,
        transition_steps=num_updates,
    )

    ### TODO: Add learning rate decay as well

    # Reset Environment
    key, subkey = jax.random.split(key, 2)
    start_state, start_env_state = vmap_reset(args.num_environments)(subkey)

    # Get first actions
    initial_q_values = jax.vmap(initial_model)(start_state)
    key, subkey = jax.random.split(key, 2)
    initial_action, initial_selected_q = epsilon_greedy(
        subkey, args.epsilon_start, initial_q_values
    )
    initial_env_carry = (
        key,
        start_env_state,
        start_state,
        initial_action,
        initial_selected_q,
    )

    step_number = 0
    env_step = 0

    # Split network for eqx
    dynamic_params, static = eqx.partition(initial_model, eqx.is_array)

    def train_step(carry, _):
        key, step_number, env_step, env_carry, carry_params = carry
        epsilon = epsilon_schedule(step_number)
        model = eqx.combine(carry_params, static)

        # Step env
        def step(carry, _):
            key, step_env_state, state, action, q_value = carry

            # Step Environment
            key, subkey = jax.random.split(key, 2)
            next_state, step_env_state, reward, done, info = vmap_step(
                args.num_environments
            )(subkey, step_env_state, action)
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

            return (key, step_env_state, next_state, next_action, next_q), (
                transition,
                info,
            )

        final_env_carry, intermediate_values = jax.lax.scan(
            step, env_carry, None, args.num_steps
        )
        env_step += args.num_steps * args.num_environments

        transitions, infos = intermediate_values

        # Compute Targets
        def targets(transition, gamma):
            return (
                transition.reward
                + (1 - transition.done) * gamma * transition.next_q_value
            )

        update_targets = jax.vmap(targets, in_axes=(0, None))(transitions, args.gamma)

        # Split network for eqx
        network_params, _ = eqx.partition(model, eqx.is_array)

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
            targets = jax.tree_util.tree_map(
                lambda x: process_data(x, epoch_rng), update_targets
            )

            # Compute the loss and update the model
            def update_model(carry, batch):
                model_params, optimizer_state = carry
                model = eqx.combine(model_params, static)
                mini_batch, targets = batch
                (loss_value, loss_q_values), grads = eqx.filter_value_and_grad(
                    loss, has_aux=True
                )(model, mini_batch.state, mini_batch.action, targets)
                updates, optimizer_state = optim.update(
                    grads, optimizer_state, eqx.filter(model, eqx.is_array)
                )
                model = eqx.apply_updates(model, updates)
                params, _ = eqx.partition(model, eqx.is_array)
                return (params, optimizer_state), (loss_value, loss_q_values)

            updates, metrics = jax.lax.scan(
                update_model, (params, optimizer_state), (minibatches, targets)
            )
            updated_params, updated_optimizer = updates
            return (next_rng, updated_params, updated_optimizer), metrics

        # Handle key split
        epoch_outs, (epoch_loss, epoch_q_values) = jax.lax.scan(
            epoch, (subkey, network_params, opt_state), None, args.num_epochs
        )
        epoch_key, epoch_params, epoch_opt_state = epoch_outs
        step_number += 1
        metrics = {
            "env_step": env_step,
            "update_steps": step_number,
            "td_loss": epoch_loss.mean(),
            "q_values": epoch_q_values.mean(),
        }
        metrics.update({k: v.mean() for k, v in infos.items()})

        # This just makes sure that the lengths and returns are only averaged for episodes which ended
        for k, v in infos.items():
            if k == "returned_episode_returns" or k == "returned_episode_lengths":
                mask = infos["returned_episode"]
                sum_val = jnp.sum(v * mask)
                count = jnp.sum(mask)
                metrics["test_" + k] = jnp.where(count > 0, sum_val / count, 0.0)

        return (
            epoch_key,
            step_number,
            env_step,
            final_env_carry,
            epoch_params,
        ), metrics

    training_carry = (key, step_number, env_step, initial_env_carry, dynamic_params)
    item1, item2 = jax.lax.scan(train_step, training_carry, None, num_updates)
    return (item1, item2)


if __name__ == "__main__":
    args = tyro.cli(Args)
    print("Starting Run")
    compiled_run = jax.jit(run)
    item1, item2 = compiled_run(args)
    metrics = item2

    ### TODO: Add better logging of results
    print(metrics["test_returned_episode_returns"])
    print(metrics["test_returned_episode_lengths"])
    print(metrics.keys())
    print("Finished Run")
