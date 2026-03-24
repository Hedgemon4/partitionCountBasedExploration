import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import gymnax
import chex

from exploration import epsilon_greedy
from wrappers import FlattenObservationWrapper, LogWrapper

"""
PQN implementation based on https://github.com/mttga/purejaxql/blob/main/purejaxql/pqn_gymnax.py
"""


@chex.dataclass(frozen=True)
class Args:
    seed: int = 0
    num_seeds: int = 30
    initial_learning_rate: float = 0.0001
    final_learning_rate: float = 1e-20
    environment: str = "CartPole-v1"
    num_environments: int = 32
    num_steps: int = 64
    total_time_steps: int = 5e5
    epsilon_start: float = 1.0
    epsilon_end: float = 0.2
    epsilon_decay: float = 0.2
    num_epochs: int = 4
    num_minibatches: int = 16
    hidden_size: int = 256
    gamma: float = 0.99
    lambda_returns: bool = True
    lam: float = 0.95
    max_grad_norm: float = 10
    reward_scale: float = 0.1
    num_episodes_for_average: int = 30
    learnable_norm_params: bool = True
    sarsa_returns: bool = False
    metrics_file_name: str = "pqn_original_cartpole_default_params.npz"


@chex.dataclass(frozen=True)
class Transition:
    state: chex.Array
    action: chex.Array
    reward: chex.Array
    selected_q_value: chex.Array
    all_q_values: chex.Array
    next_state: chex.Array
    next_action: chex.Array
    selected_next_q_value: chex.Array
    all_next_q_values: chex.Array
    done: chex.Array


class QNetwork(eqx.Module):
    layers: list

    def __init__(self, input_size, num_actions, hidden_size, key):
        key1, key2, key3 = jax.random.split(key, 3)

        ### TODO: Might need to transpose for atari
        self.layers = [
            eqx.nn.Linear(in_features=input_size, out_features=hidden_size, key=key1),
            eqx.nn.LayerNorm(
                hidden_size,
                use_weight=args.learnable_norm_params,
                use_bias=args.learnable_norm_params,
            ),
            jax.nn.relu,
            eqx.nn.Linear(in_features=hidden_size, out_features=hidden_size, key=key2),
            eqx.nn.LayerNorm(
                hidden_size,
                use_weight=args.learnable_norm_params,
                use_bias=args.learnable_norm_params,
            ),
            jax.nn.relu,
            eqx.nn.Linear(in_features=hidden_size, out_features=num_actions, key=key3),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def init_linear_weights_lecun(model, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)

    def get_weights(m):
        return [l.weight for l in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(l)]

    def get_biases(m):
        return [l.bias for l in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(l)]

    weights = get_weights(model)
    biases = get_biases(model)
    keys = jax.random.split(key, len(weights))

    # Use JAX's built-in truncated normal LeCun initializer
    init_fn = jax.nn.initializers.lecun_normal(in_axis=-1, out_axis=-2)
    new_weights = [init_fn(k, w.shape, w.dtype) for w, k in zip(weights, keys)]
    new_biases = [jnp.zeros_like(b) if b is not None else None for b in biases]

    model = eqx.tree_at(get_weights, model, new_weights)
    model = eqx.tree_at(get_biases, model, new_biases)

    return model


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
    return 0.5 * jnp.mean((selected_q_values - targets) ** 2), selected_q_values


def make_run(args: Args):
    num_updates = int(args.total_time_steps // args.num_environments // args.num_steps)

    # Environment Setup
    env, vmap_reset, vmap_step, env_params = make_env(args.environment)
    ### TODO: Add support for non-gymnax environments

    input_size = int(env.observation_space(env_params).shape[0])
    num_actions = int(env.action_space(env_params).n)

    def run(key):
        # Network Setup
        key, subkey = jax.random.split(key, 2)
        initial_model = QNetwork(
            input_size=input_size,
            num_actions=num_actions,
            hidden_size=args.hidden_size,
            key=subkey,
        )

        key, subkey = jax.random.split(key, 2)
        initial_model = init_linear_weights_lecun(initial_model, subkey)

        optim = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.radam(
                args.initial_learning_rate
                if args.initial_learning_rate == args.final_learning_rate
                else optax.linear_schedule(
                    init_value=args.initial_learning_rate,
                    end_value=args.final_learning_rate,
                    transition_steps=num_updates
                    * args.num_epochs
                    * args.num_minibatches,
                )
            ),
        )

        initial_opt_state = optim.init(eqx.filter(initial_model, eqx.is_array))

        # Epsilon Decay Setup
        epsilon_schedule = optax.linear_schedule(
            init_value=args.epsilon_start,
            end_value=args.epsilon_end,
            transition_steps=int(num_updates * args.epsilon_decay),
        )

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
            initial_q_values,
        )

        episode_return_ema = 0.0
        episode_length_ema = 0.0
        ema_alpha = 2 / (args.num_episodes_for_average + 1)
        episode_metrics = (episode_return_ema, episode_length_ema)

        step_number = 0
        env_step = 0

        # Split network for eqx
        dynamic_params, static = eqx.partition(initial_model, eqx.is_array)

        def train_step(carry, _):
            (
                key,
                step_number,
                env_step,
                env_carry,
                carry_params,
                carry_opt_state,
                train_episode_metrics,
            ) = carry
            epsilon = epsilon_schedule(step_number)
            model = eqx.combine(carry_params, static)

            # Step env
            def step(carry, _):
                key, step_env_state, state, action, selected_q_value, all_q_values = (
                    carry
                )

                # Step Environment
                key, subkey = jax.random.split(key, 2)
                next_state, step_env_state, reward, done, info = vmap_step(
                    args.num_environments
                )(subkey, step_env_state, action)
                # Get next actions
                next_q_values = jax.vmap(model)(next_state)
                key, subkey = jax.random.split(key, 2)
                next_action, next_q = epsilon_greedy(subkey, epsilon, next_q_values)
                scaled_reward = reward * args.reward_scale

                transition = Transition(
                    state=state,
                    action=action,
                    reward=scaled_reward,
                    selected_q_value=selected_q_value,
                    all_q_values=all_q_values,
                    next_state=next_state,
                    next_action=next_action,
                    selected_next_q_value=next_q,
                    all_next_q_values=next_q_values,
                    done=done,
                )

                return (
                    key,
                    step_env_state,
                    next_state,
                    next_action,
                    next_q,
                    next_q_values,
                ), (
                    transition,
                    info,
                )

            final_env_carry, intermediate_values = jax.lax.scan(
                step, env_carry, None, args.num_steps
            )
            env_step += args.num_steps * args.num_environments

            transitions, infos = intermediate_values

            # Compute Targets
            if args.lambda_returns:
                # TODO: These targets still might be wrong
                def lambda_targets(carry, transition):
                    target, next_q = carry
                    updated_target = transition.reward + (
                        1 - transition.done
                    ) * args.gamma * (args.lam * target + (1 - args.lam) * next_q)
                    next_q = (
                        transition.selected_q_value
                        if args.sarsa_returns
                        else jnp.max(transition.all_q_values, axis=-1)
                    )
                    return (updated_target, next_q), updated_target

                # Want to compute the targets. Each target will have the final q value in it, so we can start with that
                last_q_value = (
                    transitions.selected_next_q_value[-1, :]
                    if args.sarsa_returns
                    else jnp.max(transitions.all_next_q_values[-1, :], axis=-1)
                )
                last_q_value = last_q_value * (1 - transitions.done[-1])  # If done, then no q value
                initial_return = transitions.reward[-1] + args.gamma * last_q_value
                carry = (initial_return, last_q_value)
                final_target_carry, targets = jax.lax.scan(
                    lambda_targets, carry, jax.tree_util.tree_map(lambda x: x[:-1], transitions), reverse=True
                )
                update_targets = jnp.concatenate((targets, initial_return[np.newaxis]))
                # update_targets = targets
            else:

                def targets(transition, gamma):
                    return (
                        transition.reward
                        + (1 - transition.done)
                        * gamma
                        * transition.selected_next_q_value
                    )

                update_targets = jax.vmap(targets, in_axes=(0, None))(
                    transitions, args.gamma
                )

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
                epoch, (subkey, network_params, carry_opt_state), None, args.num_epochs
            )
            epoch_key, epoch_params, epoch_opt_state = epoch_outs
            step_number += 1

            ### TODO: Compute episode return metrics based on discussion with Mike

            metrics = {
                "env_step": env_step,
                "update_steps": step_number,
                "td_loss": epoch_loss.mean(),
                "q_values": epoch_q_values.mean(),
            }
            metrics.update({k: v.mean() for k, v in infos.items()})

            # Compute EMA of episode returns and lengths

            is_done = infos["returned_episode"]
            episode_returns = infos["returned_episode_returns"]
            episode_lengths = infos["returned_episode_lengths"]
            num_dones = is_done.sum()

            returns_ema, lengths_ema = train_episode_metrics

            mean_episode_return = jnp.sum(is_done * episode_returns) / jnp.maximum(
                num_dones, 1
            )
            effective_alpha = 1 - (1 - ema_alpha) ** num_dones
            updated_returns_ema = jnp.where(
                num_dones > 0,
                returns_ema + effective_alpha * (mean_episode_return - returns_ema),
                returns_ema,
            )

            mean_episode_length = jnp.sum(is_done * episode_lengths) / jnp.maximum(
                num_dones, 1
            )
            updated_episode_lengths_ema = jnp.where(
                num_dones > 0,
                lengths_ema + effective_alpha * (mean_episode_length - lengths_ema),
                lengths_ema,
            )

            metrics["moving_avg_return"] = updated_returns_ema
            metrics["moving_avg_length"] = updated_episode_lengths_ema

            return (
                epoch_key,
                step_number,
                env_step,
                final_env_carry,
                epoch_params,
                epoch_opt_state,
                (updated_returns_ema, updated_episode_lengths_ema),
            ), metrics

        training_carry = (
            key,
            step_number,
            env_step,
            initial_env_carry,
            dynamic_params,
            initial_opt_state,
            episode_metrics,
        )
        return jax.lax.scan(train_step, training_carry, None, num_updates)

    return run


if __name__ == "__main__":
    args = tyro.cli(Args)
    print("Starting Run")
    rng = jax.random.PRNGKey(args.seed)

    t0 = time.time()
    rngs = jax.random.split(rng, args.num_seeds)
    compiled_run = jax.jit(jax.vmap(make_run(args)))
    item1, metrics = jax.block_until_ready(compiled_run(rngs))
    print(f"Took: {time.time() - t0}")

    ### TODO: Add better logging of results
    np.savez("data/" + args.metrics_file_name, **metrics)
    print("Finished Run")
