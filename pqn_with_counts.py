import dataclasses
import time
from pathlib import Path
from typing import Union, Annotated

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import gymnax
import chex
import yaml
from jax import Array

import configs.defaults as configs
from exploration import epsilon_greedy
from netwoks import QNetwork
from wrappers import FlattenObservationWrapper, LogWrapper

"""
PQN implementation based on https://github.com/mttga/purejaxql/blob/main/purejaxql/pqn_gymnax.py

Adds in one-to-many activations and count-based intrinsic rewards.
"""


@chex.dataclass(frozen=True)
class Transition:
    state: chex.Array
    action: chex.Array
    reward: chex.Array
    discrete_state: chex.Array
    intrinsic_reward: chex.Array
    selected_q_value: chex.Array
    all_q_values: chex.Array
    next_state: chex.Array
    next_action: chex.Array
    selected_next_q_value: chex.Array
    all_next_q_values: chex.Array
    done: chex.Array


@chex.dataclass(frozen=True)
class EMAMetrics:
    ema_alpha: float
    extrinsic_return_ema: float
    intrinsic_return_ema: float
    episode_length_ema: float


@chex.dataclass(frozen=True)
class IntrinsicRewardData:
    intrinsic_return: Array
    returned_intrinsic_return: Array


def make_env(environment_name, episode_length):
    env, env_params = gymnax.make(environment_name)
    if episode_length is not None:
        env_params = env_params.replace(max_steps_in_episode=episode_length)
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
    q_values, _ = jax.vmap(model)(states)
    index = jnp.arange(q_values.shape[0])
    selected_q_values = q_values[index, actions]
    return 0.5 * jnp.mean((selected_q_values - targets) ** 2), selected_q_values


def make_run(args):
    num_updates = int(args.total_time_steps // args.num_environments // args.num_steps)

    # Environment Setup
    episode_length = getattr(args, "episode_length", None)
    env, vmap_reset, vmap_step, env_params = make_env(args.environment, episode_length)
    ### TODO: Add support for non-gymnax environments

    input_size = int(env.observation_space(env_params).shape[0])
    num_actions = int(env.action_space(env_params).n)

    def run(key):
        # Network Setup
        key, subkey = jax.random.split(key, 2)
        initial_model = QNetwork(
            input_size=input_size,
            num_actions=num_actions,
            key=subkey,
            network_config=args.network,
        )

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
        initial_q_values, initial_discrete_state = jax.vmap(initial_model)(start_state)
        key, subkey = jax.random.split(key, 2)
        initial_action, initial_selected_q = epsilon_greedy(
            subkey, args.epsilon_start, initial_q_values
        )

        # Initialize structure for computing intrinsic return metrics
        initial_intrinsic_returns = IntrinsicRewardData(
            intrinsic_return=jnp.zeros_like(initial_selected_q),
            returned_intrinsic_return=jnp.zeros_like(initial_selected_q),
        )

        initial_env_carry = (
            key,
            start_env_state,
            start_state,
            initial_action,
            initial_discrete_state,
            initial_selected_q,
            initial_q_values,
            initial_intrinsic_returns,
        )

        episode_metrics = EMAMetrics(
            ema_alpha=2 / (args.num_episodes_for_average + 1),
            extrinsic_return_ema=jnp.nan,
            intrinsic_return_ema=jnp.nan,
            episode_length_ema=jnp.nan,
        )

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
                (
                    key,
                    step_env_state,
                    state,
                    action,
                    discrete_state,
                    selected_q_value,
                    all_q_values,
                    intrinsic_returns,
                ) = carry

                # Step Environment
                key, subkey = jax.random.split(key, 2)
                next_state, step_env_state, reward, done, info = vmap_step(
                    args.num_environments
                )(subkey, step_env_state, action)
                # Get next actions
                next_q_values, next_discrete_state = jax.vmap(model)(next_state)
                key, subkey = jax.random.split(key, 2)
                next_action, next_q = epsilon_greedy(subkey, epsilon, next_q_values)
                scaled_reward = reward * args.reward_scale

                # Compute intrinsic reward
                intrinsic_reward = jax.vmap(model.get_intrinsic_reward)(
                    discrete_state, action
                )

                # Update intrinsic return metrics
                new_intrinsic_return = (
                    intrinsic_returns.intrinsic_return + intrinsic_reward
                )
                updated_intrinsic_returns = IntrinsicRewardData(
                    intrinsic_return=new_intrinsic_return * (1 - done),
                    returned_intrinsic_return=intrinsic_returns.returned_intrinsic_return
                    * (1 - done)
                    + new_intrinsic_return * done,
                )

                # Add to info for logging
                info["returned_intrinsic_returns"] = (
                    updated_intrinsic_returns.returned_intrinsic_return
                )

                transition = Transition(
                    state=state,
                    action=action,
                    reward=scaled_reward,
                    discrete_state=discrete_state,
                    intrinsic_reward=intrinsic_reward,
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
                    next_discrete_state,
                    next_q,
                    next_q_values,
                    updated_intrinsic_returns,
                ), (
                    transition,
                    info,
                )

            final_env_carry, intermediate_values = jax.lax.scan(
                step, env_carry, None, args.num_steps
            )
            env_step += args.num_steps * args.num_environments

            transitions, infos = intermediate_values
            flat_states = transitions.discrete_state.reshape(
                -1, *transitions.discrete_state.shape[-2:]
            )
            flat_actions = transitions.action.reshape(-1)
            model = model.update_counts(flat_states, flat_actions)

            # Compute Targets
            if args.lambda_returns:
                # TODO: These targets still might be wrong
                def lambda_targets(carry, transition):
                    target, next_q = carry
                    updated_target = (
                        transition.reward + (args.beta * transition.intrinsic_reward)
                    ) + (
                        (1 - transition.done)
                        * args.gamma
                        * (args.lam * target + (1 - args.lam) * next_q)
                    )
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
                last_q_value = last_q_value * (
                    1 - transitions.done[-1]
                )  # If done, then no q value
                initial_return = (
                    transitions.reward[-1]
                    + (args.beta * transitions.intrinsic_reward[-1])
                    + args.gamma * last_q_value
                )
                carry = (initial_return, last_q_value)
                final_target_carry, targets = jax.lax.scan(
                    lambda_targets,
                    carry,
                    jax.tree_util.tree_map(lambda x: x[:-1], transitions),
                    reverse=True,
                )
                update_targets = jnp.concatenate((targets, initial_return[np.newaxis]))
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

            metrics = {
                "env_step": env_step,
                "update_steps": step_number,
                "td_loss": epoch_loss.mean(),
                "q_values": epoch_q_values.mean(),
            }
            metrics.update({k: v.mean() for k, v in infos.items()})

            # Compute EMA of episode returns and lengths

            is_done = infos["returned_episode"]
            extrinsic_episode_returns = infos["returned_episode_returns"]
            intrinsic_episode_returns = infos["returned_intrinsic_returns"]
            episode_lengths = infos["returned_episode_lengths"]
            num_dones = is_done.sum()

            def update_ema(current_ema, new_value, num_dones, alpha):
                effective_alpha = 1 - (1 - alpha) ** num_dones
                # If current_ema is NaN, it's the first episode; use the new_value directly
                return jnp.where(
                    num_dones > 0,
                    jnp.where(
                        jnp.isnan(current_ema),
                        new_value,
                        current_ema + effective_alpha * (new_value - current_ema),
                    ),
                    current_ema,
                )

            mean_extrinsic_episode_return = jnp.sum(
                is_done * extrinsic_episode_returns
            ) / jnp.maximum(num_dones, 1)
            updated_extrinsic_return_ema = update_ema(
                train_episode_metrics.extrinsic_return_ema,
                mean_extrinsic_episode_return,
                num_dones,
                train_episode_metrics.ema_alpha,
            )

            mean_intrinsic_episode_return = jnp.sum(
                is_done * intrinsic_episode_returns
            ) / jnp.maximum(num_dones, 1)
            updated_intrinsic_return_ema = update_ema(
                train_episode_metrics.intrinsic_return_ema,
                mean_intrinsic_episode_return,
                num_dones,
                train_episode_metrics.ema_alpha,
            )

            mean_episode_length = jnp.sum(is_done * episode_lengths) / jnp.maximum(
                num_dones, 1
            )
            updated_episode_lengths_ema = update_ema(
                train_episode_metrics.episode_length_ema,
                mean_episode_length,
                num_dones,
                train_episode_metrics.ema_alpha,
            )

            # Update train episode metrics
            updated_episode_metrics = EMAMetrics(
                ema_alpha=train_episode_metrics.ema_alpha,
                extrinsic_return_ema=updated_extrinsic_return_ema,
                intrinsic_return_ema=updated_intrinsic_return_ema,
                episode_length_ema=updated_episode_lengths_ema,
            )

            metrics["extrinsic_return_ema"] = updated_extrinsic_return_ema
            metrics["length_ema"] = updated_episode_lengths_ema
            metrics["intrinsic_return_ema"] = updated_intrinsic_return_ema

            return (
                epoch_key,
                step_number,
                env_step,
                final_env_carry,
                epoch_params,
                epoch_opt_state,
                updated_episode_metrics,
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
        final_carry, metrics = jax.lax.scan(
            train_step, training_carry, None, num_updates
        )
        final_model = eqx.combine(final_carry[4], static)

        return final_model.counts, metrics

    return run


ConfigOptions = Union[
    Annotated[
        configs.CartPoleWithIntrinsicRewardsConfig,
        tyro.conf.subcommand(name="cartpole"),
    ],
    Annotated[
        configs.MountainCarWithIntrinsicRewardsConfig,
        tyro.conf.subcommand(name="mountaincar"),
    ],
]

if __name__ == "__main__":
    args = tyro.cli(
        ConfigOptions,
        default=configs.CartPoleWithIntrinsicRewardsConfig(),
        config=(tyro.conf.CascadeSubcommandArgs,),
    )

    save_path = Path("data", args.output_folder_name)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    config_path = save_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(dataclasses.asdict(args), f)

    print("Starting Run")
    rng = jax.random.PRNGKey(args.seed)

    t0 = time.time()
    rngs = jax.random.split(rng, args.num_seeds)
    compiled_run = jax.jit(jax.vmap(make_run(args)))
    counts, metrics = jax.block_until_ready(compiled_run(rngs))
    print(f"Total time: {time.time() - t0}")

    metrics_path = save_path / "metrics.npz"
    np.savez(metrics_path, **metrics)

    counts_path = save_path / "counts.npz"
    np.save(counts_path, counts)
    print("Finished Run")
