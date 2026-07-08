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
import chex
import yaml
from gymnasium.vector import AutoresetMode
from jax import Array

import configs.defaults as configs
from ale_py.vector_env import AtariVectorEnv
from exploration import epsilon_greedy
from helper_functions import update_ema
from netwoks import make_network
from wrappers import (
    ALEGymnaxWrapperXLA,
    ALEGymnaxWrapperStandard,
)

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
    next_continuous_state: chex.Array
    next_action: chex.Array
    selected_next_q_value: chex.Array
    all_next_q_values: chex.Array
    done: chex.Array


@chex.dataclass(frozen=True)
class EMAMetrics:
    ema_alpha: float
    extrinsic_return_per_game_ema: float
    clipped_extrinsic_return_per_game_ema: float
    intrinsic_return_per_game_ema: float
    episode_length_ema: float


@chex.dataclass(frozen=True)
class IntrinsicRewardData:
    game_intrinsic_return: Array
    returned_game_intrinsic_return: Array


def make_env(args):
    environment_name = args.environment

    # Check to see if the xla interface is available
    try:
        test_env = AtariVectorEnv(
            environment_name,
            num_envs=1,
            autoreset_mode=AutoresetMode.SAME_STEP,
            stack_num=args.framestack,
        )
        test_env.xla()
        del test_env
        xla_available = True
    except (AttributeError, RuntimeError):
        xla_available = False
    if xla_available:
        print("Using ale xla interface")
        wrapper = ALEGymnaxWrapperXLA
    elif not args.force_xla:
        print("Using ale default interface")
        wrapper = ALEGymnaxWrapperStandard
    else:
        raise ValueError("XLA interface not available, but force_xla is set to True")
    env, env_params = (
        wrapper(
            env_name=environment_name,
            num_envs=args.num_environments,
            seed=args.seed,
            stack_num=args.framestack,
            life_loss_info=args.life_loss_info,
            num_threads=args.num_env_threads,
        ),
        None,
    )
    return env, env_params


def compute_save_indices(num_updates, timesteps_per_update, interval):
    """Update indices whose end-of-step timestep is closest to each interval
    boundary. Returns a de-duplicated, in-order list of update indices."""
    if interval is None or interval <= 0:
        return []
    interval = int(interval)
    indices = []
    for boundary in range(interval, num_updates * timesteps_per_update + 1, interval):
        idx = min(round(boundary / timesteps_per_update) - 1, num_updates - 1)
        if idx not in indices:
            indices.append(idx)
    return indices


def make_run(args):
    num_updates = int(args.total_time_steps // args.num_environments // args.num_steps)
    timesteps_per_update = int(args.num_steps * args.num_environments)

    # Environment Setup
    env, env_params = make_env(args)

    input_size = int(env.observation_space(env_params).shape[0])
    num_actions = int(env.action_space(env_params).n)

    def run(key):
        # Network Setup
        key, subkey = jax.random.split(key, 2)
        initial_model = make_network(
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
        start_state, start_env_state = env.reset(subkey)

        # Get first actions
        initial_outputs = jax.vmap(initial_model)(start_state)
        initial_q_values = initial_outputs[0]
        initial_discrete_state = initial_outputs[1]
        key, subkey = jax.random.split(key, 2)
        initial_action, initial_selected_q = epsilon_greedy(
            subkey, args.epsilon_start, initial_q_values
        )

        # Initialize structure for computing intrinsic return metrics
        initial_intrinsic_returns = IntrinsicRewardData(
            game_intrinsic_return=jnp.zeros_like(initial_selected_q),
            returned_game_intrinsic_return=jnp.zeros_like(initial_selected_q),
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
            extrinsic_return_per_game_ema=jnp.nan,
            clipped_extrinsic_return_per_game_ema=jnp.nan,
            intrinsic_return_per_game_ema=jnp.nan,
            episode_length_ema=jnp.nan,
        )

        step_number = 0
        env_step = 0

        # Split network for eqx
        dynamic_params, static = eqx.partition(initial_model, eqx.is_array)

        def train_step(carry, save_slot):
            (
                key,
                step_number,
                env_step,
                env_carry,
                carry_params,
                carry_opt_state,
                train_episode_metrics,
                counts_buffer,
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
                next_state, step_env_state, reward, done, info = env.step(
                    subkey, step_env_state, action
                )
                # Get next actions
                model_outs = jax.vmap(model)(next_state)
                next_q_values = model_outs[0]
                next_discrete_state = model_outs[1]
                # Continuous count-layer FTA features of next_state — auxiliary target.
                next_continuous_state = model_outs[2]
                key, subkey = jax.random.split(key, 2)
                next_action, next_q = epsilon_greedy(subkey, epsilon, next_q_values)
                scaled_reward = reward * args.reward_scale

                # Compute intrinsic reward
                intrinsic_reward = jax.vmap(model.get_intrinsic_reward)(
                    discrete_state, action
                )

                # Update intrinsic return metrics. The whole game is one episode,
                # so `done` is the game-over boundary.
                new_game_intrinsic_return = (
                    intrinsic_returns.game_intrinsic_return + intrinsic_reward
                )
                updated_intrinsic_returns = IntrinsicRewardData(
                    game_intrinsic_return=new_game_intrinsic_return * (1 - done),
                    returned_game_intrinsic_return=(
                        intrinsic_returns.returned_game_intrinsic_return * (1 - done)
                        + new_game_intrinsic_return * done
                    ),
                )

                # Add to info for logging
                info["returned_game_intrinsic_returns"] = (
                    updated_intrinsic_returns.returned_game_intrinsic_return
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
                    next_continuous_state=next_continuous_state,
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
            flat_discrete_state = transitions.discrete_state.reshape(
                -1, *transitions.discrete_state.shape[-2:]
            )
            flat_discrete_actions = transitions.action.reshape(-1)
            model = model.update_counts(flat_discrete_state, flat_discrete_actions)

            # Compute Targets
            if args.lambda_returns:

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
                initial_next_q = (
                    transitions.selected_q_value[-1, :]
                    if args.sarsa_returns
                    else jnp.max(transitions.all_q_values[-1, :], axis=-1)
                )
                carry = (initial_return, initial_next_q)
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
                    (loss_value, (loss_q_values, losses)), grads = (
                        eqx.filter_value_and_grad(type(model).loss, has_aux=True)(
                            model, mini_batch, targets
                        )
                    )
                    updates, optimizer_state = optim.update(
                        grads, optimizer_state, model_params
                    )
                    model = eqx.apply_updates(model, updates)
                    params, _ = eqx.partition(model, eqx.is_array)
                    return (params, optimizer_state), (loss_q_values, losses)

                updates, metrics = jax.lax.scan(
                    update_model, (params, optimizer_state), (minibatches, targets)
                )
                updated_params, updated_optimizer = updates
                return (next_rng, updated_params, updated_optimizer), metrics

            # Handle key split
            epoch_outs, (epoch_q_values, epoch_losses) = jax.lax.scan(
                epoch, (subkey, network_params, carry_opt_state), None, args.num_epochs
            )
            epoch_key, epoch_params, epoch_opt_state = epoch_outs
            step_number += 1

            metrics = {
                "env_step": env_step,
                "update_steps": step_number,
                "q_values": epoch_q_values.mean(),
            }

            metrics.update({f"loss_{k}": v.mean() for k, v in epoch_losses.items()})
            metrics.update({k: v.mean() for k, v in infos.items()})

            # Compute EMA of episode returns and lengths

            is_done = infos["returned_episode"]
            num_dones = is_done.sum()

            extrinsic_episode_returns = infos["returned_episode_returns"]
            clipped_episode_returns = infos["clipped_returned_episode_returns"]
            game_intrinsic_returns = infos["returned_game_intrinsic_returns"]
            episode_lengths = infos["returned_episode_lengths"]

            mean_extrinsic_per_game = jnp.sum(
                is_done * extrinsic_episode_returns
            ) / jnp.maximum(num_dones, 1)
            updated_extrinsic_per_game_ema = update_ema(
                train_episode_metrics.extrinsic_return_per_game_ema,
                mean_extrinsic_per_game,
                num_dones,
                train_episode_metrics.ema_alpha,
            )

            mean_clipped_per_game = jnp.sum(
                is_done * clipped_episode_returns
            ) / jnp.maximum(num_dones, 1)
            updated_clipped_per_game_ema = update_ema(
                train_episode_metrics.clipped_extrinsic_return_per_game_ema,
                mean_clipped_per_game,
                num_dones,
                train_episode_metrics.ema_alpha,
            )

            mean_intrinsic_per_game = jnp.sum(
                is_done * game_intrinsic_returns
            ) / jnp.maximum(num_dones, 1)
            updated_intrinsic_per_game_ema = update_ema(
                train_episode_metrics.intrinsic_return_per_game_ema,
                mean_intrinsic_per_game,
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
                extrinsic_return_per_game_ema=updated_extrinsic_per_game_ema,
                clipped_extrinsic_return_per_game_ema=updated_clipped_per_game_ema,
                intrinsic_return_per_game_ema=updated_intrinsic_per_game_ema,
                episode_length_ema=updated_episode_lengths_ema,
            )

            metrics["extrinsic_return_per_game_ema"] = updated_extrinsic_per_game_ema
            metrics["clipped_extrinsic_return_per_game_ema"] = (
                updated_clipped_per_game_ema
            )
            metrics["intrinsic_return_per_game_ema"] = updated_intrinsic_per_game_ema
            metrics["length_ema"] = updated_episode_lengths_ema

            # Write this update step's counts into the save buffer only if this
            # step is a designated save point (save_slot >= 0). Avoids stacking
            # a snapshot per update step (which would be ~num_updates × counts).
            step_model = eqx.combine(epoch_params, static)
            step_counts = step_model.counts

            counts_buffer = jax.lax.cond(
                save_slot >= 0,
                lambda buf: buf.at[save_slot].set(step_counts),
                lambda buf: buf,
                counts_buffer,
            )

            return (
                epoch_key,
                step_number,
                env_step,
                final_env_carry,
                epoch_params,
                epoch_opt_state,
                updated_episode_metrics,
                counts_buffer,
            ), metrics

        # Decide the count-snapshot save points up front (fully determined by
        # static config). save_slots[i] is the buffer slot to write at update i,
        # or -1 if update i is not a save point. The buffer holds only the
        # snapshots we intend to persist, instead of one per update step.
        save_indices = compute_save_indices(
            num_updates,
            timesteps_per_update,
            getattr(args, "count_save_timestep_interval", 0),
        )
        num_saves = len(save_indices)
        save_slots = np.full(num_updates, -1, dtype=np.int32)
        for slot, idx in enumerate(save_indices):
            save_slots[idx] = slot
        save_slots = jnp.asarray(save_slots)

        counts_buffer = jnp.zeros(
            (max(num_saves, 1), *initial_model.counts.shape),
            dtype=initial_model.counts.dtype,
        )

        training_carry = (
            key,
            step_number,
            env_step,
            initial_env_carry,
            dynamic_params,
            initial_opt_state,
            episode_metrics,
            counts_buffer,
        )

        # Single scan over all update steps. save_slots (length num_updates)
        # drives which steps write a count snapshot into counts_buffer; only the
        # filled buffer (num_saves snapshots) survives in the final carry.
        final_carry, metrics = jax.lax.scan(train_step, training_carry, save_slots)

        final_model = eqx.combine(final_carry[4], static)
        counts_buffer = final_carry[7]

        # counts_buffer : (num_saves, *counts_shape) — only the snapshots at the
        # configured save points, in the same order as compute_save_indices(...).
        return (
            final_model.counts,
            counts_buffer,
            metrics,
        )

    return run


ConfigOptions = Union[
    Annotated[
        configs.AtariCountsConfig,
        tyro.conf.subcommand(name="default"),
    ],
    Annotated[
        configs.AtariCountsOneBlockConfig,
        tyro.conf.subcommand(name="one-block"),
    ]
]

if __name__ == "__main__":
    args = tyro.cli(
        ConfigOptions,
        default=configs.AtariCountsConfig(),
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

    num_updates = int(args.total_time_steps // args.num_environments // args.num_steps)
    timesteps_per_update = int(args.num_steps * args.num_environments)

    t0 = time.time()
    if args.num_seeds > 1:
        raise NotImplementedError("Multiple seeds are not currently supported")
    compiled_run = jax.jit(make_run(args))
    (
        counts,
        counts_buffer,
        metrics,
    ) = jax.block_until_ready(compiled_run(rng))
    print(f"Total time: {time.time() - t0}")

    metrics_path = save_path / "metrics.npz"
    np.savez(metrics_path, **metrics)

    # Save final counts
    counts_path = save_path / "final_counts.npy"
    np.save(counts_path, counts)

    # Save the count snapshots collected at each interval boundary. counts_buffer
    # has shape (num_saves, *counts_shape); slot i corresponds to save_indices[i],
    # using the same arithmetic make_run used to fill the buffer.
    save_indices = compute_save_indices(
        num_updates,
        timesteps_per_update,
        getattr(args, "count_save_timestep_interval", 0),
    )
    if save_indices:
        counts_path = save_path / "counts"
        counts_path.mkdir(exist_ok=True)

        for slot, idx in enumerate(save_indices):
            actual_timestep = (idx + 1) * timesteps_per_update
            np.save(
                counts_path / f"counts_timestep_{actual_timestep}.npy",
                counts_buffer[slot],
            )

    print("Finished Run")
