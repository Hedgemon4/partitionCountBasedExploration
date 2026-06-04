import dataclasses
import time
from pathlib import Path
from typing import Union, Annotated

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import yaml
from ale_py import AtariVectorEnv
from gymnasium.vector import AutoresetMode

import configs.defaults as configs
from exploration import epsilon_greedy
from helper_functions import update_ema
from netwoks import QNetworkCNN
from wrappers import ALEGymnaxWrapperXLA, ALEGymnaxWrapperStandard

"""
PQN implementation based on https://github.com/mttga/purejaxql/blob/main/purejaxql/pqn_gymnax.py
"""


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


@chex.dataclass(frozen=True)
class EMAMetrics:
    ema_alpha: float
    extrinsic_return_ema: float
    episode_length_ema: float


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
        ),
        None,
    )
    return env, env_params


def make_run(args):
    num_updates = int(args.total_time_steps // args.num_environments // args.num_steps)

    # Environment Setup
    env, env_params = make_env(args)

    input_size = int(env.observation_space(env_params).shape[0])
    num_actions = int(env.action_space(env_params).n)

    def run(key):
        # Network Setup
        key, subkey = jax.random.split(key, 2)
        initial_model = QNetworkCNN(
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

        episode_metrics = EMAMetrics(
            ema_alpha=2 / (args.num_episodes_for_average + 1),
            extrinsic_return_ema=jnp.nan,
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
                key, step_env_state, state, action, selected_q_value, all_q_values = (
                    carry
                )

                # Step Environment
                key, subkey = jax.random.split(key, 2)
                next_state, step_env_state, reward, done, info = env.step(
                    subkey, step_env_state, action
                )
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
                last_q_value = last_q_value * (
                    1 - transitions.done[-1]
                )  # If done, then no q value
                initial_return = transitions.reward[-1] + args.gamma * last_q_value
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

            ### TODO: Compute episode return metrics based on discussion with Mike

            metrics = {
                "env_step": env_step,
                "update_steps": step_number,
                "q_values": epoch_q_values.mean(),
            }
            # `epoch_losses` is a dict; one entry per loss component. Mean each
            # one across the scan dimensions and prefix with `loss/` for logging.
            metrics.update({f"loss_{k}": v.mean() for k, v in epoch_losses.items()})
            metrics.update({k: v.mean() for k, v in infos.items()})

            # Compute EMA of episode returns and lengths

            is_done = infos["returned_episode"]
            extrinsic_episode_returns = infos["returned_episode_returns"]
            episode_lengths = infos["returned_episode_lengths"]
            num_dones = is_done.sum()

            mean_extrinsic_episode_return = jnp.sum(
                is_done * extrinsic_episode_returns
            ) / jnp.maximum(num_dones, 1)
            updated_extrinsic_return_ema = update_ema(
                train_episode_metrics.extrinsic_return_ema,
                mean_extrinsic_episode_return,
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
                episode_length_ema=updated_episode_lengths_ema,
            )

            metrics["extrinsic_return_ema"] = updated_extrinsic_return_ema
            metrics["length_ema"] = updated_episode_lengths_ema

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

        return metrics

    return run


ConfigOptions = Union[
    Annotated[
        configs.AtariConfig,
        tyro.conf.subcommand(name="pong"),
    ]
]


if __name__ == "__main__":
    args = tyro.cli(
        ConfigOptions,
        default=configs.AtariConfig(),
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
    if args.num_seeds > 1:
        raise NotImplementedError("Multiple seeds are not currently supported")
    compiled_run = jax.jit(make_run(args))
    metrics = jax.block_until_ready(compiled_run(rng))
    print(f"Total time: {time.time() - t0}")

    metrics_path = save_path / "metrics.npz"
    np.savez(metrics_path, **metrics)

    print("Finished Run")
