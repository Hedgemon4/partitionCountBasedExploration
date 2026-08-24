import jax
import jax.numpy as jnp


def epsilon_greedy(random_key, epsilon, q_values):
    # Choose action
    action_key, epsilon_key = jax.random.split(random_key)
    max_action = jnp.argmax(q_values, axis=-1)
    epsilon_matrix = jax.random.uniform(epsilon_key, shape=max_action.shape)
    random_action = jax.random.randint(
        action_key, shape=max_action.shape, minval=0, maxval=q_values.shape[-1]
    )
    action_selection = jnp.where(epsilon_matrix < epsilon, random_action, max_action)
    batch_indices = jnp.arange(q_values.shape[0])
    selected_q_values = q_values[batch_indices, action_selection]
    return action_selection, selected_q_values


"""
Epsilon-greedy action selection that incorporates both extrinsic and intrinsic Q-values
"""


def epsilon_greedy_with_intrinsic_q_values(
    random_key, epsilon, extrinsic_q_values, intrinsic_q_values, beta
):
    # Choose action
    action_key, epsilon_key = jax.random.split(random_key)

    # Find max over the sum of extrinsic and intrinsic Q-values
    total_q_values = extrinsic_q_values + (beta * intrinsic_q_values)
    max_action = jnp.argmax(total_q_values, axis=-1)

    epsilon_matrix = jax.random.uniform(epsilon_key, shape=max_action.shape)
    random_action = jax.random.randint(
        action_key,
        shape=max_action.shape,
        minval=0,
        maxval=extrinsic_q_values.shape[-1],
    )
    # Returned so callers can tell an exploratory step from a greedy one. Inferring
    # it as `action == max_action` would miscount the random draws that land on the
    # greedy action anyway -- 1/|A| of exploratory steps.
    explored = epsilon_matrix < epsilon
    action_selection = jnp.where(explored, random_action, max_action)
    batch_indices = jnp.arange(extrinsic_q_values.shape[0])

    selected_extrinsic_q_values = extrinsic_q_values[batch_indices, action_selection]
    selected_intrinsic_q_values = intrinsic_q_values[batch_indices, action_selection]

    return (
        action_selection,
        selected_extrinsic_q_values,
        selected_intrinsic_q_values,
        explored,
    )


def intrinsic_greedy_divergence(extrinsic_q_values, intrinsic_q_values, beta):
    """Per-element indicator that the intrinsic head moved the greedy action.

    Compares argmax(Q_e + beta * Q_i) against argmax(Q_e). Independent of epsilon,
    so it measures the policy the intrinsic head induces rather than how often
    exploration happened to fire -- and it is identically 0 when beta == 0, which
    makes the control arm self-checking.

    This answers a question the exploration share beta*Q_i / (Q_e + beta*Q_i)
    cannot: that ratio compares the two heads' *levels*, while what changes
    behaviour is their spread across actions. A ratio of 1% is compatible with the
    argmax flipping constantly, and a ratio near 1 with it never flipping.

    Recomputes the fused argmax that epsilon_greedy_with_intrinsic_q_values also
    takes internally. One argmax over (num_envs, num_actions) is free next to the
    forward pass, and keeping this separate leaves that function's callers alone.
    """
    fused = jnp.argmax(extrinsic_q_values + beta * intrinsic_q_values, axis=-1)
    extrinsic = jnp.argmax(extrinsic_q_values, axis=-1)
    return (fused != extrinsic).astype(jnp.float32)
