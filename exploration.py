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
def epsilon_greedy_with_intrinsic_q_values(random_key, epsilon, extrinsic_q_values, intrinsic_q_values, beta):
    # Choose action
    action_key, epsilon_key = jax.random.split(random_key)

    # Find max over the sum of extrinsic and intrinsic Q-values
    total_q_values = extrinsic_q_values + (beta * intrinsic_q_values)
    max_action = jnp.argmax(total_q_values, axis=-1)

    epsilon_matrix = jax.random.uniform(epsilon_key, shape=max_action.shape)
    random_action = jax.random.randint(
        action_key, shape=max_action.shape, minval=0, maxval=extrinsic_q_values.shape[-1]
    )
    action_selection = jnp.where(epsilon_matrix < epsilon, random_action, max_action)
    batch_indices = jnp.arange(extrinsic_q_values.shape[0])

    selected_extrinsic_q_values = extrinsic_q_values[batch_indices, action_selection]
    selected_intrinsic_q_values = intrinsic_q_values[batch_indices, action_selection]

    return action_selection, selected_extrinsic_q_values, selected_intrinsic_q_values
