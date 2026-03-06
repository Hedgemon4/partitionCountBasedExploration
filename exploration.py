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
    return action_selection
