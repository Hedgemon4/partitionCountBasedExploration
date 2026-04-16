import jax.numpy as jnp


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
