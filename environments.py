from typing import Any

from gymnax.environments.classic_control.mountain_car import MountainCar, EnvState, EnvParams
import jax
import jax.numpy as jnp


class MountainCarPessimistic(MountainCar):
    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jnp.ndarray, jnp.ndarray, dict[Any, Any]]:
        """Perform single timestep state transition."""
        velocity = (
            state.velocity
            + (action - 1) * params.force
            - jnp.cos(3 * state.position) * params.gravity
        )
        velocity = jnp.clip(velocity, -params.max_speed, params.max_speed)
        position = state.position + velocity
        position = jnp.clip(position, params.min_position, params.max_position)
        velocity = velocity * (1 - (position == params.min_position) * (velocity < 0))

        reward = 0.0

        # Update state dict and evaluate termination conditions
        state = EnvState(position=position, velocity=velocity, time=state.time + 1)
        reached_goal = self.reached_goal(state, params)
        reward = jnp.where(reached_goal, 1.0, reward)
        done = self.is_terminal(state, params)

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            jnp.array(reward),
            done,
            {"discount": self.discount(state, params)},
        )

    def reached_goal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        return (state.position >= params.goal_position) * (
            state.velocity >= params.goal_velocity
        )
