from functools import partial
from typing import Tuple, Optional, Union, Any, cast

import chex
import jax
from ale_py import AtariVectorEnv
from gymnasium.vector import AutoresetMode
from gymnax.environments import environment, spaces
import jax.numpy as jnp
import numpy as np


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


@chex.dataclass(frozen=True)
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=0,
            episode_lengths=0,
            returned_episode_returns=0,
            returned_episode_lengths=0,
            timestep=0,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


class PessimisticMountainCarWrapper(GymnaxWrapper):
    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        goal_reached = (state.position >= params.goal_position) * (
            state.velocity >= params.goal_velocity
        )
        reward = jnp.where(goal_reached, 1.0, 0.0)
        return obs, state, reward, done, info


@chex.dataclass(frozen=True)
class GymVecEnvState:
    handle: Any


class ALEGymnaxWrapperXLA:
    """Fast XLA-compiled Atari environment wrapper."""

    def __init__(self, env_name, num_envs, seed, **kwargs):
        self._env = AtariVectorEnv(
            env_name,
            num_envs=num_envs,
            autoreset_mode=AutoresetMode.SAME_STEP,
            **kwargs
        )
        self.init_handle, self._reset, self._step = self._env.xla()
        self.init_reset_seed = seed

    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, GymVecEnvState]:
        handle, (obs, _) = self._reset(self.init_handle, seed=self.init_reset_seed)
        state = GymVecEnvState(handle=handle)
        return obs[0], state

    def step(
        self,
        key: chex.PRNGKey,
        state: GymVecEnvState,
        action: chex.Array,
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, GymVecEnvState, chex.Array, chex.Array, dict[str, Any]]:
        if action.ndim == 0:
            action = jnp.expand_dims(action, axis=0)  # envpool always expects batch dim
        handle = state.handle
        handle, (obs, rew, term, trunc, info) = self._step(handle, action)
        done = term | trunc

        next_state = GymVecEnvState(handle=handle)
        info["truncated"] = trunc[0]

        return obs[0], next_state, rew[0], done[0], info

    def observation_space(self, params: Optional[environment.EnvParams] = None):
        obs_space = cast(Any, self._env.observation_space)
        return spaces.Box(
            low=obs_space.low,
            high=obs_space.high,
            shape=obs_space.shape[1:],
            dtype=obs_space.dtype,
        )

    def action_space(self, params: Optional[environment.EnvParams] = None):
        action_space = cast(Any, self._env.action_space)
        return spaces.Discrete(
            num_categories=action_space.nvec[0],
        )


class ALEGymnaxWrapperStandard:
    """Standard ale_py Atari environment wrapper with JAX compatibility."""

    def __init__(self, env_name, seed, **kwargs):
        self._env = AtariVectorEnv(
            env_name, num_envs=1, autoreset_mode=AutoresetMode.SAME_STEP, **kwargs
        )
        self._env.reset(seed=seed)
        self.init_reset_seed = seed

    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, GymVecEnvState]:
        obs, _ = self._env.reset(seed=self.init_reset_seed)
        state = GymVecEnvState(handle=None)
        return obs[0], state

    def step(
        self,
        key: chex.PRNGKey,
        state: GymVecEnvState,
        action: chex.Array,
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, GymVecEnvState, chex.Array, chex.Array, dict[str, Any]]:
        if action.ndim == 0:
            action = jnp.expand_dims(action, axis=0)

        # Use pure_callback to call non-JAX ale_py from inside JIT
        obs, rew, term, trunc = self._step_callback(action)
        rew = jnp.atleast_1d(jnp.asarray(rew, dtype=jnp.float32))
        term = jnp.atleast_1d(jnp.asarray(term, dtype=jnp.bool_))
        trunc = jnp.atleast_1d(jnp.asarray(trunc, dtype=jnp.bool_))
        done = jnp.logical_or(term, trunc)

        next_state = GymVecEnvState(handle=None)
        info = {}  # Empty info dict for standard ale_py
        info["truncated"] = trunc[0]
        obs_out = cast(Any, obs)[0]
        rew_out = cast(Any, rew)[0]
        done_out = cast(Any, done)[0]

        return obs_out, next_state, rew_out, done_out, info

    def _step_callback(
        self, action: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """Wrapped callback for use inside JIT-compiled functions."""
        obs_space = cast(Any, self._env.observation_space)
        obs_shape = (1, *obs_space.shape[1:])

        obs, rew, term, trunc = jax.pure_callback(
            lambda a: self._step_callback_impl(a),
            (
                jax.ShapeDtypeStruct(obs_shape, dtype=np.uint8),
                jax.ShapeDtypeStruct((1,), dtype=np.float32),
                jax.ShapeDtypeStruct((1,), dtype=np.bool_),
                jax.ShapeDtypeStruct((1,), dtype=np.bool_),
            ),
            action,
        )

        # Convert to JAX arrays if needed
        obs = jnp.asarray(obs)
        rew = jnp.asarray(rew, dtype=jnp.float32)
        term = jnp.asarray(term, dtype=jnp.bool_)
        trunc = jnp.asarray(trunc, dtype=jnp.bool_)

        return obs, rew, term, trunc

    def _step_callback_impl(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Implementation of step callable from pure_callback."""
        action_np = np.asarray(action)
        obs, rew, term, trunc, _ = self._env.step(action_np)
        rew = np.asarray(rew, dtype=np.float32)
        return obs, rew, term, trunc

    def observation_space(self, params: Optional[environment.EnvParams] = None):
        obs_space = cast(Any, self._env.observation_space)
        return spaces.Box(
            low=obs_space.low,
            high=obs_space.high,
            shape=obs_space.shape[1:],
            dtype=obs_space.dtype,
        )

    def action_space(self, params: Optional[environment.EnvParams] = None):
        action_space = cast(Any, self._env.action_space)
        return spaces.Discrete(
            num_categories=action_space.nvec[0],
        )
