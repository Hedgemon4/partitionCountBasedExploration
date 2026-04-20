import struct
from functools import partial
from typing import Tuple, Optional, Union

import chex
import jax
from gymnax.environments import environment, spaces
import jax.numpy as jnp
import numpy as np
from navix import Environment

try:
    import navix as _navix  # noqa: F401
    _NAVIX_AVAILABLE = True
except ImportError:
    _NAVIX_AVAILABLE = False


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)



def NavixFlattenObservationWrapper(env: Environment):
    """A wrapper to flatten the observation space of the environment."""
    flatten_obs_fn = lambda x: jnp.ravel(env.observation_fn(x))
    flatten_obs_shape = (int(np.prod(env.observation_space.shape)),)
    return env.replace(
        observation_fn=flatten_obs_fn,
        observation_space=env.observation_space.replace(shape=flatten_obs_shape),
    )


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


class NavixGymnaxWrapper:
    """Adapts a navix environment to the gymnax-style API used throughout this project.

    navix uses a Timestep-based API (reset(key) -> Timestep, step(timestep, action) -> Timestep),
    whereas this project expects gymnax's API (reset(key, params) -> (obs, state),
    step(key, state, action, params) -> (obs, state, reward, done, info)).

    This wrapper bridges that gap so NavixGymnaxWrapper can be passed directly into
    FlattenObservationWrapper and LogWrapper without any other changes.
    """

    def __init__(self, navix_env):
        if not _NAVIX_AVAILABLE:
            raise ImportError(
                "navix is not installed. Run: pip install navix"
            )
        self._env = navix_env
        # Probe observation shape with a single dummy reset (runs once at init, not during training)
        dummy_timestep = navix_env.reset(jax.random.PRNGKey(0))
        self._obs_shape = dummy_timestep.observation.shape
        self._num_actions = int(navix_env.action_space.n)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def observation_space(self, params=None) -> spaces.Box:
        """Return a gymnax-compatible Box matching the navix observation shape."""
        return spaces.Box(
            low=0.0,
            high=255.0,
            shape=self._obs_shape,
            dtype=np.float32,
        )

    def action_space(self, params=None) -> spaces.Discrete:
        """Return a gymnax-compatible Discrete matching the navix action count."""
        return spaces.Discrete(self._num_actions)

    def reset(
        self, key: chex.PRNGKey, params=None
    ) -> Tuple[chex.Array, object]:
        """Reset the environment and return (obs, timestep).

        The returned timestep is used as the 'state' token passed to step().
        """
        timestep = self._env.reset(key)
        obs = timestep.observation.astype(jnp.float32)
        return obs, timestep

    def step(
        self,
        key: chex.PRNGKey,
        timestep,
        action: Union[int, float],
        params=None,
    ) -> Tuple[chex.Array, object, float, bool, dict]:
        """Step the environment.

        key is accepted for API compatibility but ignored (navix is deterministic
        given the timestep; stochasticity lives inside the navix state).
        """
        new_timestep = self._env.step(timestep, action)
        obs = new_timestep.observation.astype(jnp.float32)
        reward = new_timestep.reward
        done = new_timestep.is_done()
        return obs, new_timestep, reward, done, {}
