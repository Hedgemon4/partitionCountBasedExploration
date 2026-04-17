import xminigrid
from gymnax.environments import spaces
import jax.numpy as jnp

class MiniGrid: 
    def __init__(self, env_name) -> None:
        self.env, self.env_params = xminigrid.make(env_name)
        self.timestep = None
        print(self.env_params)

    def reset(self, key, env_params):
        self.timestep = self.env.reset(env_params, key)
        return self.timestep.observation, self.timestep.state

    def step(self, key_step, state, action, env_params):
        self.timestep = self.env.step(env_params, self.timestep, action)
        info = {'discount': self.timestep.discount}
        return self.timestep.observation, self.timestep.state, self.timestep.reward, self.timestep.last(), info

    def render(self):
        self.env.render()

    def observation_space(self, env_params):
        #TODO: The low and high are written for doorkey env, need to be changed for other envs
        low = jnp.array(
            [0, 0, 0],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [env_params.height - 1, env_params.width - 1, 1],
            dtype=jnp.float32,
        )
        return spaces.Box(low=low, high=high, shape=self.env.observation_shape(env_params), dtype=jnp.float32)
    
    def action_space(self, env_params):
        num_actions = self.env.num_actions(env_params)
        return spaces.Discrete(num_actions)
    
    