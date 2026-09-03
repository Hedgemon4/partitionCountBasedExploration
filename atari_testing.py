import jax
from ale_py import AtariVectorEnv
from gymnasium.vector import AutoresetMode
import jax.numpy as jnp
from wrappers import ALEGymnaxWrapperStandard

test_env = AtariVectorEnv(
    "freeway",
    num_envs=8,
    autoreset_mode=AutoresetMode.SAME_STEP,
    stack_num=4,
    frameskip=4,
)

env_name = "freeway"
num_envs = 8
seed = 0

# env = ALEGymnaxWrapperStandard(env_name, num_envs=num_envs, seed=seed)
env = test_env

rng = jax.random.PRNGKey(0)
keys = jax.random.split(rng, 3)
obs, env_state = env.reset()


def step_env(carry, _):
    rng, state = carry
    key, subkey = jax.random.split(rng)
    obs, state, reward, done, info = env.step(
        key, env_state, jnp.array([0, 0, 0, 0, 0, 0, 0, 0])
    )
    next_carry = (subkey, state)
    return next_carry, (obs, reward, done)


carry = (keys[1], env_state)

jit_step = jax.jit(step_env)
final_carry, outs = jax.lax.scan(jit_step, carry, None, length=10)

print(outs[0].shape)
