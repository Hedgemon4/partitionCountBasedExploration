import gymnax
from craftax.craftax_env import make_craftax_env_from_name

gymnax_envs = [
    "CartPole-v1",
    "Pendulum-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
]

craftax_envs = [
    "Craftax-Symbolic-v1",
    "Craftax-Symbolic-AutoReset-v1",
    "Craftax-Pixels-v1",
    "Craftax-Pixels-AutoReset-v1",
    "Craftax-Classic-Symbolic-v1",
    "Craftax-Classic-Symbolic-AutoReset-v1",
    "Craftax-Classic-Pixels-v1",
    "Craftax-Classic-Pixels-AutoReset-v1",
    "Craftax-Symbolic-v1",
    "Craftax-Pixels-v1",
    "Craftax-Classic-Symbolic-v1",
    "Craftax-Classic-Pixels-v1",
]

def make(environment_name):
    if environment_name in gymnax_envs:
        env, env_params = gymnax.make(environment_name)
        return env, env_params
    
    elif environment_name in craftax_envs:
        env = make_craftax_env_from_name(environment_name, auto_reset=True)
        env_params = env.default_params
        return env, env_params

    else:
        raise NotImplementedError()