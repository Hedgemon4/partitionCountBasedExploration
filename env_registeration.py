from envs.minigrid import MiniGrid
import gymnax

def make(env_name: str, **env_kwargs):
    ## TODO: Put the supported envs in a list and raise an error if the env_name is not in the list
    if env_name == "MiniGrid-DoorKey-8x8":
        env = MiniGrid(env_name)
        env_params = env.env_params
    else:
        env, env_params = gymnax.make(env_name)

    return env, env_params