from ale_py import AtariVectorEnv
from gymnasium.vector import AutoresetMode

test_env = AtariVectorEnv(
    "freeway",
    num_envs=8,
    autoreset_mode=AutoresetMode.SAME_STEP,
    stack_num=4,
    frameskip=4,
)

item = test_env.reset()

print(item[0].shape)
