from torchrl.envs import GymEnv

from src.env.GridWorldEnv import GridWorldEnv

if __name__ == '__main__':
    env = GymEnv.register_gym(id="gymnasium_env/GridWorld-v0", entry_point=GridWorldEnv, backend="gymnasium")
    print(env)