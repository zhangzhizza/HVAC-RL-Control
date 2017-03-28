import gym
import core.eplus-env.eplus_env

env = gym.make('Eplus-v0');
print ('reset')
env.reset()

for _ in range(500):
    env.step([]);

env.end_env()