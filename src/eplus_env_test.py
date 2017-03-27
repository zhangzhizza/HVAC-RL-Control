import gym
import core.eplus_env.eplus8_6

env = gym.make('Eplus-v0');
print ('reset')
env.reset()
print ('take step 1')
env.step([]);
print ('take step 2')
env.step([]);

env.end_env()