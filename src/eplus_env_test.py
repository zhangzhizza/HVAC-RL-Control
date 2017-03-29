import gym
import eplus_env
import numpy as np


# Noforecast env
env = gym.make('Eplus-v0');
is_terminal = env.reset()[-1];
while not is_terminal:
    time, ob, is_terminal = env.step([24,24]);
    print (ob)
    
is_terminal = env.reset()[-1];
while not is_terminal:
    time, ob, is_terminal = env.step([24,24]);
    print (ob)
env.end_env()
"""
while is_terminal != True:
    is_terminal = env.step([])[-1];

print ('resetting...................')
is_terminal = env.reset()

while is_terminal != True:
    is_terminal = env.step([])[-1];

env.end_env()

# Forecast env
env = gym.make('Eplus-forecast-v0');

for _ in range(50):
    time, data, forecast, is_terminal = env.reset();
    print (time, data, np.array(forecast)[:,0], is_terminal);
    while is_terminal != True:
        time, data, forecast, is_terminal = env.step([]);

env.end_env()
"""