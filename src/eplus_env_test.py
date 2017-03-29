import gym
import eplus_env
import numpy as np

"""
# Noforecast env
env = gym.make('Eplus-noforecast-v0');
is_terminal = env.reset()[-1];

while is_terminal != True:
    is_terminal = env.step([])[-1];

print ('resetting...................')
is_terminal = env.reset()

while is_terminal != True:
    is_terminal = env.step([])[-1];

env.end_env()
"""
# Forecast env
env = gym.make('Eplus-forecast-v0');
time, data, forecast, is_terminal = env.reset();
print (np.array(forecast)[:,0]);

while is_terminal != True:
    time, data, forecast, is_terminal = env.step([]);
    print (np.array(forecast)[:,0]);

print ('resetting...................')
time, data, forecast, is_terminal = env.reset()
print (np.array(forecast)[:,0]);
while is_terminal != True:
    time, data, forecast, is_terminal = env.step([]);
    print (np.array(forecast)[:,0]);

env.end_env()
