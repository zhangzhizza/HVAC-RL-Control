import gym
import eplus_env
import numpy as np


# Noforecast env
env = gym.make('Eplus-eval-multiagent-v0');
print (env.min_max_limits, env.start_year, env.start_mon, env.start_day, env.start_weekday)
is_terminal = env.reset()[-1];
while not is_terminal:
    time, ob, is_terminal = env.step([24,24, 24, 24, 24, 24, 24, 24]);
    print (time, ob)

"""
for _ in range(0):
    time, ob, is_terminal = env.step([24,24, 24, 24, 24, 24, 24, 24]);

is_terminal = env.reset()[-1];
while not is_terminal:
    time, ob, is_terminal = env.step([15,15, 15, 15, 15, 15, 15, 15]);
    
is_terminal = env.reset()[-1];
while not is_terminal:
    time, ob, is_terminal = env.step([30,30,30, 30, 30, 30, 30, 30]);
    
curTime, ob, is_terminal = env.reset();
is_print = True;
while not is_terminal:
    time, ob, is_terminal = env.step([15,30]);
    if is_print:
        print (ob);
        is_print = False;
env.end_env()

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
