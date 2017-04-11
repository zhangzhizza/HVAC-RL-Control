import gym
import eplus_env
import numpy as np
import threading
import time

class Eplus_worker_0:
    def work(self):
        env = gym.make('Eplus-v0');
        curSimTime, ob, is_terminal = env.reset(); # Reset the env (creat the EnergyPlus subprocess)
        while not is_terminal:
            time, ob, is_terminal = env.step([24,24]);
        is_terminal = env.reset()[-1];
        while not is_terminal:
            time, ob, is_terminal = env.step([15,15]);
        env.end_env();
    
class Eplus_worker_1:
    def work(self):
        env = gym.make('Eplus-v0');
        curSimTime, ob, is_terminal = env.reset(); # Reset the env (creat the EnergyPlus subprocess)
        while not is_terminal:
            time, ob, is_terminal = env.step([15,24]);
        is_terminal = env.reset()[-1];
        for _ in range(50):
            time, ob, is_terminal = env.step([15,15]);
        env.reset();
        env.end_env();
        
worker_work_0 = lambda: Eplus_worker_0().work();
thread_0 = threading.Thread(target = (worker_work_0));
print (thread_0.getName());
thread_0.start();

time.sleep(1)

worker_work_1 = lambda: Eplus_worker_1().work();
thread_1 = threading.Thread(target = (worker_work_1));
print (thread_1.getName());
thread_1.start();


thread_0.join(); 
thread_1.join();  

## Noforecast env
#env = gym.make('Eplus-v0');
#curSimTime, ob, is_terminal = env.reset(); # Reset the env (creat the EnergyPlus subprocess)
#print (env.min_max_limits, env.start_year, env.start_mon, env.start_day, env.start_weekday)
#while not is_terminal:
    #time, ob, is_terminal = env.step([24,24]);

#for _ in range(0):
    #time, ob, is_terminal = env.step([24,24]);

#is_terminal = env.reset()[-1];
#while not is_terminal:
    #time, ob, is_terminal = env.step([15,15]);
    
#is_terminal = env.reset()[-1];
#while not is_terminal:
    #time, ob, is_terminal = env.step([30,30]);
    
#curTime, ob, is_terminal = env.reset();
#is_print = True;
#while not is_terminal:
    #time, ob, is_terminal = env.step([15,30]);
    #if is_print:
        #print (ob);
        #is_print = False;
#env.end_env()
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