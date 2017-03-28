# eplus-env

This environment wraps the EnergyPlus-v-8-6 into the OpenAI gym environment interface. 
### Installation
EnergyPlus is platform dependent. Current version only works in Linux OS. 
```sh
$ cd eplus-env
$ pip install -e .
```
### Usage
```python
import gym;
import eplus_env;

env = gym.make('Eplus-v0');
ob = env.reset(); # Reset the env (creat the EnergyPlus subprocess)
for _ in range(10000):
    action = someFuncToGetAction(ob); # Should be a python list of float
    ob = env.step(action);

ob = env.reset(); # Reset the env again (previous EnergyPlus subprocess
                  # will be killed; the output from EnergyPlus can be found
                  # under the $pwd/Eplus-env-runX/Eplus-env-sub_runX; new
                  # EnergyPlus subprocess will be created.)
env.end_env(); # Safe end the environment after use. 
