from gym.envs.registration import register
import os

FD = os.path.dirname(os.path.realpath(__file__));

register(
    id='Eplus-v0',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/variables.cfg',
            'idf_path':FD + '/envs/5ZoneAutoDXVAV.idf'});

register(
    id='Eplus-forecast-v0',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/variables.cfg',
            'idf_path':FD + '/envs/5ZoneAutoDXVAV.idf',
            'incl_forecast': True,
            'forecast_step': 36});
    
register(
    id='Eplus-eval-v0',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/pennstate.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/variables.cfg',
            'idf_path':FD + '/envs/5ZoneAutoDXVAV_eval.idf'});