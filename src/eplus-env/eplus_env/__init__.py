from gym.envs.registration import register
import os
import fileinput

from eplus_env.eplus_env_statelimits import min_max_limits_dict;

FD = os.path.dirname(os.path.realpath(__file__));
MIN_MAX_LIMITS_DICT = min_max_limits_dict;

"""Legacy envs
register(
    id='Eplus-v0',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/cfg/variables_v0.cfg',
            'idf_path':FD + '/envs/idf/envs/v0/learning/5ZoneAutoDXVAV_v0.idf',
            'env_name': 'Eplus-v0'});

register(
    id='Eplus-v1',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/cfg/variables_v1.cfg',
            'idf_path':FD + '/envs/idf/envs/v1/learning/5ZoneAutoDXVAV_v1.idf',
            'env_name': 'Eplus-v1'});

register(
    id='Eplus-v2',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/cfg/variables_v2.cfg',
            'idf_path':FD + '/envs/idf/envs/v2/learning/5ZoneAutoDXVAV_v2.idf',
            'env_name': 'Eplus-v2'});

register(
    id='Eplus-v3',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/cfg/variables_v3.cfg',
            'idf_path':FD + '/envs/idf/envs/v3/learning/5ZoneAutoDXVAV_v3.idf',
            'env_name': 'Eplus-v3'});

register(
    id='Eplus-forecast-v0',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/cfg/variables_v0.cfg',
            'idf_path':FD + '/envs/idf/envs/v0/learning/5ZoneAutoDXVAV_v0.idf',
            'incl_forecast': True,
            'forecast_step': 36,
            'env_name': 'Eplus-forecast-v0'});
    
register(
    id='Eplus-eval-v0',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/weather/pennstate.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/cfg/variables_v0.cfg',
            'idf_path':FD + '/envs/idf/envs/v0/learning/5ZoneAutoDXVAV_eval_v0.idf',
            'env_name': 'Eplus-eval-v0'});

register(
    id='Eplus-eval-v1',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/weather/pennstate.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/cfg/variables_v1.cfg',
            'idf_path':FD + '/envs/idf/envs/v1/learning/5ZoneAutoDXVAV_eval_v1.idf',
            'env_name': 'Eplus-eval-v1'});

register(
    id='Eplus-eval-v2',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/weather/pennstate.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/cfg/variables_v2.cfg',
            'idf_path':FD + '/envs/idf/envs/v2/learning/5ZoneAutoDXVAV_eval_v2.idf',
            'env_name': 'Eplus-eval-v2'});

register(
    id='Eplus-eval-v3',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/weather/pennstate.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/cfg/variables_v3.cfg',
            'idf_path':FD + '/envs/idf/envs/v3/learning/5ZoneAutoDXVAV_eval_v3.idf',
            'env_name': 'Eplus-eval-v3'});
    
register(
    id='Eplus-eval-multiagent-v0',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/weather/pennstate.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/cfg/variables_multiagent_v0.cfg',
            'idf_path':FD + '/envs/idf/envs/v0/learning/5ZoneAutoDXVAV_eval_multiagent.idf',
            'env_name': 'Eplus-eval-multiagent-v0'});

register(
    id='Eplus-eval-multiagent-v1',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/weather/pennstate.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/cfg/variables_multiagent_v1.cfg',
            'idf_path':FD + '/envs/idf/envs/v1/learning/5ZoneAutoDXVAV_multiagent_eval_v1.idf',
            'env_name': 'Eplus-eval-multiagent-v1'});

register(
    id='Eplus-multiagent-v1',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/cfg/variables_multiagent_v1.cfg',
            'idf_path':FD + '/envs/idf/envs/v1/learning/5ZoneAutoDXVAV_multiagent_v1.idf',
            'env_name': 'Eplus-multiagent-v1'});
register(
    id='IW-v57',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/envs/iw/learning/iw_v57_learning.cfg',
            'idf_path':FD + '/envs/idf/envs/iw/learning/iw_v57_learning.idf',
            'env_name': 'IW-v57'});

register(
    id='IW-eval-v57',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/envs/iw/learning/iw_v57_learning_eval.cfg',
            'idf_path':FD + '/envs/idf/envs/iw/learning/iw_v57_learning_eval.idf',
            'env_name': 'IW-eval-v57'});

register(
    id='IW-v5702',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/envs/iw/learning/iw_v57_2_learning.cfg',
            'idf_path':FD + '/envs/idf/envs/iw/learning/iw_v57_2_learning.idf',
            'env_name': 'IW-v5702'});

register(
    id='IW-eval-v5702',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/envs/iw/learning/iw_v57_2_learning_eval.cfg',
            'idf_path':FD + '/envs/idf/envs/iw/learning/iw_v57_2_learning_eval.idf',
            'env_name': 'IW-eval-v5702'});

register(
    id='IW-v570202',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/envs/iw/learning/iw_v57_2_learning.cfg',
            'idf_path':FD + '/envs/idf/envs/iw/learning/iw_v57_2_learning.idf',
            'env_name': 'IW-v570202'});

register(
    id='IW-eval-v570202',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/envs/iw/learning/iw_v57_2_learning_eval.cfg',
            'idf_path':FD + '/envs/idf/envs/iw/learning/iw_v57_2_learning_eval.idf',
            'env_name': 'IW-eval-v570202'});

register(
    id='IW-v570203',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/envs/iw/learning/iw_v57_2_learning.cfg',
            'idf_path':FD + '/envs/idf/envs/iw/learning/iw_v57_2_oneMon_learning.idf',
            'env_name': 'IW-v570203'});

register(
    id='IW-v570204',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/envs/iw/learning/iw_v57_2_learning.cfg',
            'idf_path':FD + '/envs/idf/envs/iw/learning/iw_v57_2_learning.idf',
            'env_name': 'IW-v570204',
            'act_repeat': 6});

register(
    id='IW-v82',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/idf/envs/iw/learning/iw_v82_learning.cfg',
            'idf_path':FD + '/envs/idf/envs/iw/learning/iw_v82_learning.idf',
            'env_name': 'IW-v82',
            'act_repeat': 6});
"""
register(
    id='IW-tmy3Weather-v9601',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/iw_v96/learning/cfg/tmy3Weather.cfg',
            'idf_path':FD + '/envs/eplus_models/iw_v96/learning/idf/tmy3Weather.idf',
            'env_name': 'IW-tmy3Weather-v9601',
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-tmy3Weather-v9601'],
            'incl_forecast': True,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': 'tmy3',
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 3});

register(
    id='IW-realWeather-v9601',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/iw_v96/learning/cfg/realWeather.cfg',
            'idf_path':FD + '/envs/eplus_models/iw_v96/learning/idf/realWeather.idf',
            'env_name': 'IW-realWeather-v9601',
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-realWeather-v9601'],
            'incl_forecast': True,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': FD + '/envs/eplus_models/iw_v96/weather/x.csv',
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 3});

register(
    id='IW-tmy3Weather-v9602',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/iw_v96/learning/cfg/tmy3Weather.cfg',
            'idf_path':FD + '/envs/eplus_models/iw_v96/learning/idf/tmy3Weather.idf',
            'env_name': 'IW-tmy3Weather-v9602',
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-tmy3Weather-v9602'],
            'incl_forecast': False,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': 'tmy3',
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 3});

register(
    id='IW-realWeather-v9602',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/iw_v96/learning/cfg/realWeather.cfg',
            'idf_path':FD + '/envs/eplus_models/iw_v96/learning/idf/realWeather.idf',
            'env_name': 'IW-realWeather-v9602',
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-tmy3Weather-v9602'],
            'incl_forecast': False,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': FD + '/envs/eplus_models/iw_v96/weather/x.csv',
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 3});

register(
    id='IW-tmy3Weather-v9603',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/iw_v96/learning/cfg/tmy3Weather.cfg',
            'idf_path':FD + '/envs/eplus_models/iw_v96/learning/idf/tmy3Weather.idf',
            'env_name': 'IW-tmy3Weather-v9603',
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-tmy3Weather-v9603'],
            'incl_forecast': True,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': 'tmy3',
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 6});

register(
    id='IW-realWeather-v9603',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/iw_v96/learning/cfg/realWeather.cfg',
            'idf_path':FD + '/envs/eplus_models/iw_v96/learning/idf/realWeather.idf',
            'env_name': 'IW-realWeather-v9603',
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-realWeather-v9603'],
            'incl_forecast': True,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': FD + '/envs/eplus_models/iw_v96/weather/x.csv',
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 6});

register(
    id='IW-tmy3Weather-v9604',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/iw_v96/learning/cfg/tmy3Weather.cfg',
            'idf_path':FD + '/envs/eplus_models/iw_v96/learning/idf/tmy3Weather.idf',
            'env_name': 'IW-tmy3Weather-v9604',
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-tmy3Weather-v9604'],
            'incl_forecast': False,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': 'tmy3',
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 6});

register(
    id='IW-realWeather-v9604',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/iw_v96/learning/cfg/realWeather.cfg',
            'idf_path':FD + '/envs/eplus_models/iw_v96/learning/idf/realWeather.idf',
            'env_name': 'IW-realWeather-v9604',
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-realWeather-v9604'],
            'incl_forecast': False,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': FD + '/envs/eplus_models/iw_v96/weather/x.csv',
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 6});

register(
    id='IW-tmy3Weather-v9606',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/iw_v96/learning/cfg/tmy3Weather.cfg',
            'idf_path':FD + '/envs/eplus_models/iw_v96/learning/idf/tmy3Weather.idf',
            'env_name': 'IW-tmy3Weather-v9606',
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-tmy3Weather-v9606'],
            'incl_forecast': False,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': 'tmy3',
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 1});

register(
    id='IW-realWeather-v9606',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/iw_v96/learning/cfg/realWeather.cfg',
            'idf_path':FD + '/envs/eplus_models/iw_v96/learning/idf/realWeather.idf',
            'env_name': 'IW-realWeather-v9606',
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-realWeather-v9606'],
            'incl_forecast': False,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': FD + '/envs/eplus_models/iw_v96/weather/x.csv',
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 1});

register(
    id='IW-tmy3Weather-v9706',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/iw_v97/learning/cfg/tmy3Weather.cfg',
            'idf_path':FD + '/envs/eplus_models/iw_v97/learning/idf/tmy3Weather.idf',
            'env_name': 'IW-tmy3Weather-v9706',
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-tmy3Weather-v9706'],
            'incl_forecast': False,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': 'tmy3',
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 1});

register(
    id='IW-realWeather-v9706',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/iw_v97/learning/cfg/realWeather.cfg',
            'idf_path':FD + '/envs/eplus_models/iw_v97/learning/idf/realWeather.idf',
            'env_name': 'IW-realWeather-v9706',
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-realWeather-v9706'],
            'incl_forecast': False,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': FD + '/envs/eplus_models/iw_v97/weather/x.csv',
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 1});

register(
    id='IW-tmy3Weather-fore-v9706',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/iw_v97/learning/cfg/tmy3Weather.cfg',
            'idf_path':FD + '/envs/eplus_models/iw_v97/learning/idf/tmy3Weather.idf',
            'env_name': 'IW-tmy3Weather-fore-v9706',
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-tmy3Weather-fore-v9706'],
            'incl_forecast': True,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': 'tmy3',
            'forecastFilePath': None,
            'forecast_hour': 6,
            'act_repeat': 1});

register(
    id='IW-realWeather-fore-v9706',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/iw_v97/learning/cfg/realWeather.cfg',
            'idf_path':FD + '/envs/eplus_models/iw_v97/learning/idf/realWeather.idf',
            'env_name': 'IW-realWeather-fore-v9706',
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-realWeather-fore-v9706'],
            'incl_forecast': True,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': FD + '/envs/eplus_models/iw_v97/weather/x.csv',
            'forecastFilePath': None,
            'forecast_hour': 6,
            'act_repeat': 1});

register(
    id='IW-tmy3Weather-v9701',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/iw_v97/learning/cfg/tmy3Weather.cfg',
            'idf_path':FD + '/envs/eplus_models/iw_v97/learning/idf/tmy3Weather.idf',
            'env_name': 'IW-tmy3Weather-v9701',
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-tmy3Weather-v9701'],
            'incl_forecast': False,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': 'tmy3',
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 3});

register(
    id='IW-realWeather-v9701',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/iw_v97/learning/cfg/realWeather.cfg',
            'idf_path':FD + '/envs/eplus_models/iw_v97/learning/idf/realWeather.idf',
            'env_name': 'IW-realWeather-v9701',
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-realWeather-v9701'],
            'incl_forecast': False,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': FD + '/envs/eplus_models/iw_v97/weather/x.csv',
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 3});

register(
    id='5z-tmy3Weather-v1',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/5z_v1/learning/cfg/5ZoneAutoDXVAV_v1.cfg',
            'idf_path':FD + '/envs/eplus_models/5z_v1/learning/idf/5ZoneAutoDXVAV_v1.idf',
            'env_name': '5z-tmy3Weather-v1',
            'min_max_limits': MIN_MAX_LIMITS_DICT['5z-tmy3Weather-v1'],
            'incl_forecast': False,
            'forecastRandMode': 'tmy3',
            'forecastRandStd': 0.15,
            'forecastSource': None,
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 1});

register(
    id='IW-imp-v9701',
    entry_point='eplus_env.envs:IW_IMP_V97',
    kwargs={'site_server_ip': 'localhost',
            'rd_port': 61221,
            'wt_port': 61222,
            'env_name': 'IW-imp-v9701',
            'defaultObValues': [50, 50, 0, 0, 50, 72, 72, 72, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            'localLat': 40.4406,
            'localLong': -79.9959,
            'ctrl_step_size_s': 300,
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-imp-v9701'],
            'incl_forecast': False,
            'forecastRandMode': 'tmy3',
            'forecastRandStd': 0.15,
            'forecastSource': None,
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 3,
            'isPPDBk': False});

register(
    id='IW-imp-v9702',
    entry_point='eplus_env.envs:IW_IMP_V97',
    kwargs={'site_server_ip': 'localhost',
            'rd_port': 61221,
            'wt_port': 61222,
            'env_name': 'IW-imp-v9702-deprecated',
            'defaultObValues': [50, 50, 0, 0, 50, 72, 72, 72, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            'localLat': 40.4406,
            'localLong': -79.9959,
            'ctrl_step_size_s': 300,
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-imp-v9702'],
            'incl_forecast': False,
            'forecastRandMode': 'tmy3',
            'forecastRandStd': 0.15,
            'forecastSource': None,
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 3,
            'iat_thres': 22});


register(
    id='IW-imp-v9703',
    entry_point='eplus_env.envs:IW_IMP_V97',
    kwargs={'site_server_ip': 'localhost',
            'rd_port': 61221,
            'wt_port': 61222,
            'env_name': 'IW-imp-v9703',
            'defaultObValues': [50, 50, 0, 0, 50, 72, 72, 72, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            'localLat': 40.4406,
            'localLong': -79.9959,
            'ctrl_step_size_s': 300,
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-imp-v9701'],
            'incl_forecast': False,
            'forecastRandMode': 'tmy3',
            'forecastRandStd': 0.15,
            'forecastSource': None,
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 3,
            'isPPDBk': True,
            'clo': 1.0,
            'met': 1.2, 
            'airVel': 0.1});

register(
    id='IW-imp-v9704',
    entry_point='eplus_env.envs:IW_IMP_V97',
    kwargs={'site_server_ip': 'localhost',
            'rd_port': 61221,
            'wt_port': 61222,
            'env_name': 'IW-imp-v9704',
            'defaultObValues': [50, 50, 0, 0, 50, 72, 72, 72, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            'localLat': 40.4406,
            'localLong': -79.9959,
            'ctrl_step_size_s': 300,
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-imp-v9701'],
            'incl_forecast': False,
            'forecastRandMode': 'tmy3',
            'forecastRandStd': 0.15,
            'forecastSource': None,
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 3,
            'isPPDBk': True,
            'clo': 1.0,
            'met': 1.05, 
            'airVel': 0.1});

register(
    id='IW-imp-v9705',
    entry_point='eplus_env.envs:IW_IMP_V97',
    kwargs={'site_server_ip': 'localhost',
            'rd_port': 61221,
            'wt_port': 61222,
            'env_name': 'IW-imp-v9705',
            'defaultObValues': [50, 50, 0, 0, 50, 72, 72, 72, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            'localLat': 40.4406,
            'localLong': -79.9959,
            'ctrl_step_size_s': 300,
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-imp-v9701'],
            'incl_forecast': False,
            'forecastRandMode': 'tmy3',
            'forecastRandStd': 0.15,
            'forecastSource': None,
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 3,
            'isPPDBk': True,
            'clo': 0.8,
            'met': 1.2, 
            'airVel': 0.1});

register(
    id='IW-imp-v9706',
    entry_point='eplus_env.envs:IW_IMP_V97',
    kwargs={'site_server_ip': 'localhost',
            'rd_port': 61221,
            'wt_port': 61222,
            'env_name': 'IW-imp-v9706',
            'defaultObValues': [50, 50, 0, 0, 50, 72, 72, 72, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            'localLat': 40.4406,
            'localLong': -79.9959,
            'ctrl_step_size_s': 300,
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-imp-v9701'],
            'incl_forecast': False,
            'forecastRandMode': 'tmy3',
            'forecastRandStd': 0.15,
            'forecastSource': None,
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 3,
            'isPPDBk': True,
            'clo': 0.8,
            'met': 1.2, 
            'airVel': 0.1,
            'isMullSspLowerLimit': True});

register(
    id='IW-imp-v9707',
    entry_point='eplus_env.envs:IW_IMP_V97',
    kwargs={'site_server_ip': 'localhost',
            'rd_port': 61221,
            'wt_port': 61222,
            'env_name': 'IW-imp-v9707',
            'defaultObValues': [50, 50, 0, 0, 50, 72, 72, 72, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            'localLat': 40.4406,
            'localLong': -79.9959,
            'ctrl_step_size_s': 300,
            'min_max_limits': MIN_MAX_LIMITS_DICT['IW-imp-v9701'],
            'incl_forecast': False,
            'forecastRandMode': 'tmy3',
            'forecastRandStd': 0.15,
            'forecastSource': None,
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 3,
            'isPPDBk': True,
            'clo': 0.8,
            'met': 1.2, 
            'airVel': 0.1,
            'isMullSspLowerLimit': True,
            'useCSLWeather': True});

register(
    id='Model1-Cool-v1',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/csl_vav_dx/learning/cfg/cooling_1.cfg',
            'idf_path':FD + '/envs/eplus_models/csl_vav_dx/learning/idf/1.csl.vavDx.light.pittsburgh.cool.idf',
            'env_name': 'Model1-Cool-v1',
            'min_max_limits': MIN_MAX_LIMITS_DICT['Model1-Cool-v1'],
            'incl_forecast': False,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': None,
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 1});

register(
    id='Model1-Test-Cool-v1',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh_test.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/csl_vav_dx/learning/cfg/cooling_1.cfg',
            'idf_path':FD + '/envs/eplus_models/csl_vav_dx/learning/idf/1.csl.vavDx.light.pittsburgh.cool.test.idf',
            'env_name': 'Model1-Test-Cool-v1',
            'min_max_limits': MIN_MAX_LIMITS_DICT['Model1-Cool-v1'],
            'incl_forecast': False,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': None,
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 1});

# v2 version includes AHU Setpoint in the state
register(
    id='Model1-Cool-v2',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/csl_vav_dx/learning/cfg/cooling_2.cfg',
            'idf_path':FD + '/envs/eplus_models/csl_vav_dx/learning/idf/1.csl.vavDx.light.pittsburgh.cool.idf',
            'env_name': 'Model1-Cool-v2',
            'min_max_limits': MIN_MAX_LIMITS_DICT['Model1-Cool-v2'],
            'incl_forecast': False,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': None,
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 1});

register(
    id='Model1-Test-Cool-v2',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/pittsburgh_test.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/csl_vav_dx/learning/cfg/cooling_2.cfg',
            'idf_path':FD + '/envs/eplus_models/csl_vav_dx/learning/idf/1.csl.vavDx.light.pittsburgh.cool.test.idf',
            'env_name': 'Model1-Test-Cool-v2',
            'min_max_limits': MIN_MAX_LIMITS_DICT['Model1-Cool-v2'],
            'incl_forecast': False,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': None,
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 1});

register(
    id='Model5-Cool-v1',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/beijing.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/csl_vav_dx/learning/cfg/cooling_1.cfg',
            'idf_path':FD + '/envs/eplus_models/csl_vav_dx/learning/idf/5.csl.vavDx.medium.beijing.cool.idf',
            'env_name': 'Model5-Cool-v1',
            'min_max_limits': MIN_MAX_LIMITS_DICT['Model2-Cool-v1'],
            'incl_forecast': False,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': None,
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 1});

register(
    id='Model5-Test-Cool-v1',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-3-0/',
            'weather_path':FD + '/envs/weather/beijing_2017.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/eplus_models/csl_vav_dx/learning/cfg/cooling_1.cfg',
            'idf_path':FD + '/envs/eplus_models/csl_vav_dx/learning/idf/5.csl.vavDx.medium.beijing.cool.test.idf',
            'env_name': 'Model5-Test-Cool-v1',
            'min_max_limits': MIN_MAX_LIMITS_DICT['Model5-Cool-v1'],
            'incl_forecast': False,
            'forecastRandMode': 'normal',
            'forecastRandStd': 0.15,
            'forecastSource': None,
            'forecastFilePath': None,
            'forecast_hour': 12,
            'act_repeat': 1});

def setSchedulePath(sch_path_dict, tgtIDFPath):
    """
    Set the abosolute path for the Schedule:File.
    """
    contents = None;
    with open(tgtIDFPath, 'r', encoding = 'ISO-8859-1') as idf:
        contents = idf.readlines();
        remember_str = None;
        remember_idx = -1;
        schedule_on = False;
        i = 0;
        for line in contents:
            effectiveContent = line.strip().split('!')[0] # Ignore contents after '!'
            effectiveContent = effectiveContent.strip().split(',')[0]
                                                            # Remove tailing ','
            if effectiveContent == 'Schedule:File':
                schedule_on = True;
            if effectiveContent in list(sch_path_dict.keys()):
                if schedule_on:
                    remember_str = effectiveContent;
                    remember_idx = i + 2; 
            if i == remember_idx:
                contents[i] = sch_path_dict[remember_str] + ', !- File Name-eval_env' + '\n';
                schedule_on = False;
            i += 1;
    with open(tgtIDFPath, 'w', encoding = 'ISO-8859-1') as idf:
        idf.writelines(contents);

# Replace some schedule file path in 5ZoneAutoDXVAV_eval.idf with the 
# absolute path
"""for legacy env
sch_path_dict = {'S1_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v0.csv',
                 'S2_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v0.csv',
                 'S3_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v0.csv',
                 'S4_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v0.csv',
                 'S5_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v0.csv',
                 'S1_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v0.csv',
                 'S2_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v0.csv',
                 'S3_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v0.csv',
                 'S4_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v0.csv',
                 'S5_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v0.csv'};
setSchedulePath(sch_path_dict, FD + '/envs/idf/envs/v0/learning/5ZoneAutoDXVAV_eval_v0.idf');
# Replace some schedule file path in 5ZoneAutoDXVAV_eval_v1.idf with the 
# absolute path
sch_path_dict = {'S1_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v1.csv',
                 'S2_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v1.csv',
                 'S3_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v1.csv',
                 'S4_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v1.csv',
                 'S5_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v1.csv',
                 'S1_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v1.csv',
                 'S2_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v1.csv',
                 'S3_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v1.csv',
                 'S4_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v1.csv',
                 'S5_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v1.csv'};
setSchedulePath(sch_path_dict, FD + '/envs/idf/envs/v1/learning/5ZoneAutoDXVAV_eval_v1.idf');

# Replace some schedule file path in 5ZoneAutoDXVAV_multiagent_eval_v1.idf with the 
# absolute path
sch_path_dict = {'S1_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v1.csv',
                 'S2_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v1.csv',
                 'S3_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v1.csv',
                 'S4_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v1.csv',
                 'S5_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v1.csv',
                 'S1_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v1.csv',
                 'S2_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v1.csv',
                 'S3_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v1.csv',
                 'S4_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v1.csv',
                 'S5_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v1.csv'};
setSchedulePath(sch_path_dict, FD + '/envs/idf/envs/v1/learning/5ZoneAutoDXVAV_multiagent_eval_v1.idf');


# Replace some schedule file path in 5ZoneAutoDXVAV_eval_v2.idf with the 
# absolute path
sch_path_dict = {'S1_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v2.csv',
                 'S2_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v2.csv',
                 'S3_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v2.csv',
                 'S4_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v2.csv',
                 'S5_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v2.csv',
                 'S1_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v2.csv',
                 'S2_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v2.csv',
                 'S3_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v2.csv',
                 'S4_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v2.csv',
                 'S5_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v2.csv'};
setSchedulePath(sch_path_dict, FD + '/envs/idf/envs/v2/learning/5ZoneAutoDXVAV_eval_v2.idf');

# Replace some schedule file path in 5ZoneAutoDXVAV_eval_v3.idf with the 
# absolute path
sch_path_dict = {'S1_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v3.csv',
                 'S2_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v3.csv',
                 'S3_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v3.csv',
                 'S4_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v3.csv',
                 'S5_Office_-_Private People Schedule': FD + '/envs/idf/schedules/stochastic_occup_v3.csv',
                 'S1_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v3.csv',
                 'S2_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v3.csv',
                 'S3_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v3.csv',
                 'S4_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v3.csv',
                 'S5_Office_-_Private Equip Schedule': FD + '/envs/idf/schedules/stochastic_equip_v3.csv'};
setSchedulePath(sch_path_dict, FD + '/envs/idf/envs/v3/learning/5ZoneAutoDXVAV_eval_v3.idf');

# Replace some schedule file path in iw_v57_learning_eval.idf with the 
# absolute path
sch_path_dict = {'oat_2017': FD + '/envs/idf/envs/iw/learning/x.csv',
                 'oah_2017': FD + '/envs/idf/envs/iw/learning/x.csv',
                 'oadwp_2017': FD + '/envs/idf/envs/iw/learning/x.csv',
                 'oawds_2017': FD + '/envs/idf/envs/iw/learning/x.csv',
                 'oawdd_2017': FD + '/envs/idf/envs/iw/learning/x.csv',
                 'solDir_2017': FD + '/envs/idf/envs/iw/learning/x.csv',
                 'solDif_2017': FD + '/envs/idf/envs/iw/learning/x.csv'};
setSchedulePath(sch_path_dict, FD + '/envs/idf/envs/iw/learning/iw_v57_learning_eval.idf');

# Replace some schedule file path in iw_v57_2_learning_eval.idf with the 
# absolute path
sch_path_dict = {'oat_2017': FD + '/envs/idf/envs/iw/learning/x.csv',
                 'oah_2017': FD + '/envs/idf/envs/iw/learning/x.csv',
                 'oadwp_2017': FD + '/envs/idf/envs/iw/learning/x.csv',
                 'oawds_2017': FD + '/envs/idf/envs/iw/learning/x.csv',
                 'oawdd_2017': FD + '/envs/idf/envs/iw/learning/x.csv',
                 'solDir_2017': FD + '/envs/idf/envs/iw/learning/x.csv',
                 'solDif_2017': FD + '/envs/idf/envs/iw/learning/x.csv'};
setSchedulePath(sch_path_dict, FD + '/envs/idf/envs/iw/learning/iw_v57_2_learning_eval.idf');
"""
# Replace some schedule file path in /envs/eplus_models/iw_v96/learning/idf/realWeather.idf with the absolute path
sch_path_dict = {'oat_2017': FD + '/envs/eplus_models/iw_v96/weather/x.csv',
                 'oah_2017': FD + '/envs/eplus_models/iw_v96/weather/x.csv',
                 'oadwp_2017': FD + '/envs/eplus_models/iw_v96/weather/x.csv',
                 'oawds_2017': FD + '/envs/eplus_models/iw_v96/weather/x.csv',
                 'oawdd_2017': FD + '/envs/eplus_models/iw_v96/weather/x.csv',
                 'solDir_2017': FD + '/envs/eplus_models/iw_v96/weather/x.csv',
                 'solDif_2017': FD + '/envs/eplus_models/iw_v96/weather/x.csv'};
setSchedulePath(sch_path_dict, FD + '/envs/eplus_models/iw_v96/learning/idf/realWeather.idf');

sch_path_dict = {'oat_2017': FD + '/envs/eplus_models/iw_v97/weather/x.csv',
                 'oah_2017': FD + '/envs/eplus_models/iw_v97/weather/x.csv',
                 'oadwp_2017': FD + '/envs/eplus_models/iw_v97/weather/x.csv',
                 'oawds_2017': FD + '/envs/eplus_models/iw_v97/weather/x.csv',
                 'oawdd_2017': FD + '/envs/eplus_models/iw_v97/weather/x.csv',
                 'solDir_2017': FD + '/envs/eplus_models/iw_v97/weather/x.csv',
                 'solDif_2017': FD + '/envs/eplus_models/iw_v97/weather/x.csv'};
setSchedulePath(sch_path_dict, FD + '/envs/eplus_models/iw_v97/learning/idf/realWeather.idf');
