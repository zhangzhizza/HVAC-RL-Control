from gym.envs.registration import register
import os
import fileinput

FD = os.path.dirname(os.path.realpath(__file__));

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