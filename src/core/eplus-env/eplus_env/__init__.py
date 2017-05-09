from gym.envs.registration import register
import os
import fileinput

FD = os.path.dirname(os.path.realpath(__file__));

register(
    id='Eplus-v0',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/variables_v0.cfg',
            'idf_path':FD + '/envs/5ZoneAutoDXVAV.idf',
            'env_name': 'Eplus-v0'});

register(
    id='Eplus-v1',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/variables_v1.cfg',
            'idf_path':FD + '/envs/5ZoneAutoDXVAV_v1.idf',
            'env_name': 'Eplus-v1'});

register(
    id='Eplus-forecast-v0',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/pittsburgh.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/variables_v0.cfg',
            'idf_path':FD + '/envs/5ZoneAutoDXVAV.idf',
            'incl_forecast': True,
            'forecast_step': 36,
            'env_name': 'Eplus-forecast-v0'});
    
register(
    id='Eplus-eval-v0',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/pennstate.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/variables_v0.cfg',
            'idf_path':FD + '/envs/5ZoneAutoDXVAV_eval.idf',
            'env_name': 'Eplus-eval-v0'});

register(
    id='Eplus-eval-v1',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/pennstate.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/variables_v1.cfg',
            'idf_path':FD + '/envs/5ZoneAutoDXVAV_eval_v1.idf',
            'env_name': 'Eplus-eval-v1'});
    
register(
    id='Eplus-eval-multiagent-v0',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/',
            'weather_path':FD + '/envs/pennstate.epw',
            'bcvtb_path':FD + '/envs/bcvtb/',
            'variable_path':FD + '/envs/variables_multiagent.cfg',
            'idf_path':FD + '/envs/5ZoneAutoDXVAV_eval_multiagent.idf',
            'env_name': 'Eplus-eval-multiagent-v0'});

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
sch_path_dict = {'S1_Office_-_Private People Schedule': FD + '/envs/stochastic_occup.csv',
                 'S2_Office_-_Private People Schedule': FD + '/envs/stochastic_occup.csv',
                 'S3_Office_-_Private People Schedule': FD + '/envs/stochastic_occup.csv',
                 'S4_Office_-_Private People Schedule': FD + '/envs/stochastic_occup.csv',
                 'S5_Office_-_Private People Schedule': FD + '/envs/stochastic_occup.csv',
                 'S1_Office_-_Private Equip Schedule': FD + '/envs/stochastic_equip.csv',
                 'S2_Office_-_Private Equip Schedule': FD + '/envs/stochastic_equip.csv',
                 'S3_Office_-_Private Equip Schedule': FD + '/envs/stochastic_equip.csv',
                 'S4_Office_-_Private Equip Schedule': FD + '/envs/stochastic_equip.csv',
                 'S5_Office_-_Private Equip Schedule': FD + '/envs/stochastic_equip.csv'};
setSchedulePath(sch_path_dict, FD + '/envs/5ZoneAutoDXVAV_eval.idf');
# Replace some schedule file path in 5ZoneAutoDXVAV_eval_v1.idf with the 
# absolute path
sch_path_dict = {'S1_Office_-_Private People Schedule': FD + '/envs/stochastic_occup_v1.csv',
                 'S2_Office_-_Private People Schedule': FD + '/envs/stochastic_occup_v1.csv',
                 'S3_Office_-_Private People Schedule': FD + '/envs/stochastic_occup_v1.csv',
                 'S4_Office_-_Private People Schedule': FD + '/envs/stochastic_occup_v1.csv',
                 'S5_Office_-_Private People Schedule': FD + '/envs/stochastic_occup_v1.csv',
                 'S1_Office_-_Private Equip Schedule': FD + '/envs/stochastic_equip_v1.csv',
                 'S2_Office_-_Private Equip Schedule': FD + '/envs/stochastic_equip_v1.csv',
                 'S3_Office_-_Private Equip Schedule': FD + '/envs/stochastic_equip_v1.csv',
                 'S4_Office_-_Private Equip Schedule': FD + '/envs/stochastic_equip_v1.csv',
                 'S5_Office_-_Private Equip Schedule': FD + '/envs/stochastic_equip_v1.csv'};
setSchedulePath(sch_path_dict, FD + '/envs/5ZoneAutoDXVAV_eval_v1.idf');