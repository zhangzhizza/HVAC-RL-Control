import eplus_env_util.idf_parser as idf
import os

FD = os.path.dirname(os.path.realpath(__file__));
GYM_INIT_PATH = FD + '/../eplus-env/eplus_env/__init__.py';
GYM_ENVLIMIT_PATH = FD + '/../eplus-env/eplus_env/eplus_env_statelimits.py';
GYM_REG_TEMPLATE = ('\nregister(\nid=\'%s\',\nentry_point=\'eplus_env.envs:EplusEnv\',\n'
					'kwargs={\n\'eplus_path\':FD + \'/envs/EnergyPlus-8-3-0/\',\n'
            				'\'weather_path\':\'%s\',\n'
            				'\'bcvtb_path\':FD + \'/envs/bcvtb/\',\n'
            				'\'variable_path\':\'%s\',\n'
            				'\'idf_path\':\'%s\',\n'
            				'\'env_name\':\'%s\',\n'
            				'\'min_max_limits\': MIN_MAX_LIMITS_DICT[\'%s\'],\n'
            				'\'incl_forecast\': False,\n'
            				'\'forecastRandMode\': \'normal\',\n'
            				'\'forecastRandStd\': 0.15,\n'
            				'\'forecastSource\': None,\n'
            				'\'forecastFilePath\': None,\n'
            				'\'forecast_hour\': 12,\n'
            				'\'act_repeat\': 1});')

class EplusEnvCreator(object):

	def __init__(self):
		pass;

	def create_env(self, source_idf_path, add_idf_path, cfg_path, 
					env_name, weather_path, schedule_file_paths = []):
		# Create a new idf file with the addtional contents
		source_idf = idf.IdfParser(source_idf_path);
		add_idf = idf.IdfParser(add_idf_path);
		# Remove the original output variable
		source_idf.remove_objects_all('Output:Variable') 
		# Remove the schedules in the original idf
		tgt_class_name_in_add = 'ExternalInterface:Schedule';
		tgt_sch_names_in_org = [source_idf.get_object_name(add_content) 
								for add_content in add_idf.idf_dict[tgt_class_name_in_add]]
		tgt_class_name_in_org = 'Schedule:Compact';
		for to_rm_obj_name in tgt_sch_names_in_org:
			source_idf.remove_object(tgt_class_name_in_org, to_rm_obj_name);
		# Localize the schedule files
		for schedule_file_path in schedule_file_paths:
			source_idf.localize_schedule(schedule_file_path)
		# Add the addition to the source idf
		source_idf.add_objects(add_idf.idf_dict);
		# Write the new idf out. The name has '.env' before the file idf extension
		new_idf_name_add_idx = source_idf_path.rfind('.idf');
		new_idf_name = source_idf_path[:new_idf_name_add_idx] + '.env' + source_idf_path[new_idf_name_add_idx:];
		source_idf.write_idf(new_idf_name);
		# Create a new env in the gym __init__ file
		gym_register = GYM_REG_TEMPLATE%(env_name, weather_path, cfg_path, new_idf_name, env_name, env_name);
		with open(GYM_INIT_PATH, 'a') as init_file:
			init_file.write(gym_register);





