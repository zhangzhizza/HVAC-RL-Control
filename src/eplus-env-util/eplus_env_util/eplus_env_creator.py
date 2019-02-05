import eplus_env_util.idf_parser as idf
import pandas as pd
import os

FD = os.path.dirname(os.path.realpath(__file__));
GYM_INIT_PATH = FD + '/../../eplus-env/eplus_env/__init__.py';
GYM_ENVLIMIT_PATH = FD + '/../eplus-env/eplus_env/eplus_env_statelimits.py';
GYM_REG_TEMPLATE = ('\nregister(\nid=\'%s\',\nentry_point=\'eplus_env.envs:EplusEnv\',\n'
					'kwargs={\'eplus_path\':FD + \'/envs/EnergyPlus-%s/\',\n'
            				'\'weather_path\':\'%s\',\n'
            				'\'bcvtb_path\':FD + \'/envs/bcvtb/\',\n'
            				'\'variable_path\':\'%s\',\n'
            				'\'idf_path\':\'%s\',\n'
            				'\'env_name\':\'%s\',\n'
            				'\'min_max_limits\': %s,\n'
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

	def get_existing_env_names(self):
		env_names = [];
		with open(GYM_INIT_PATH, 'r') as init_file:
			init_lines = init_file.readlines();
			for init_line in init_lines:
				if 'id=' in init_line:
					env_id = init_line.split('id=')[-1].split(',')[0][1:-1];
					env_names.append(env_id);
		return env_names;


	def create_env(self, source_idf_path, add_idf_path, cfg_path, 
					env_name, weather_path, state_limit_path, schedule_file_paths = [],
					eplus_version = '8-3-0'):
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
		# Check whether or not the tgt_sch_names have been actually used in the source idf
		for tgt_sch_name in tgt_sch_names_in_org:
			tgt_sch_ref_ct = source_idf.get_obj_reference_count(tgt_sch_name);
			if tgt_sch_ref_ct < 1:
				print('WARNING!!!!! The target schedule %s may not be used the source IDF.'%tgt_sch_name)
		# Localize the schedule files
		for schedule_file_path in schedule_file_paths:
			source_idf.localize_schedule(schedule_file_path)
		# Add the addition to the source idf
		source_idf.add_objects(add_idf.idf_dict);
		# Write the new idf out. The name has '.env' before the file idf extension
		new_idf_name = source_idf_path + '.env';
		source_idf.write_idf(new_idf_name);
		# State limits
		state_limits_df = pd.read_csv(state_limit_path, sep=',', header=None);
		state_limits_array = state_limits_df.values;
		state_limits = [];
		for col_i in range(state_limits_array.shape[1]):
			state_limits.append((state_limits_array[0, col_i], state_limits_array[1, col_i]));
		# Create a new env in the gym __init__ file
		gym_register = GYM_REG_TEMPLATE%(env_name, eplus_version, weather_path, cfg_path, 
			new_idf_name, env_name, state_limits);
		if env_name not in self.get_existing_env_names():
			with open(GYM_INIT_PATH, 'a') as init_file:
				init_file.write(gym_register);
			return 0;
		else:
			return 1;





