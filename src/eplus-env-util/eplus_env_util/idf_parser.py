import os, copy, time,subprocess
FD = os.path.dirname(os.path.realpath(__file__))
EPLUS_PATH = {'8_3':FD + '/../../eplus-env/eplus_env/envs/EnergyPlus-8-3-0'}
WEATHER_PATH_DF = FD + '/../../eplus-env/eplus_env/envs/weather/pittsburgh.epw'

class IdfParser(object):

	def __init__(self, idf_dir, version='8_3'):
		self._idf_dir = idf_dir;
		# idf_dict is:
		# {idf_class_name:[obj_content_str, obj_content_str]}
		self._idf_dict = {};
		self._version = version;
		self._parser_idf();

	def _parser_idf(self):
		with open(self._idf_dir, 'r') as idf_file:
			idf_lines = idf_file.readlines();
			is_obj_start = False;
			obj_content = '';
			obj_name = '';
			for idf_line in idf_lines:
				idf_line_prcd = idf_line.split('\n')[0].split('!')[0].strip();
				if is_obj_start == False:
					if len(idf_line_prcd) > 0:
						if idf_line_prcd[-1] == ',':
							obj_name = idf_line_prcd[:-1];
							is_obj_start = True;
				else:
					obj_content += idf_line;
					if len(idf_line_prcd) > 0:
						if idf_line_prcd[-1] == ';':
							if obj_name in self._idf_dict:
								self._idf_dict[obj_name].append(obj_content);
							else:
								self._idf_dict[obj_name] = [obj_content];
							# Reset obj temp fields
							is_obj_start = False;
							obj_content = '';
							obj_name = '';

	def write_idf(self, to_write_dir):
		to_write_str = '';
		# Construct the string to write
		for idf_obj_name in self._idf_dict:
			obj_contents = self._idf_dict[idf_obj_name];
			for obj_content in obj_contents:
				to_write_str += idf_obj_name + ',\n';
				to_write_str += obj_content + '\n';
		with open(to_write_dir, 'w') as idf_file:
			idf_file.write(to_write_str);

	def write_object_in_idf(self, to_write_dir, object_name):
		to_write_str = '';
		# Construct the string to write
		obj_contents = self._idf_dict[object_name];
		for obj_content in obj_contents:
			to_write_str += object_name + ',\n';
			to_write_str += obj_content + '\n';
		with open(to_write_dir, 'w') as idf_file:
			idf_file.write(to_write_str);


	def remove_objects_all(self, class_name):
		self._idf_dict.pop(class_name);

	def get_obj_reference_count(self, obj_name):
		ref_ct = 0;
		for key, value in self._idf_dict.items():
			for obj in value:
				obj_lines = obj.split(',')[1: ] # Exclude the obj name itself from the reference
				for obj_line in obj_lines:
					effc_obj_line = obj_line.split('\n')[-1].strip();
					if obj_name == effc_obj_line: 
						ref_ct += 1;
		return ref_ct;


	def remove_object(self, class_name, obj_name):
		try:
			tgt_objects = self._idf_dict[class_name];
			tgt_idx = 0;
			for obj in tgt_objects:
				obj_name_this = self.get_object_name(obj);
				if obj_name_this == obj_name:
					break;
				else:
					tgt_idx += 1;
			self._idf_dict[class_name].pop(tgt_idx);
		except Exception as e:
			print('Func: remove_object, args:(%s, %s), error: %s'%(class_name, obj_name, traceback.format_exc()))

	def get_object_name(self, object_content):
		obj_name = object_content.split(',')[0].split('\n')[-1].strip();
		return obj_name;

	def get_schedule_type_init_value(self, schedule_name):
		schedule_content = None;
		for cmp_schedule_content in self._idf_dict['Schedule:Compact']:
			if self.get_object_name(cmp_schedule_content) == schedule_name:
				schedule_content = cmp_schedule_content;
				break;
		schedule_content = schedule_content.split(';')[0].split(',');
		schedule_type = schedule_content[1].split('\n')[-1].strip();
		# Schedule init value
		for schedule_line_i in schedule_content[2:]:
			try:
				init_value = float(schedule_line_i.split('\n')[-1].strip());
				break;
			except Exception as e:
				pass;
		return (schedule_type, init_value);


	def get_all_compact_schedules_names(self):
		returned_list = [];
		for cmp_schedule_content in self._idf_dict['Schedule:Compact']:
			returned_list.append(self.get_object_name(cmp_schedule_content));
		return returned_list;

	def localize_schedule(self, local_file_path):
		file_name = local_file_path.split(os.sep)[-1];
		file_dir = local_file_path[:local_file_path.rfind(os.sep)];
		sch_file_contents = self._idf_dict['Schedule:File'];
		content_i = 0;
		for sch_file_obj in copy.deepcopy(sch_file_contents):
			if file_name in sch_file_obj:
				file_name_st_idx = sch_file_obj.rfind(file_name);
				full_path_st_idx = sch_file_obj.rfind(',', 0, file_name_st_idx);
				sch_file_obj = sch_file_obj[0:full_path_st_idx] + ',\n' + file_dir + '/' + sch_file_obj[file_name_st_idx:];
				sch_file_contents[content_i] = sch_file_obj;
			content_i += 1;
		self._idf_dict['Schedule:File'] = sch_file_contents;

	def is_contain_filesch(self):
		result = 'Schedule:File' in self._idf_dict
		return (result);

	def add_objects(self, dict_to_add):
		for key in dict_to_add:
			objects_to_add = dict_to_add[key];
			if key in self._idf_dict:
				self._idf_dict[key].extend(objects_to_add);
			else:
				self._idf_dict[key] = objects_to_add;

	def add_dxf_output(self):
		self._idf_dict['Output:Surfaces:Drawing'] = ['DXF,!- Report Type\n'+
													 'Triangulate3DFace;\n'];

	def set_minimum_run(self):
		self._idf_dict['SimulationControl'] = ['Yes,!- Do Zone Sizing Calculation\n' +
    									'No,!- Do System Sizing Calculation\n' +
                                        'No,!- Do Plant Sizing Calculation\n' +
    									'No,!- Run Simulation for Sizing Periods\n' +
    									'No;!- Run Simulation for Weather File Run Periods\n'];
		if 'Schedule:File' in self._idf_dict:
			self._idf_dict.pop('Schedule:File', None);

	def run_eplus_minimum(self, out_dir):
		eplus_path_this = EPLUS_PATH[self._version];
		if not os.path.isdir(FD + '/tmp'):
			os.makedirs(FD + '/tmp');
		idf_dir = FD + '/tmp/%s.idf'%(time.time());
		self.write_idf(idf_dir);
		print ('%s -w %s -d %s %s'
						%(eplus_path_this + '/energyplus', WEATHER_PATH_DF, 
                          out_dir, idf_dir))
		eplus_process = subprocess.call('%s -w %s -d %s %s'
						%(eplus_path_this + '/energyplus', WEATHER_PATH_DF, 
                          out_dir, idf_dir),
                        shell = True,
                        stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE,
                        preexec_fn=os.setsid)


    



	@property
	def idf_dict(self):
		return self._idf_dict
	






