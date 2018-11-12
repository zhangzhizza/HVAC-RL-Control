import os, copy

class IdfParser(object):

	def __init__(self, idf_dir):
		self._idf_dir = idf_dir;
		# idf_dict is:
		# {idf_class_name:[obj_content_str, obj_content_str]}
		self._idf_dict = {}; 
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

	def remove_objects_all(self, class_name):
		self._idf_dict.pop(class_name);

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

	def localize_schedule(self, local_file_path):
		file_name = local_file_path.split(os.sep)[-1];
		file_dir = local_file_path[:local_file_path.rfind(os.sep)];
		sch_file_contents = self._idf_dict['Schedule:File'];
		content_i = 0;
		for sch_file_obj in copy.deepcopy(sch_file_contents):
			if file_name in sch_file_obj:
				file_name_st_idx = sch_file_obj.rfind(file_name);
				full_path_st_idx = sch_file_obj.rfind(',', 0, file_name_st_idx);
				sch_file_obj = sch_file_obj[0:full_path_st_idx] + ',\n' + file_dir + sch_file_obj[file_name_st_idx:];
				sch_file_contents[content_i] = sch_file_obj;
			content_i += 1;
		self._idf_dict['Schedule:File'] = sch_file_contents;

	def add_objects(self, dict_to_add):
		for key in dict_to_add:
			objects_to_add = dict_to_add[key];
			if key in self._idf_dict:
				self._idf_dict[key].extend(objects_to_add);
			else:
				self._idf_dict[key] = objects_to_add;


	@property
	def idf_dict(self):
		return self._idf_dict
	






