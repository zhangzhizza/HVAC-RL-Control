

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


	@property
	def idf_dict(self):
		return self._idf_dict
	






