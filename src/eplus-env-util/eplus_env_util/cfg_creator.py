import eplus_env_util.idf_parser as idf
import os

FD = os.path.dirname(os.path.realpath(__file__));
CFG_HEAD = ('<?xml version="1.0" encoding="ISO-8859-1"?>\n' + 
		   '<!DOCTYPE BCVTB-variables SYSTEM "variables.dtd">\n' +
		   '<BCVTB-variables>\n')
CFG_RECV = ('<variable source="EnergyPlus">\n' +
   			'<EnergyPlus name="%s" type="%s"/>\n' +
  			'</variable>\n')
CFG_SEND = ('<variable source="Ptolemy">\n' +
   			'<EnergyPlus schedule="%s"/>\n' +
 			'</variable>\n')

class CfgCreator(object):

	def __init__(self):
		pass;

	def create_cfg(self, add_idf_path, out_cfg_path):
		# Write cdg head 
		with open(out_cfg_path, 'w') as cfg_file:
			cfg_file.write(CFG_HEAD);
		# Write cfg eplus out
		add_idf = idf.IdfParser(add_idf_path);
		with open(out_cfg_path, 'a') as cfg_file:
			cfg_file.write('<!-- Recieve from E+ -->\n');
			for output_var in add_idf.idf_dict['Output:Variable']:
				output_var_list = output_var.split(',');
				output_var_key = output_var_list[0].split('\n')[-1].strip();
				output_var_name = output_var_list[1].split('\n')[-1].strip();
				cfg_file.write(CFG_RECV%(output_var_key, output_var_name));
			# Write cfg eplus in
			cfg_file.write('<!-- Send to E+ -->\n');
			for input_var in add_idf.idf_dict['ExternalInterface:Schedule']:
				input_var_list = input_var.split(',');
				input_var_name = input_var_list[0].split('\n')[-1].strip();
				cfg_file.write(CFG_SEND%(input_var_name));





