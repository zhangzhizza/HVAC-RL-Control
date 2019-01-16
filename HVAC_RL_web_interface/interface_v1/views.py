from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from xml.dom.minidom import parseString
from django.contrib.auth.decorators import login_required
from .forms import *

import requests, json
import pandas as pd
import numpy as np
import logging, time, traceback
import os, shutil, subprocess, json, socket, ast
import eplus_env_util.idf_parser as idf_parser
import eplus_env_util.cfg_creator as cfg_creator
import eplus_env_util.eplus_env_creator as eplus_env_creator;

this_dir_path = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE_PATH = this_dir_path + '/../configurations/configurations.json';
GYM_INIT_PATH = this_dir_path + '/../../src/eplus-env/eplus_env/__init__.py';
RUNS_PATH = this_dir_path + '/../../src/runs/';
eplus_model_path = this_dir_path + '/../../src/eplus-env/eplus_env/envs/eplus_models/';
worker_server = json.load(open(CONFIG_FILE_PATH, 'r'))['worker_server_addr']

logger = logging.getLogger(__name__)

"""
When the idf_file is uploaded, it is firstly copied to idf_file.local, and then an add file
idf_file.add is generated, and the cfg file idf_file.cfg is generated.
"""

# Create your views here.
def get_info_from_config_file(key):
	with open(CONFIG_FILE_PATH, 'r') as config_f:
		config_data = json.load(config_f);
	return config_data[key];

@login_required
def index(request):
	run_dirs_names, run_dirs = getRuns();
	return render(request, 'interface_v1/html/srtdash/index.html',\
    	{'run_dirs_names': run_dirs_names,
    	 'available_computers': get_info_from_config_file('available_worker_clients')})

@login_required
def openJSCAD(request):
	group_name = request.GET.get('group_name')
	idf_name = request.GET.get('idf_name')
	dxf_dir = (this_dir_path + '/../../src/eplus-env/eplus_env/envs/eplus_models/'
						+ group_name + '/idf/' + idf_name + '_base_out/eplusout.dxf');
	with open(dxf_dir, 'r') as dxf_file:
		dxf_content = dxf_file.read();
	print ('\n' in dxf_content)
	dxf_content = dxf_content.replace("\n", "\\n")
	return render(request, 'interface_v1/html/OpenJSCAD/index.html', {'dxf_content': [dxf_content]});

@login_required
def test(request):
	action_select = SelectActionForm(options = (('1','e'),('1','2')))
	return HttpResponse(action_select);

@login_required
def get_all_envs(request):
	# List all available envs
	all_envs = [];
	with open(GYM_INIT_PATH, 'r') as init_file:
		init_lines = init_file.readlines();
	line_i = 0;
	while line_i < len(init_lines):
		init_line_i = init_lines[line_i];
		if ('register(' in init_line_i):
			this_env = [];
			# Env id
			env_id = init_lines[line_i + 1].split(',')[0].split('id=')[-1][1:-1];
			this_env.append(env_id)
			# Epw
			epw = init_lines[line_i + 4].split(',')[0].split('/')[-1][0:-1];
			this_env.append(epw);
			# idf
			idf = init_lines[line_i + 7].split(',')[0].split('/')[-1][0:-1];
			this_env.append(idf)
			# Min-max limit
			minmax = init_lines[line_i + 9][0:-1].split(':')[-1];
			this_env.append(minmax);
			all_envs.append(this_env);
			# Increment index
			line_i += 17;
		else:
			line_i += 1;
	all_env_pd = pd.DataFrame(all_envs);
	all_env_pd.columns = ['ID', 'Weather', 'IDF', 'Min-max Limits']
	return HttpResponse(all_env_pd.to_html());

@login_required
def simulator_eplus(request):
	form = UploadFileForm()
	action_select = ''#SelectActionForm(options = [['','']])
	sch_select = ''#SelectSchForm(options = [['','']])
	minmax_limit_form = MinMaxLimitsForm();

	return render(request, 'interface_v1/html/srtdash/simulator.html', 
		{'form': form, 'action_select':action_select, 
		 'sch_select': sch_select, 'minmax_limit_form': minmax_limit_form});

@login_required
def simulator_eplus_idf_upload(request):
	if request.method == 'POST':
		form = UploadFileForm(request.POST, request.FILES)
		if form.is_valid():
			print (request.FILES)
			file_epw = request.FILES['file_epw'] if 'file_epw' in request.FILES else None;
			file_sch = request.FILES['file_sch'] if 'file_sch' in request.FILES else None;
			handle_uploaded_idf_file(request.POST['title'], request.FILES['file_idf'],
									 file_epw, file_sch)
			return HttpResponse('simulator_eplus/openjscad/'+
								'?group_name=%s&idf_name=%s'
								%(request.POST['title'], request.FILES['file_idf'].name));

@login_required
def generate_epw_names(request):
	# Get common variables
	epw_path = eplus_model_path + '/../weather';
	epws = os.listdir(epw_path)
	epws = sorted(epws);
	form_options = [];
	for i in range(len(epws)):
		form_options.append([epws[i], epws[i]])
	wea_select = SelectWeaForm(options = form_options);
	return HttpResponse(wea_select)

@login_required
def generate_idf_fileschedule_names(request):
	# Get common variables
	group_name = request.GET.get('group_name');
	idf_name = request.GET.get('idf_name');
	env_path = eplus_model_path + group_name;
	env_idf_store_dir = (env_path + '/idf/' + idf_name + '.local');
	org_idf_parser = idf_parser.IdfParser(env_idf_store_dir);
	if request.method == 'POST':
		query = dict(request.POST.lists())
		selected_sch = query['SCHEDULE'];
		org_idf_parser.localize_schedule(env_path + '/schedules/' + selected_sch[0])
		new_idf_name = env_idf_store_dir; 
		org_idf_parser.write_idf(new_idf_name);
		return HttpResponse('hELLO');
	if (org_idf_parser.is_contain_filesch()):
		sch_names = sorted(get_fileschedule_names(group_name));
		form_options = [];
		for i in range(len(sch_names)):
			form_options.append([sch_names[i], sch_names[i]])
		filesch_select = SelectSchForm(options = form_options);
	else:
		filesch_select = None;
	return HttpResponse(filesch_select)

@login_required
def get_state_names(request):
	group_name = request.GET.get('group_name');
	idf_name = request.GET.get('idf_name');
	env_path = eplus_model_path + group_name;
	add_idf_store_dir = (env_path + '/idf/' + idf_name + '.add');
	add_idf_parser = idf_parser.IdfParser(add_idf_store_dir);
	state_names = [];
	for output_var in add_idf_parser.idf_dict['Output:Variable']:
		output_var_list = output_var.split(',');
		output_var_key = output_var_list[0].split('\n')[-1].strip();
		output_var_name = output_var_list[1].split('\n')[-1].strip();
		state_names.append(output_var_key + ':' + output_var_name);
	to_response = {};
	to_response['state_names'] = state_names;
	return JsonResponse(to_response, json_dumps_params={'indent': 2});

@login_required
def get_fileschedule_names(group_name):
	file_name_list = []
	try:
		file_name_list = os.listdir(eplus_model_path + '/' + group_name + '/schedules')
	except Exception as e:
		file_name_list = [];
	return file_name_list;

@login_required	
def create_env(request):
	env_name = request.GET.get('env_name');
	group_name = request.GET.get('group_name');
	idf_name = request.GET.get('idf_name');
	epw_name = request.GET.get('epw_name');
	# Create the cfg file
	create_cfg(group_name, idf_name);
	# Prepare the creator arguments
	idf_model_dir = eplus_model_path + group_name + '/idf/';
	idf_model_path = idf_model_dir + idf_name + '.local';
	cfg_path = idf_model_dir + idf_name + '.cfg';
	add_path = idf_model_dir + idf_name + '.add';
	epw_path = eplus_model_path + '/../weather/' + epw_name;
	limit_path = idf_model_dir + idf_name + '.limit';
	env_creator = eplus_env_creator.EplusEnvCreator();
	create_env_rt = env_creator.create_env(idf_model_path, add_path, cfg_path, 
							env_name, epw_path, limit_path);
	return HttpResponse(create_env_rt);


def create_cfg(group_name, idf_name):
	add_idf_dir = eplus_model_path + group_name + '/idf/';
	add_idf_path = add_idf_dir + idf_name + '.add';
	out_cfg_path = add_idf_dir + idf_name + '.cfg';
	cfg_creator_this = cfg_creator.CfgCreator();
	cfg_creator_this.create_cfg(add_idf_path, out_cfg_path);

@login_required
def generate_minmax_limits(request):
	group_name = request.GET.get('group_name');
	idf_name = request.GET.get('idf_name');
	env_path = eplus_model_path + group_name;
	add_idf_store_dir = (env_path + '/idf/' + idf_name + '.add');
	add_idf_parser = idf_parser.IdfParser(add_idf_store_dir);
	state_num = len(add_idf_parser.idf_dict['Output:Variable'])
	if request.method == 'POST':
		query = dict(request.POST.lists())
		minm = query['minm'][0].split(',');
		maxm = query['maxm'][0].split(',');
		# Convert all inputs to number
		try:
			minm = np.array([float(i) for i in minm]).reshape(1, -1);
			maxm = np.array([float(i) for i in maxm]).reshape(1, -1);
		except Exception as e:
			return HttpResponse(1) # 1: non-number inputs
		# Check input number 
		if not(minm.shape[1] == state_num and maxm.shape[1] == state_num):
			return HttpResponse(2) # 2: not enough or too many input numbers
		# Check max is large than min
		if np.any((maxm - minm) < 0):
			return HttpResponse(3) # 3: max is smaller than min
		# Passed all the checks
		min_max = np.concatenate([minm, maxm]);
		np.savetxt(env_path + '/idf/' + idf_name + '.limit', min_max, delimiter = ',');
	return HttpResponse(0)

@login_required
def generate_idf_schedule_names(request):
	# Get common variables
	group_name = request.GET.get('group_name');
	idf_name = request.GET.get('idf_name');
	env_path = eplus_model_path + group_name + '/idf/';
	env_idf_store_dir = (env_path + idf_name + '.local');
	org_idf_parser = idf_parser.IdfParser(env_idf_store_dir);
	if request.method == 'POST':
		query = dict(request.POST.lists())
		selected_actions = query['ACTIONS'];
		with open(env_path + idf_name + '.add', 'a') as idf_add_file:
			ext_int_head = ('\nExternalInterface,\n' +
							'PtolemyServer;!- Name of External Interface\n');
			idf_add_file.write(ext_int_head)
			for selected_action in selected_actions:
				sch_type, sch_init = org_idf_parser.get_schedule_type_init_value(selected_action);
				ext_sch = ('\nExternalInterface:Schedule,\n' + 
						   '%s,!- Name\n'%(selected_action) +
						   '%s,!- Schedule Type Limits Name\n'%(sch_type) +
    					   '%s;!- Initial Value\n'%(sch_init))
				idf_add_file.write(ext_sch);
		return HttpResponse('hELLO');
	sch_names = sorted(org_idf_parser.get_all_compact_schedules_names());
	form_options = [];
	for i in range(len(sch_names)):
		form_options.append([sch_names[i], sch_names[i]])
	action_select = SelectActionForm(options = form_options);
	# Write the output variable objects to file
	org_idf_parser.write_object_in_idf(env_path + idf_name + '.add', 'Output:Variable')
	return HttpResponse(action_select)

def handle_uploaded_idf_file(group_name, file_idf, file_epw=None, file_sch=None):
	# epw file
	if file_epw is not None:
		env_epw_store_dir = (this_dir_path + '/../../src/eplus-env/eplus_env/envs/weather/');
		with open(env_epw_store_dir + file_epw.name, 'wb') as epw_file:
			for chunk in file_epw.chunks():
				epw_file.write(chunk);
	# sch file
	if file_sch is not None:
		env_sch_store_dir = (this_dir_path + '/../../src/eplus-env/eplus_env/envs/eplus_models/'
									   + group_name + '/schedules/');
		if not os.path.isdir(env_sch_store_dir):
			os.makedirs(env_sch_store_dir);
		with open(env_sch_store_dir + file_sch.name, 'wb') as sch_file:
			for chunk in file_sch.chunks():
				sch_file.write(chunk);
	# idf file
	env_idf_store_dir = (this_dir_path + '/../../src/eplus-env/eplus_env/envs/eplus_models/'
									   + group_name);
	if not os.path.isdir(env_idf_store_dir):
		os.makedirs(env_idf_store_dir + '/idf');
	with open(env_idf_store_dir + '/idf/' + file_idf.name + '.local', 'wb') as idf_file:
		for chunk in file_idf.chunks():
			idf_file.write(chunk);
	org_idf_parser_for_dxf = idf_parser.IdfParser(env_idf_store_dir + '/idf/' + file_idf.name);
	org_idf_parser_for_dxf.add_dxf_output();
	org_idf_parser_for_dxf.set_minimum_run();
	org_idf_parser_for_dxf.run_eplus_minimum(env_idf_store_dir + '/idf/' + file_idf.name + '_base_out')

def getRuns():
	scan_dir = RUNS_PATH;
	dirs = os.listdir(scan_dir);
	run_dirs_names = [];
	run_dirs = []
	for this_dir in dirs:
		if os.path.isdir(scan_dir + this_dir):
			run_dirs_names.append(this_dir);
			run_dirs.append(scan_dir + this_dir)
	return run_dirs_names, run_dirs;

@login_required
def get_eval_res_hist(request):
	ids_list = [];
	to_response = {};
	counter = 1;
	hist_col_num = int(request.GET.get("col")); 
	while True:
		arg_i = "exp%d"%(counter);
		arg_i_value = request.GET.get(arg_i);
		print(arg_i_value)
		if arg_i_value == None:
			break;
		else:
			ids_list.append(arg_i_value);
			counter += 1;
	pd_list = []
	for exp_id in ids_list:
		this_exp_res = [];
		exp_run_name, exp_run_num = exp_id.split(":");
		eval_file_full_dir = (RUNS_PATH + exp_run_name 
						+ '/' + exp_run_num + '/eval_res_hist.csv');
		eval_res_hist_pd = pd.read_csv(eval_file_full_dir, index_col = 0, header = None);
		to_response[exp_id] = eval_res_hist_pd[hist_col_num].reset_index().as_matrix().tolist();

	return JsonResponse(to_response, json_dumps_params={'indent': 2})

@login_required
def get_worker_status(request):
	"""

	"""
	ip = request.GET.get('ip').strip()
	port = int(request.GET.get('port'))
	# Create a socket
	s = socket.socket();
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	interface_server_addr = get_info_from_config_file('interface_server_addr')
	this_server_ip, this_server_port = interface_server_addr.split(':');
	while True:
		try:
			s.bind((this_server_ip, int(this_server_port)))
			break;
		except Exception as e:
			logger.error('Socker binding for getting worker status is unsuccessful with the error: ' 
						+ traceback.format_exc() + ', will retry after 2 seconds.')
			time.sleep(2);
	try:
		s.connect((ip, port));
		s.sendall(b'getstatus');
		recv_str = s.recv(4096).decode(encoding = 'utf-8');
		recv_json = json.loads(recv_str)
		s.close();
		return JsonResponse(recv_json, json_dumps_params={'indent': 2})
	except Exception as e:
		logger.error(traceback.format_exc());
		return JsonResponse({'Error': True}, json_dumps_params={'indent': 2})

@login_required
def reset_exp(request):
	exp_id = request.GET.get('id')
	exp_run_name, exp_run_num = exp_id.split(":");
	# Create a socket to communicate with the workerserver
	s = socket.socket();
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	interface_server_addr = get_info_from_config_file('interface_server_addr')
	this_server_ip, this_server_port = interface_server_addr.split(':');
	while True:
		try:
			s.bind((this_server_ip, int(this_server_port)))
			break;
		except Exception as e:
			logger.error('Socker binding for reseting the run is unsuccessful with the error: ' 
						+ traceback.format_exc() + ', will retry after 2 seconds.')
			time.sleep(2);
	workerserver_ip, workerserver_port = worker_server.split(':');
	workerserver_port = int(workerserver_port) + 2;
	s.connect((workerserver_ip, workerserver_port));
	send_code = '%s:%s'%('resetexp', exp_id)
	s.sendall(bytearray(send_code, encoding = 'utf-8'));
	recv_str = s.recv(1024).decode(encoding = 'utf-8');
	return HttpResponse(recv_str);


@login_required
def run_exp(request):
	exp_id = request.GET.get('id')
	mch_ip = request.GET.get('ip')
	exp_run_name, exp_run_num = exp_id.split(":");
	exp_full_dir = RUNS_PATH + exp_run_name + '/' + exp_run_num;
	to_write_meta = {}
	to_write_meta['status'] = 'starting';
	to_write_meta['machine'] = mch_ip;
	to_write_meta['step'] = 'None';
	return_msg = None;
	ip_idx_in_list = get_ip_idx_in_list(mch_ip);
	if ip_idx_in_list > -1:
		if os.path.isdir(exp_full_dir):
			with open(exp_full_dir + '/run.meta', 'w') as meta_file:
				json.dump(to_write_meta, meta_file);
			# Deploy the exp
			available_worker_clients = get_info_from_config_file('available_worker_clients')
			port = int(available_worker_clients[ip_idx_in_list].split(":")[1])
			recv_msg = deploy_run(exp_full_dir, exp_id, mch_ip, port)
			if recv_msg == 'exp_queuing':
				return_msg = "Successful"
			else:
				return_msg = recv_msg
		else:
			return_msg = "ERROR: The specified experiement does not exist!"
	else:
		return_msg = "ERROR: The specified IP does not exist!"
	return HttpResponse(return_msg)


def deploy_run(exp_full_dir, exp_id, ip, port):
	# Create a socket to communicate with the workers
	s = socket.socket();
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	interface_server_addr = get_info_from_config_file('interface_server_addr')
	this_server_ip, this_server_port = interface_server_addr.split(':');
	while True:
		try:
			s.bind((this_server_ip, int(this_server_port)))
			break;
		except Exception as e:
			logger.error('Socker binding for deploying the run is unsuccessful with the error: ' 
						+ traceback.format_exc() + ', will retry after 2 seconds.')
			time.sleep(2);
	s.connect((ip, port));
	s.sendall(b'deployrun');
	recv_str = s.recv(1024).decode(encoding = 'utf-8');
	# Send files to the worker
	if recv_str == "ready_to_receive":
		# Send the exp id
		s.sendall(bytearray(exp_id, encoding = 'utf-8'))
		# Send seperator
		s.sendall(b'$%^next^%$')
		# Send run.sh and run.meta in order
		files_to_send = ['run.sh', 'run.meta']
		file_sent_count = 0;
		for file_name in files_to_send:
			file_full_dir = exp_full_dir + '/' + file_name;
			if os.path.isfile(file_full_dir):
				f = open(file_full_dir, 'rb');
				f_line = f.readline(1024);
				while len(f_line)>0:
					print (f_line);
					s.sendall(f_line);
					f_line = f.readline(1024);
			else:
				pass;
			file_sent_count += 1;
			if file_sent_count < len(files_to_send):
				s.sendall(b'$%^next^%$');
		s.sendall(b'$%^endtransfer^%$');
		recv_str = s.recv(1024).decode(encoding = 'utf-8');
	s.close();
	return recv_str;

@login_required
def get_exp_status(request):
	run_name = request.GET.get('run_name');
	response = {};
	# List all available runs
	run_dirs_names, run_dirs = getRuns();
	run_dir_this = run_dirs[run_dirs_names.index(run_name)]
	run_this_exp_dirs = os.listdir(run_dir_this);
	# Loop all exps in the run
	for run_this_exp_dir in run_this_exp_dirs:
		run_this_exp_full_dir = run_dir_this + os.sep + run_this_exp_dir
		if os.path.isdir(run_this_exp_full_dir):
			this_exp_status, this_exp_machine, this_exp_step = check_exp_status(run_this_exp_full_dir);
			response[':'.join([run_name, run_this_exp_dir])] = {'status': this_exp_status,
																'machine': this_exp_machine,
																'step': this_exp_step}
	return JsonResponse(response, json_dumps_params={'indent': 2})

@login_required
def get_all_exp(request):
	run_name = request.GET.get('run_name');
	# List all available runs
	run_dirs_names, run_dirs = getRuns();
	run_dir_this = run_dirs[run_dirs_names.index(run_name)]
	run_this_exp_dirs = os.listdir(run_dir_this);
	pd_list = [];
	# Loop all exps in the run
	for run_this_exp_dir in run_this_exp_dirs:
		run_this_exp_full_dir = run_dir_this + os.sep + run_this_exp_dir
		if os.path.isdir(run_this_exp_full_dir):
			if os.path.isfile(run_this_exp_full_dir + '/run.sh'):
				# If the args.csv exists, run run.sh to get the args
				if not os.path.isfile(run_this_exp_full_dir + '/args.csv'):
					with open(run_this_exp_full_dir + '/run.sh', 'r') as ext_run_f:
						ext_run_f_lines = ext_run_f.read().splitlines()
					ext_run_f_lines[-1] += ' --check_args_only True';
					# Generate a temp run.sh file
					with open(run_this_exp_full_dir + '/~run.sh', 'w') as temp_run_f:
						[temp_run_f.write('%s\n'%line) for line in ext_run_f_lines]
					subprocess.call('bash ' + run_this_exp_full_dir + '/~run.sh', shell=True,
									cwd = run_this_exp_full_dir)
				# Read args.csv
				this_exp_pd = pd.read_csv(run_this_exp_full_dir + '/args.csv');
				this_exp_pd['exp_id'] = pd.Series([int(run_this_exp_dir)]);
				this_exp_status, this_exp_machine, this_exp_step = check_exp_status(run_this_exp_full_dir);
				this_exp_pd.insert(0, 'step', '%s'%(this_exp_step));
				this_exp_pd.insert(0, 'machine', this_exp_machine)
				this_exp_pd.insert(0, 'status', this_exp_status)
				this_exp_pd.insert(0, 'action', None)
				pd_list.append(this_exp_pd);
			else:
				pass;
	all_exp_args_pd = pd.concat(pd_list, sort = False);
	all_exp_args_pd.set_index('exp_id', drop = True, inplace = True);
	all_exp_args_pd.sort_index(axis=0, inplace = True);
	response_html = all_exp_args_pd.to_html(bold_rows = True, table_id = 'runs_args_tb');

	# add ids to the html
	def add_id_to_html(response_html, table_cols):
		dom = parseString(response_html);
		tbody = dom.getElementsByTagName('tbody')[0];
		trs = tbody.getElementsByTagName('tr');
		for tr in trs:
			th = tr.getElementsByTagName('th')[0];
			tr_id = th.firstChild.data;
			tr.setAttribute('id', ':'.join([run_name, tr_id]));
			tds = tr.getElementsByTagName('td');
			for i in range(len(tds)):
				td = tds[i];
				td.setAttribute('id', ':'.join([run_name, tr_id, table_cols[i]]))
		return dom.toxml();

	# add button to the html
	def add_button_to_html(response_html, run_name, ids):
		dom = parseString(response_html);
		for exp_id in ids:
			action_row_id = ':'.join([run_name, str(exp_id), 'action']);
			print (dom.getElementById('rl_parametric_runs_v1:4:action'))
			action_row_obj = dom.getElementById(action_row_id);
			print (action_row_id)
			print (action_row_obj.toxml())
			action_row_btn = dom.createElement('button');
			action_row_btn.innerText = 'run';
			action_row_obj.appenChild(action_row_btn);
		return dom.toxml();

	table_col_names = all_exp_args_pd.columns.values;
	response_html = add_id_to_html(response_html, table_col_names);
	return HttpResponse(response_html)


def check_exp_status(exp_full_dir):
	meta_dir = exp_full_dir + '/run.meta';
	meta_run_step = None;
	meta_run_status = None;
	meta_run_machine = None;

	if os.path.isfile(meta_dir):
		with open(meta_dir, 'r') as meta_dir_f:
			meta_dir_f_json = json.load(meta_dir_f)
		meta_run_status = meta_dir_f_json['status']
		meta_run_machine = meta_dir_f_json['machine'];
		meta_run_step = meta_dir_f_json['step'];

	return (meta_run_status, meta_run_machine, meta_run_step);

def get_ip_idx_in_list(ip_to_test):
	index = 0;
	to_return = -1;
	available_worker_clients = get_info_from_config_file('available_worker_clients')
	for ip_port in available_worker_clients:
		ip, port = ip_port.split(':');
		if ip == ip_to_test:
			to_return = index;
			break;
	return to_return;













