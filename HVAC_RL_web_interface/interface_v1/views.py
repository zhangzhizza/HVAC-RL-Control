from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from xml.dom.minidom import parseString

import requests
import pandas as pd
import os, shutil, subprocess, json, socket, ast

this_dir_path = os.path.dirname(os.path.realpath(__file__))
available_computers = ["0.0.0.0:7777"]

# Create your views here.
def index(request):
	run_dirs_names, run_dirs = getRuns();
	return render(request, 'interface_v1/html/srtdash/index.html',\
    	{'run_dirs_names': run_dirs_names,
    	 'available_computers': available_computers})


def getRuns():
	scan_dir = this_dir_path + '/../../src/';
	dirs = os.listdir(scan_dir);
	run_dirs_names = [];
	run_dirs = []
	for this_dir in dirs:
		if 'run' in this_dir and os.path.isdir(scan_dir + this_dir):
			run_dirs_names.append(this_dir);
			run_dirs.append(scan_dir + this_dir)
	return run_dirs_names, run_dirs;

def get_worker_status(request):
	"""
	Args:
		arguments: str
			In the pattern "ip=0.0.0.0&port=9999"
	"""
	ip = request.GET.get('ip')
	port = int(request.GET.get('port'))
	# Create a socket
	s = socket.socket();
	s.connect((ip, port));
	s.sendall(b'getstatus');
	recv_str = s.recv(1024).decode(encoding = 'utf-8');
	recv_list = ast.literal_eval(recv_str)
	recv_json = {};
	recv_json['cpu'] = recv_list[1]
	recv_json['mem'] = recv_list[2]
	recv_json['running'] = recv_list[3]
	recv_json['queue'] = recv_list[4]
	recv_json['steps'] = recv_list[5]

	return JsonResponse(recv_json, json_dumps_params={'indent': 2})

def run_exp(request):
	exp_id = request.GET.get('id')
	mch_ip = request.GET.get('ip')
	exp_run_name, exp_run_num = exp_id.split(":");
	exp_full_dir = this_dir_path + '/../../src/' + exp_run_name + '/' + exp_run_num;
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
			port = int(available_computers[ip_idx_in_list].split(":")[1])
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
			print (file_name)
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
		print('endtranser')
		s.sendall(b'$%^endtransfer^%$');
		print('final msg sent')
		recv_str = s.recv(1024).decode(encoding = 'utf-8');
	return recv_str;



def get_exp_status(request, run_name):
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

def get_all_exp(request, run_name):
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
	for ip_port in available_computers:
		ip, port = ip_port.split(':');
		if ip == ip_to_test:
			to_return = index;
			break;
	return to_return;













