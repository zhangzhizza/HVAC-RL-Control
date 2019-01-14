import socket
import subprocess
import copy
import time
import os
import psutil
import queue
import threading, json
import xml.etree.ElementTree as ET

from util.logger import Logger

FD = os.path.dirname(os.path.realpath(__file__));
LOG_LEVEL = 'DEBUG';
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";
CONFIG_FILE_PATH = FD + '/../../HVAC_RL_web_interface/configurations/configurations.json';
TRUSTED_ADDR = json.load(open(CONFIG_FILE_PATH, 'r'))['TRUSTED_ADDR']
RUNS_PATH = FD + '/../runs/'
worker_server_addr = json.load(open(CONFIG_FILE_PATH, 'r'))['worker_server_addr']
class WorkerClient(object):

	def __init__(self, max_work_num, ip, port = 16785):
		self._is_exp_worker_manager_run = True;
		self._exp_queue = queue.Queue();
		self._current_working_processes = {};
		self._main_thread_manager = None;
		self._logger_main = Logger().getLogger('worker_client_logger', LOG_LEVEL, LOG_FMT, 
			log_file_path = '%s/log/%s_%s_client.log'%(FD, socket.gethostname(), time.time()));
		self._run_exp_worker_manager(max_work_num)
		self._port = port;
		self._ip = ip;

	def runWorkerClient(self):
		"""
		SSL example
		https://carlo-hamalainen.net/2013/01/24/python-ssl-socket-echo-test-with-self-signed-certificate/
		"""
		# Set the logger
		
		# Create the socket
		s = socket.socket();
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.bind((self._ip, self._port))        
		s.listen(5)
		self._logger_main.info('Socket starts at ' + (':'.join(str(e) for e in s.getsockname())))
		working_list = [];                 
		while True:
			self._logger_main.info("Listening...")
			c, addr = s.accept()
			addr = (':'.join(str(e) for e in addr));   
			self._logger_main.info('Got connection from ' + addr);
			if addr not in TRUSTED_ADDR:
				self._logger_main.warning('Got untrusted connection, server exits.');
				break;
			recv = c.recv(1024).decode(encoding = 'utf-8')
			to_send = None;
			if recv.lower() == 'getstatus': 
				self._logger_main.info('Received GETSTATUS request from ' + addr)
				# Get status return [cpu, memory, running_tasks, waiting_tasks, running_task_steps]
				# Get cpu and memory usage
				cpu = psutil.cpu_percent()
				memory = psutil.virtual_memory().percent
				st = os.statvfs('.')
				disk = round(st.f_bavail * st.f_frsize/1000/1000/1000, 1) # in GB
				running_tasks = [subp_name for subp_name in list(self._current_working_processes)];
				waitng_tasks = [exp[0] for exp in list(self._exp_queue.queue)]
				to_send = {};
				to_send['cpu'] = str(cpu);
				to_send['mem'] = str(memory);
				to_send['dsk'] = str(disk);
				exp_dict = {}
				# All processes
				all_processes = []
				all_processes.extend(running_tasks);
				all_processes.extend(waitng_tasks);
				all_processes = running_tasks + waitng_tasks;
				for process_i in all_processes:
					process_i_status, process_i_step = self._get_exp_status(process_i)
					exp_dict[process_i] = [process_i_status, process_i_step];
				to_send['exps'] = exp_dict;
				to_send['running_queuing'] = '%s/%s'%(len(running_tasks), len(waitng_tasks));
				to_send = json.dumps(to_send);
				c.sendall(bytearray(str(to_send), encoding = 'utf-8'));
				self._logger_main.info('Messages sent to %s: %s'%(addr, to_send))
			elif recv.lower() == 'deployrun':
				self._logger_main.info('Received DEPLOYRUN request from ' + addr)
				this_exp_run_id = None;
				this_exp_run_name = None;
				this_exp_run_num = None;
				transfer_file_dir_base = None;
				file_name = None;
				io_f = None;
				c.sendall(bytearray('ready_to_receive', encoding = 'utf-8'));
				# Send order: exp_id, run.sh file, run.meta file
				# Separated by "$%^next^%$"
				recv_byte = b'';
				while True:
					recv = c.recv(1024);
					recv_byte += recv;
					recv_decode_this = recv.decode(encoding = 'utf-8');
					if '$%^endtransfer^%$' in recv_decode_this:
						break;
				recv_decode = recv_byte.decode(encoding = 'utf-8');
				recv_decode_list = recv_decode.split('$%^next^%$');
				# Remove the ending strings
				recv_decode_list[-1] = recv_decode_list[-1].split('$%^endtransfer^%$')[0]
				this_exp_run_id = recv_decode_list[0];
				self._logger_main.info('Request for exp_id %s'%this_exp_run_id);
				this_exp_run_name, this_exp_run_num = this_exp_run_id.split(':');
				transfer_file_dir_base = RUNS_PATH + this_exp_run_name + '/' \
											   + this_exp_run_num;
				# Create the exp base dir if not exist
				if not os.path.isdir(transfer_file_dir_base):
					os.makedirs(transfer_file_dir_base);
				file_names_to_write = ['run.sh', 'run.meta'];
				file_counter = 1;
				for file_name in file_names_to_write:
					with open(transfer_file_dir_base + '/' + file_name, 'wb') as io_f:
						self._logger_main.info('Writing to %s...'%file_name);
						io_f.write(bytearray(recv_decode_list[file_counter], encoding = 'utf-8'));
					self._logger_main.info('Writing to %s finished.'%file_name);
					file_counter += 1;
				# Push the exp to the queue
				self._exp_queue.put(this_exp_run_id)
				# Set the meta status
				self._set_meta_status(transfer_file_dir_base + '/run.meta', 'queuing');
				c.sendall(b'exp_queuing');
			elif recv.lower() == 'getexpstate':
				self._logger_main.info('Received GETEXPSTATE request from ' + addr)
				# Send run.meta and eval_res_hist.csv
				c.sendall(bytearray('ready_to_send', encoding = 'utf-8'));
				recv = c.recv(1024);
				target_exp_run_id = recv.decode(encoding = 'utf-8');
				self._logger_main.info('Received the request for run_id %s'%target_exp_run_id);
				target_run_name. target_run_num = target_exp_run_id.split(':');
				base_dir = RUNS_PATH + target_run_name + '/' + target_run_num;
				files_to_send = ['run.meta', 'eval_res_hist.csv']
				file_sent_count = 0;
				for file_name in files_to_send:
					file_full_dir = base_dir + '/' + file_name;
					self._logger_main.info('Start sending %s'%file_full_dir)
					if os.path.isfile(file_full_dir):
						f = open(file_full_dir, 'rb');
						f_line = f.readline(1024);
						while len(f_line)>0:
							c.sendall(f_line);
							f_line = f.readline(1024);
						self._logger_main.info('Finish sending %s'%file_full_dir)
					else:
						self._logger_main.warning('%s does not exist!'%file_full_dir)
						c.sendall(b'$%^file_not_exist^%$');
					file_sent_count += 1;
					if file_sent_count < len(files_to_send):
						c.sendall(b'$%^next^%$');
				c.sendall(b'$%^endtransfer^%$');
			
	def _run_exp_worker_manager(self, max_work_num):
		
		def thread_worker(max_work_num):
			self._logger_main.info('Main thread worker manager starts.')
			while self._is_exp_worker_manager_run:
				time.sleep(1)
				# Remove finsihed threads from the list
				if len(self._current_working_processes) > 0:
					for process_name in list(self._current_working_processes):
						process = self._current_working_processes[process_name];
						if not process.poll() == None:
							del self._current_working_processes[process_name];
							self._logger_main.info('Process %s finished.'%process_name);
							# Modify the meta file
							run_name, exp_id = process_name.split(':');
							meta_file_path = RUNS_PATH + run_name + '/' + exp_id + '/run.meta';
							self._set_meta_status(meta_file_path, 'complete')
							# Send the results files to the server
							self._logger_main.info('Sending the %s results files to the server.'%(process_name));
							recv_msg = self._send_results_to_server(run_name, exp_id);
							self._logger_main.info('Files sending status: %s.'%(recv_msg));
				# Add new worker tasks to run
				if len(self._current_working_processes) < max_work_num:
					if self._exp_queue.qsize() > 0:
						exp_id = self._exp_queue.get();
						exp_run_name, exp_run_num = exp_id.split(':');
						cwd = RUNS_PATH + exp_run_name + '/' + exp_run_num;
						out_log_file = open(cwd + '/out.log', 'w+');
						process = subprocess.Popen('bash run.sh', shell = True, 
							preexec_fn = os.setsid, stdout = out_log_file, 
							stderr = out_log_file, cwd = cwd);
						self._logger_main.info('A new worker subprocess %s starts.' %exp_id)
						# Modify the meta file
						meta_file_dir = cwd + '/run.meta'
						self._set_meta_status(meta_file_dir, 'running')
						self._current_working_processes[exp_id] = process;
			self._logger_main.info('Main thread worker manager ends.')

		self._main_thread_manager = threading.Thread(target = thread_worker, args = (max_work_num, ));
		self._main_thread_manager.start();


	def _run_exp_worker(self):
		return;

	def _get_exp_status(self, task_id):
		run_name, exp_id = task_id.split(':');
		meta_file_path = RUNS_PATH + run_name + '/' + exp_id + '/run.meta';
		if os.path.isfile(meta_file_path):
			with open(meta_file_path, 'r') as meta_file:
				meta_file_json = json.load(meta_file);
				current_status = meta_file_json['status'];
				current_step = meta_file_json['step'];
		return [current_status, current_step];

	def _set_meta_status(self, meta_file_dir, status_str):
		if os.path.isfile(meta_file_dir):
			with open(meta_file_dir, 'r+') as meta_file:
				meta_file_json = json.load(meta_file);
				meta_file.seek(0);
				meta_file_json['status'] = status_str;
				json.dump(meta_file_json, meta_file);
				meta_file.truncate()

	def _send_results_to_server(self, run_name, run_num):
		s = socket.socket();
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		while True:
			try:
				s.bind((self._ip, self._port + 1));
				break;
			except Exception as e:
				logger.error('Socker binding for sending results is unsuccessful with the error: ' 
							+ traceback.format_exc() + ', will retry after 2 seconds.')
				time.sleep(2);
		server_ip, server_port = worker_server_addr.split(':');
		s.connect((server_ip, int(server_port)));
		s.sendall(b'recvevallog');
		recv_str = s.recv(1024).decode(encoding = 'utf-8');
		# Send files to the server
		if recv_str == "ready_to_receive":
			# Send the exp id
			s.sendall(bytearray(':'.join([run_name, run_num]), encoding = 'utf-8'))
			# Send seperator
			s.sendall(b'$%^next^%$')
			# Send eval_res_hist.csv in order
			files_to_send = ['eval_res_hist.csv']
			file_sent_count = 0;
			exp_full_dir = RUNS_PATH + run_name + '/' + run_num;
			for file_name in files_to_send:
				file_full_dir = exp_full_dir + '/' + file_name;
				if os.path.isfile(file_full_dir):
					f = open(file_full_dir, 'rb');
					f_line = f.readline(1024);
					while len(f_line)>0:
						s.sendall(f_line);
						f_line = f.readline(1024);
				else:
					pass;
				file_sent_count += 1;
				if file_sent_count < len(files_to_send):
					s.sendall(b'$%^next^%$');
			s.sendall(b'$%^endtransfer^%$');
		recv_str = s.recv(1024).decode(encoding = 'utf-8');
		return recv_str;



	







