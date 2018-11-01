import socket
import subprocess
import copy
import time
import os
import psutil
import queue
import threading
import xml.etree.ElementTree as ET

from util.logger import Logger
from SDCServer.bacnet.bacenum import bacenumMap, bacenumMap_inv

FD = os.path.dirname(os.path.realpath(__file__));
LOG_LEVEL = 'DEBUG';
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";     

class WorkerClient(object):

	def __init__(self, port, max_work_num):
		self._is_exp_worker_manager_run = True;
		self._exp_queue = queue.Queue();
		self._current_working_processes = {};
		self._main_thread_manager = None;
		self._logger_main = Logger().getLogger('worker_client_logger', LOG_LEVEL, LOG_FMT, 
			log_file_path = '%s/log/%s_%s.log'%(FD, socket.gethostname(), time.time()));
		self._run_exp_worker_manager(max_work_num)
		self._port = port;

	def runWorkerClient(self):
		# Set the logger
		
		# Create the socket
		s = socket.socket();
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.bind(('0.0.0.0', self._port))        
		s.listen(5)
		self._logger_main.info('Socket starts at ' + (':'.join(str(e) for e in s.getsockname())))
		working_list = [];                 
		while True:
			self._logger_main.info("Listening...")
			c, addr = s.accept()
			addr = (':'.join(str(e) for e in addr));   
			self._logger_main.info('Got connection from ' + addr);
			recv = c.recv(1024).decode(encoding = 'utf-8')
			to_send = None;
			if recv.lower() == 'getstatus': 
				self._logger_main.info('Received GETSTATUS request from ' + addr)
				# Get status return [cpu, memory, running_tasks, waiting_tasks, running_task_steps]
				# Get cpu and memory usage
				cpu = psutil.cpu_percent()
				memory = psutil.virtual_memory().percent
				running_tasks = [subp_name for subp_name in self._current_working_processes];
				waitng_tasks = [exp[0] for exp in list(self._exp_queue.queue)]
				running_tasks_steps = self._get_running_tasks_steps(running_tasks)
				to_send = ['getstatus', cpu, memory, running_tasks, waitng_tasks, running_tasks_steps];
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
				# Separated by "$%^next^%$", and ended by "$%^endtransfer^%$"
				item_counter = 0;
				recv_counter = 0;
				while True:
					recv = c.recv(1024);
					recv_decode = recv.decode(encoding = 'utf-8');
					if recv_counter == 0:
						this_exp_run_id = recv_decode;
						this_exp_run_name, this_exp_run_num = recv_decode.split(':');
						transfer_file_dir_base = FD + '/../' + this_exp_run_name + '/' \
											   + this_exp_run_num;
						# Create the exp base dir if not exist
						if not os.path.isdir(transfer_file_dir_base):
							os.makedirs(transfer_file_dir_base);
					else:
						# Receive files
						if recv_decode.lower() == '$%^next^%$':
							item_counter += 1;
							# Set file name
							if item_counter == 1:
								file_name = 'run.sh';
							elif item_counter == 2:
								io_f.close() # Close the last file io object
								self._logger_main.info('Finish receiving %s'%file_name);
								file_name = 'run.meta';
							io_f = open(transfer_file_dir_base + '/' + file_name, 'wb');
							self._logger_main.info('Start receiving %s'%file_name);
						elif recv_decode.lower() != '$%^endtransfer^%$':
							# Write byte to file
							io_f.write(recv);
						elif recv_decode.lower() == '$%^endtransfer^%$':
							# End
							io_f.close();
							self._logger_main.info('Finish receiving %s'%file_name);
							break;
					recv_counter += 1;
				# Push the exp to the queue
				self._exp_queue.put(this_exp_run_id)
			elif recv.lower() == 'getexpstate':
				self._logger_main.info('Received GETEXPSTATE request from ' + addr)
				# Send run.meta and eval_res_hist.csv
				c.sendall(bytearray('ready_to_send', encoding = 'utf-8'));
				recv = c.recv(1024);
				target_exp_run_id = recv.decode(encoding = 'utf-8');
				self._logger_main.info('Received the request for run_id %s'%target_exp_run_id);
				target_run_name. target_run_num = target_exp_run_id.split(':');
				base_dir = FD + '/../' + target_run_name + '/' + target_run_num;
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
						c.sendall('$%^file_not_exist^%$');
					file_sent_count += 1;
					if file_sent_count < len(files_to_send):
						c.sendall('$%^next^%$');
				c.sendall('$%^endtransfer^%$');
			
	def _run_exp_worker_manager(self, max_work_num):
		
		def thread_worker(max_work_num):
			self._logger_main.info('Main thread worker manager starts.')
			while self._is_exp_worker_manager_run:
				time.sleep(1)
				# Remove finsihed threads from the list
				if len(self._current_working_processes) > 0:
					for process_name in self._current_working_processes:
						process = self._current_working_processes[process_name];
						if not process.poll() == None:
							del self._current_working_processes[process_name];
							self._logger_main.info('Process %s finished.'%process_name);
							# Modify the meta file
							run_name, exp_id = process_name.split(':');
							meta_file_path = FD + '/../' + run_name + '/' + exp_id + '/run.meta';
							with open(meta_file_path, 'r+') as meta_file:
								meta_file_json = json.load(meta_file);
								meta_file_json['status'] = 'complete';
								json.dump(meta_file_json, meta_file);
				# Add new worker tasks to run
				if len(self._current_working_processes) < max_work_num:
					if self._exp_queue.qsize() > 0:
						exp_id = self._exp_queue.get();
						exp_run_name, exp_run_num = exp_id.split(':');
						cwd = FD + '/../' + exp_run_name + '/' + exp_run_num;
						out_log_file = open(cwd + '/out.log', 'w+');
						process = subprocess.Popen('bash run.sh', shell = True, 
							preexec_fn = os.setsid, stdout = out_log_file, 
							stderr = out_log_file, cwd = cwd);
						self._logger_main.info('A new worker subprocess %s starts.' %exp_id)
						self._current_working_processes[exp_id] = process;
			self._logger_main.info('Main thread worker manager ends.')

		self._main_thread_manager = threading.Thread(target = thread_worker, args = (max_work_num, ));
		self._main_thread_manager.start();


	def _run_exp_worker(self):
		return;

	def _get_running_tasks_steps(self, running_task_ids):
		running_tasks_steps = [None, None]
		for running_task_id in running_task_ids:
			run_name, exp_id = running_task_id.split(':');
			meta_file_path = FD + '/../' + run_name + '/' + exp_id + '/run.meta';
			if os.path.isfile(meta_file_path):
				with open(meta_file_path, 'r') as meta_file:
					meta_file_json = json.load(meta_file);
					current_step = meta_file_json['step'];
		return running_tasks_steps;





	







