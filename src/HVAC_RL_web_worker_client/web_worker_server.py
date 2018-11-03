import socket
import time
import os
import threading, json, traceback

from util.logger import Logger

FD = os.path.dirname(os.path.realpath(__file__));
LOG_LEVEL = 'DEBUG';
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";     
available_computers = ["0.0.0.0:7777"]

class WorkerServer(object):

	def __init__(self, state_syn_interval):
		self._logger_main = Logger().getLogger('worker_client_logger', LOG_LEVEL, LOG_FMT, 
			log_file_path = '%s/log/%s_%s_server.log'%(FD, socket.gethostname(), time.time()));
		self._is_run_file_syncher = True;
		self._threads = [];
		self._state_syn_interval = state_syn_interval;

	def runWorkerServer(self):
		state_file_syncher_thread = threading.Thread(target = self._state_file_syncher
													, args = (self._state_syn_interval, ));
		state_file_syncher_thread.start();
		self._logger_main.info('state_file_syncher started.')


	def _state_file_syncher(self, interval):
		while self._is_run_file_syncher:
			for worker_ip_port in available_computers:
				try:
					ip_this, port_this = worker_ip_port.split(":");
					s = socket.socket();
					s.connect((ip_this, int(port_this)));
					self._logger_main.info('STATE_SYNCHER: Connected to %s.'%(worker_ip_port))
					s.sendall(b'getstatus');
					recv_str = s.recv(4096).decode(encoding = 'utf-8');
					recv_json = json.loads(recv_str)
					exps_this_worker = recv_json['exps'];
					for exp_this_worker_name in list(exps_this_worker):
						exp_this_worker_status, exp_this_worker_step = exps_this_worker[exp_this_worker_name];
						exp_this_run_name, exp_this_run_num = exp_this_worker_name.split(':');
						exp_this_meta_dir = FD + '/../' + exp_this_run_name + '/' + exp_this_run_num;
						if not os.path.isdir(exp_this_meta_dir):
							os.makedirs(exp_this_meta_dir);
						self._set_meta_status(exp_this_meta_dir + '/run.meta', ip_this
											, exp_this_worker_status, exp_this_worker_step);
					self._logger_main.info('STATE_SYNCHER: Finished updating for exps %s.'%(list(exps_this_worker)));
				except Exception as e:
					self._logger_main.info('STATE_SYNCHER: ERROR: %s'%(traceback.format_exc()))
			time.sleep(interval)

	
	def _set_meta_status(self, meta_file_dir, ip, status_str, step_str):
		if os.path.isfile(meta_file_dir):
			with open(meta_file_dir, 'r+') as meta_file:
				meta_file_json = json.load(meta_file);
				meta_file.seek(0);
				meta_file_json['status'] = status_str;
				meta_file_json['step'] = step_str;
				json.dump(meta_file_json, meta_file);
				meta_file.truncate()
		else:
			with open(meta_file_dir, 'w') as meta_file:
				meta_file_json = {}
				meta_file_json['status'] = status_str;
				meta_file_json['step'] = step_str;
				meta_file_json['machine'] = ip;
				json.dump(meta_file_json, meta_file);




	







