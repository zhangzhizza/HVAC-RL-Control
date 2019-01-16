import socket
import time
import os, shutil
import threading, json, traceback

from util.logger import Logger

FD = os.path.dirname(os.path.realpath(__file__));
LOG_LEVEL = 'DEBUG';
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";
CONFIG_FILE_PATH = FD + '/../../HVAC_RL_web_interface/configurations/configurations.json';
RUNS_PATH = FD + '/../runs/'
TRUSTED_ADDR = json.load(open(CONFIG_FILE_PATH, 'r'))['TRUSTED_ADDR']
available_worker_clients = json.load(open(CONFIG_FILE_PATH, 'r'))['available_worker_clients']

class WorkerServer(object):

	def __init__(self, state_syn_interval, ip, port = 14786):
		self._logger_main = Logger().getLogger('worker_server_logger', LOG_LEVEL, LOG_FMT, 
			log_file_path = '%s/log/%s_%s_server.log'%(FD, socket.gethostname(), time.time()));
		self._is_run_file_syncher = True;
		self._threads = [];
		self._state_syn_interval = state_syn_interval;
		self._port = port;
		self._ip = ip;

	def runWorkerServer(self):
		state_file_syncher_thread = threading.Thread(target = self._state_file_syncher
													, args = (self._state_syn_interval, self._port + 1));
		self._threads.append(state_file_syncher_thread);
		state_file_syncher_thread.start();
		self._logger_main.info('state_file_syncher started.')
		eval_log_file_recver_thread = threading.Thread(target = self._eval_log_file_receiver
													, args = (self._port, ));
		self._threads.append(eval_log_file_recver_thread);
		eval_log_file_recver_thread.start();
		self._logger_main.info('eval_log_file_receiver started.')
		run_exp_reseter_thread = threading.Thread(target = self._run_exp_reseter
													, args = (self._port + 2, ));
		self._threads.append(run_exp_reseter_thread);
		run_exp_reseter_thread.start();
		self._logger_main.info('run_exp_reseter_thread started.')

	def _eval_log_file_receiver(self, port):
		# Create the socket
		s = socket.socket();
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.bind((self._ip, self._port))        
		s.listen(5)
		self._logger_main.info('EVALLOG_RECVER: Socket starts at ' + (':'.join(str(e) for e in s.getsockname())));
		while True:
			self._logger_main.info("EVALLOG_RECVER: Listening...")
			c, addr = s.accept()
			addr = (':'.join(str(e) for e in addr));   
			self._logger_main.info('EVALLOG_RECVER: Got connection from ' + addr);
			if addr not in TRUSTED_ADDR:
				self._logger_main.warning('Got untrusted connection, server exits.');
				break;
			recv = c.recv(1024).decode(encoding = 'utf-8')
			if recv.lower() == 'recvevallog':
				self._logger_main.info('EVALLOG_RECVER: Received RECVEVALLOG request from ' + addr)
				this_exp_run_id = None;
				this_exp_run_name = None;
				this_exp_run_num = None;
				transfer_file_dir_base = None;
				file_name = None;
				io_f = None;
				c.sendall(bytearray('ready_to_receive', encoding = 'utf-8'));
				# Send order: exp_id, eval_res_hist.csv file
				# Separated by "$%^next^%$", ended by '$%^endtransfer^%$'
				recv_byte = b'';
				while True:
					recv = c.recv(1024);
					recv_byte += recv;
					recv_decode_this = recv.decode(encoding = 'utf-8');
					print(recv_decode_this)
					if '$%^endtransfer^%$' in recv_decode_this:
						break;
				recv_decode = recv_byte.decode(encoding = 'utf-8');
				recv_decode_list = recv_decode.split('$%^next^%$');
				# Remove the ending strings
				recv_decode_list[-1] = recv_decode_list[-1].split('$%^endtransfer^%$')[0]
				this_exp_run_id = recv_decode_list[0];
				self._logger_main.info('EVALLOG_RECVER: Request for exp_id %s'%this_exp_run_id);
				this_exp_run_name, this_exp_run_num = this_exp_run_id.split(':');
				transfer_file_dir_base = RUNS_PATH + this_exp_run_name + '/' \
											   + this_exp_run_num;
				# Create the exp base dir if not exist
				if not os.path.isdir(transfer_file_dir_base):
					os.makedirs(transfer_file_dir_base);
				file_names_to_write = ['eval_res_hist.csv', 'run.meta'];
				file_counter = 1;
				for file_name in file_names_to_write:
					with open(transfer_file_dir_base + '/' + file_name, 'wb') as io_f:
						self._logger_main.info('EVALLOG_RECVER: Writing to %s...'%file_name);
						io_f.write(bytearray(recv_decode_list[file_counter], encoding = 'utf-8'));
					self._logger_main.info('EVALLOG_RECVER: Writing to %s finished.'%file_name);
					file_counter += 1;
				c.sendall(b'received'); 

	def _state_file_syncher(self, interval, port):
		while self._is_run_file_syncher:
			for worker_ip_port in available_worker_clients:
				try:
					ip_this, port_this = worker_ip_port.split(":");
					s = socket.socket();
					s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
					s.bind((self._ip, port))  
					s.connect((ip_this, int(port_this)));
					self._logger_main.info('STATE_SYNCHER: Connected to %s.'%(worker_ip_port))
					s.sendall(b'getstatus');
					recv_str = s.recv(4096).decode(encoding = 'utf-8');
					recv_json = json.loads(recv_str)
					exps_this_worker = recv_json['exps'];
					for exp_this_worker_name in list(exps_this_worker):
						exp_this_worker_status, exp_this_worker_step = exps_this_worker[exp_this_worker_name];
						exp_this_run_name, exp_this_run_num = exp_this_worker_name.split(':');
						exp_this_meta_dir = RUNS_PATH + exp_this_run_name + '/' + exp_this_run_num;
						if not os.path.isdir(exp_this_meta_dir):
							os.makedirs(exp_this_meta_dir);
						self._set_meta_status(exp_this_meta_dir + '/run.meta', ip_this
											, exp_this_worker_status, exp_this_worker_step);
					self._logger_main.info('STATE_SYNCHER: Finished updating for exps %s.'%(list(exps_this_worker)));
					s.close()
				except Exception as e:
					self._logger_main.error('STATE_SYNCHER: when connecting %s, %s'
											%(worker_ip_port, traceback.format_exc()))
			time.sleep(interval)

	def _run_exp_reseter(self, port):
		# Create the socket
		s = socket.socket();
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.bind((self._ip, port))        
		s.listen(5)
		self._logger_main.info('EXP_RESETER: Socket starts at ' + (':'.join(str(e) for e in s.getsockname())));
		while True:
			self._logger_main.info("EXP_RESETER: Listening...")
			c, addr = s.accept()
			addr = (':'.join(str(e) for e in addr));   
			self._logger_main.info('EXP_RESETER: Got connection from ' + addr);
			if addr not in TRUSTED_ADDR:
				self._logger_main.warning('EXP_RESETER: Got untrusted connection, server exits.');
				break;
			recv = c.recv(1024).decode(encoding = 'utf-8')
			code, prj_name, run_id = recv.lower().split(':');
			if code == 'resetexp':
				self._logger_main.info('EXP_RESETER: Received RESETEXP request from ' + addr);
				tgt_run_dir = RUNS_PATH + '/' + prj_name + '/' + run_id;
				# Get the remote worker addr
				try:
					with open(tgt_run_dir + '/run.meta') as run_meta_f:
						run_meta = json.load(run_meta_f)
						rmt_worker_ip = run_meta['machine'];
				except Exception as e:
					self._logger_main.error('EXP_RESETER: ERROR: %s'%(traceback.format_exc()));
					rmt_worker_ip = None;
				# Clear files in the server
				files_to_keep = ['run.sh'];
				for tgt_run_file in os.listdir(tgt_run_dir):
					if tgt_run_file not in files_to_keep:
						if os.path.isfile(tgt_run_dir + '/' + tgt_run_file):
							os.remove(tgt_run_dir + '/' + tgt_run_file);
						else:
							shutil.rmtree(tgt_run_dir + '/' + tgt_run_file);
				self._logger_main.info('EXP_RESETER: Cleared the run directory for %s:%s in the main server'
										%(prj_name, run_id));
				# Clear files in the remote worker
				if rmt_worker_ip != None:
					rmt_worker_addr = self._get_client_addr(rmt_worker_ip);
					rmt_worker_ip, rmt_worker_port = rmt_worker_addr.split(":");
					s_st = socket.socket();
					s_st.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
					s_st.bind((self._ip, port + 1))  
					s_st.connect((rmt_worker_ip, int(rmt_worker_port)));
					self._logger_main.info('EXP_RESETER: Connected to %s. to clear the run directory'
											%(rmt_worker_addr))
					s_st.sendall(b'%s:%s:%s'%(code, prj_name, run_id));
					recv_str = s_st.recv(4096).decode(encoding = 'utf-8');
					self._logger_main.info('EXP_RESETER: Remote experiment directory cleared');
				else:
					self._logger_main.info('EXP_RESETER: The remote worker of the experiment is not defined');
				c.sendall(b'reset_successful');

	def _get_client_addr(self, ip):
		for addr in available_worker_clients:
			if ip in addr:
				return addr;
		return None;

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




	







