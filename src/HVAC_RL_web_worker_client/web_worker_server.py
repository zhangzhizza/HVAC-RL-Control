import socket
import time
import os
import threading, json, traceback

from util.logger import Logger

FD = os.path.dirname(os.path.realpath(__file__));
LOG_LEVEL = 'DEBUG';
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";
TRUSTED_ADDR = ['0.0.0.0:7777', '0.0.0.0:6666']
available_computers = ["0.0.0.0:7777"]

class WorkerServer(object):

	def __init__(self, state_syn_interval, port):
		self._logger_main = Logger().getLogger('worker_client_logger', LOG_LEVEL, LOG_FMT, 
			log_file_path = '%s/log/%s_%s_server.log'%(FD, socket.gethostname(), time.time()));
		self._is_run_file_syncher = True;
		self._threads = [];
		self._state_syn_interval = state_syn_interval;
		self._port = port;

	def runWorkerServer(self):
		state_file_syncher_thread = threading.Thread(target = self._state_file_syncher
													, args = (self._state_syn_interval, ));
		self._threads.append(state_file_syncher_thread);
		state_file_syncher_thread.start();
		self._logger_main.info('state_file_syncher started.')
		eval_log_file_recver_thread = threading.Thread(target = self._eval_log_file_receiver
													, args = (self._port, ));
		self._threads.append(eval_log_file_recver_thread);
		eval_log_file_recver_thread.start();
		self._logger_main.info('eval_log_file_receiver started.')

	def _eval_log_file_receiver(self, port):
		# Create the socket
		s = socket.socket();
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.bind(('0.0.0.0', self._port))        
		s.listen(5)
		self._logger_main.info('EVALLOG_RECVER: Socket starts at ' + (':'.join(str(e) for e in s.getsockname())));
		while True:
			self._logger_main.info("Listening...")
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
				transfer_file_dir_base = FD + '/../' + this_exp_run_name + '/' \
											   + this_exp_run_num;
				# Create the exp base dir if not exist
				if not os.path.isdir(transfer_file_dir_base):
					os.makedirs(transfer_file_dir_base);
				file_names_to_write = ['eval_res_hist.csv'];
				file_counter = 1;
				for file_name in file_names_to_write:
					with open(transfer_file_dir_base + '/' + file_name, 'wb') as io_f:
						self._logger_main.info('EVALLOG_RECVER: Writing to %s...'%file_name);
						io_f.write(bytearray(recv_decode_list[file_counter], encoding = 'utf-8'));
					self._logger_main.info('EVALLOG_RECVER: Writing to %s finished.'%file_name);
					file_counter += 1;
				c.sendall(b'received'); 

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




	







