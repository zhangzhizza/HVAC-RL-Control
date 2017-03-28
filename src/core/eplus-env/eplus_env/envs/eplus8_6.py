#Wrap the EnergyPLus simulator into the Openai gym env
import socket              
import os
import time
import signal
import _thread
import logging
import subprocess

from shutil import copyfile
from gym import Env, spaces
from gym.envs.registration import register
from xml.etree.ElementTree import Element, SubElement, Comment, tostring

from ..util.logger import Logger  


CWD = os.getcwd();
LOG_LEVEL = 'DEBUG';
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";
LOGGER = Logger();

class EplusEnv(Env):
    """EnergyPlus v8.6 environment

    Args
    ----------
    eplus_path: String
      EnergyPlus executive command path.
    weather_path: String
      EnergyPlus weather file path (.epw). 
    bcvtb_path: String
      BCVTB installation path.
    variable_path: String
      variable.cfg path.
    idf_path: String
      EnergyPlus input description file (.idf).

    Attributes
    ----------
    """
    def __init__(self, eplus_path, 
                 weather_path, bcvtb_path, 
                 variable_path, idf_path):
            
        self.logger_main = LOGGER.getLogger('ROOT', LOG_LEVEL, LOG_FMT);
        
        # Set the environment variable for bcvtb
        os.environ['BCVTB_HOME'] = bcvtb_path;
        
        # Create a socket for communication with the EnergyPlus
        self.logger_main.info('Creating socket for communication...')
        s = socket.socket()
        host = socket.gethostname() # Get local machine name
        s.bind((host, 0))           # Bind to the host and any available port
        sockname = s.getsockname();
        port = sockname[1];         # Get the port number
        s.listen(60)                # Listen on request
        self.logger_main.info('Socket is listening on host %s port %d'%(sockname));
  
        self._env_working_dir_parent = self._get_eplus_working_folder(CWD, '-run');
        self._host = host;
        self._port = port;
        self._socket = s;
        self._eplus_path = eplus_path
        self._weather_path = weather_path
        self._variable_path = variable_path
        self._idf_path = idf_path
        self._episode_existed = False;

        
        
        #######DON NOT FORGET TO CLOSE THE eplus_process.stdout
        
    def _reset(self):
        """Reset the environment.

        This method does the followings:
        1: Make a new EnergyPlus working directory
        2: Copy .idf and variables.cfg file to the working directory
        3: Create the socket.cfg file in the working directory
        4: Create the EnergyPlus subprocess
        5: Establish the connection with EnergyPlus
        6: Read the first sensor data from the EnergyPlus
        
        Return (current_simulation_time, 
                [EnergyPlus results list requested by the variable.cfg])
        """
        # End the last episode if exists
        if self._episode_existed:
            self._end_episode()
        
        # Create EnergyPlus simulaton process
        self.logger_main.info('Creating EnergyPlus simulation environment...')
        eplus_working_dir = self._get_eplus_working_folder(
                                self._env_working_dir_parent, '-sub_run');
        os.makedirs(eplus_working_dir);
                                    # Create the Eplus working directory
        eplus_working_idf_path = (eplus_working_dir + 
                                  '/' + 
                                  self._get_file_name(self._idf_path));
        eplus_working_var_path = (eplus_working_dir + 
                                  '/' + 
                                  self._get_file_name(self._variable_path));
        eplus_working_out_path = (eplus_working_dir + 
                                  '/' + 
                                  'output');
        copyfile(self._idf_path, eplus_working_idf_path);
                                    # Copy the idf file to the working directory
        copyfile(self._variable_path, eplus_working_var_path);
                                    # Copy the variable.cfg file to the working dir
        self._create_socket_cfg(self._host, 
                                self._port,
                                eplus_working_dir); 
                                    # Create the socket.cfg file in the working dir
        self.logger_main.info('EnergyPlus working directory is in %s'
                              %(eplus_working_dir));
        eplus_process = self._create_eplus(self._eplus_path, self._weather_path, 
                                            eplus_working_idf_path,
                                            eplus_working_out_path,
                                            eplus_working_dir);
        self.logger_main.debug('EnergyPlus process is still running ? %r' 
                                %self._get_is_subprocess_running(eplus_process))
        self._eplus_process = eplus_process;
        # Log the Eplus output
        eplus_logger = LOGGER.getLogger('ENERGYPLUS', LOG_LEVEL, LOG_FMT)
        _thread.start_new_thread(self._log_subprocess_info,
                                (eplus_process.stdout,
                                 eplus_logger));
        _thread.start_new_thread(self._log_subprocess_err,
                                (eplus_process.stderr,
                                 eplus_logger));                       
            
        # Establish connection with EnergyPlus
        conn, addr = self._socket.accept()     # Establish connection with client.
        self.logger_main.info('Got connection from %s at port %d.'%(addr));
        # Start the first data exchange
        rcv_1st = conn.recv(1024).decode();
        self.logger_main.debug('Got the first message successfully: ' + rcv_1st);
        version, flag, nDb, nIn, nBl, curSimTim, Dblist \
                                                = self._disassembleMsg(rcv_1st);
        # Remember the message header, useful when send data back to EnergyPlus
        self._eplus_msg_header = [version, flag];
        self._curSimTim = curSimTim;
        
        self._conn = conn;
        self._eplus_working_dir = eplus_working_dir;
        self._episode_existed = True;
        
        return (curSimTim, Dblist)

    def _step(self, action):
        """Execute the specified action.
        
        This method does the followings:
        1: Send a list of float to EnergyPlus
        2: Recieve EnergyPlus results for the next step (state)

        Parameters
        ----------
        action: python list of float
          Control actions that will be passed to the EnergyPlus

        Returns
        (current_simulation_time, 
                [EnergyPlus results list requested by the variable.cfg])
        """
        # Send to the EnergyPlus
        self.logger_main.debug('Perform one step.')
        header = self._eplus_msg_header;
        tosend = self._assembleMsg(header[0], header[1], len(action), 0,
                                   0, self._curSimTim, action);
        self._conn.send(tosend.encode());
        
        # Recieve from the EnergyPlus
        rcv = self._conn.recv(1024).decode();
        self.logger_main.debug('Got message successfully: ' + rcv);
        version, flag, nDb, nIn, nBl, curSimTim, Dblist \
                                        = self._disassembleMsg(rcv);
        self._curSimTim = curSimTim;
        
        return (curSimTim, Dblist);
        

    def _render(self, mode='human', close=False):
        pass;
    
    def _create_eplus(self, eplus_path, weather_path, 
                      idf_path, out_path, eplus_working_dir):
        
        eplus_process = subprocess.Popen('%s -w %s -d %s -r %s'
                        %(eplus_path + '/energyplus', weather_path, 
                          out_path, idf_path),
                        shell = True,
                        cwd = eplus_working_dir,
                        stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE,
                        preexec_fn=os.setsid);
        return eplus_process;
    
    def _get_eplus_working_folder(self, parent_dir, dir_sig = '-run'):
        """Return Eplus output folder. Author: CMU-10703 Spring 2017 TA

        Assumes folders in the parent_dir have suffix -run{run
        number}. Finds the highest run number and sets the output folder
        to that number + 1. 

        Parameters
        ----------
        parent_dir: str
        Parent dir of the Eplus output directory.

        Returns
        -------
        parent_dir/run_dir
        Path to Eplus save directory.
        """
        os.makedirs(parent_dir, exist_ok=True)
        experiment_id = 0
        for folder_name in os.listdir(parent_dir):
            if not os.path.isdir(os.path.join(parent_dir, folder_name)):
                continue
            try:
                folder_name = int(folder_name.split(dir_sig)[-1])
                if folder_name > experiment_id:
                    experiment_id = folder_name
            except:
                pass
        experiment_id += 1

        parent_dir = os.path.join(parent_dir, 'Eplus-env')
        parent_dir = parent_dir + '%s%d'%(dir_sig, experiment_id)
        return parent_dir

    def _create_socket_cfg(self, host, port, write_dir):
        top = Element('BCVTB-client');
        ipc = SubElement(top, 'ipc');
        socket = SubElement(ipc, 'socket'
                            ,{'port':str(port),
                              'hostname':host,})
        xml_str = tostring(top, encoding='ISO-8859-1').decode();
        print(xml_str);
        
        with open(write_dir + '/' + 'socket.cfg', 'w+') as socket_file:
            socket_file.write(xml_str);
    
    def _get_file_name(self, file_path):
        path_list = file_path.split('/');
        return path_list[-1];
    
    def _log_subprocess_info(self, out, logger):
        for line in iter(out.readline, b''):
            logger.info(line.decode())
            
    def _log_subprocess_err(self, out, logger):
        for line in iter(out.readline, b''):
            logger.error(line.decode())
            
    def _get_is_subprocess_running(self, subprocess):
        if subprocess.poll() is None:
            return True;
        else:
            return False;
        
    def get_is_eplus_running(self):
        return self._get_is_subprocess_running(self._eplus_process);
    
    def end_env(self):
        """
        This method must be called after finishing using the environment
        because EnergyPlus runs on a different process. EnergyPlus process
        won't terminating until this method is called. 
        """
        self._conn.close();
        self._conn = None;
        self._socket.shutdown(socket.SHUT_RDWR);
        self._socket.close();
        self._run_eplus_outputProcessing();
        time.sleep(1);# Sleep the thread so EnergyPlus has time to do the
                      # post processing
        os.killpg(os.getpgid(self._eplus_process.pid), signal.SIGTERM);
        
        
    def _end_episode(self):
        """
        This method terminates the current EnergyPlus subprocess 
        and run the EnergyPlus output processing function (ReadVarsESO).
        
        This method is usually called by the reset() function before it
        resets the EnergyPlus environment.
        """
        self._conn.close();
        self._conn = None;
        self._run_eplus_outputProcessing();
        time.sleep(1);# Sleep the thread so EnergyPlus has time to do the
                      # post processing
        os.killpg(os.getpgid(self._eplus_process.pid), signal.SIGTERM);
        
        
    def _run_eplus_outputProcessing(self):
        eplus_outputProcessing_process =\
         subprocess.Popen('%s'
                        %(self._eplus_path + '/PostProcess/ReadVarsESO'),
                        shell = True,
                        cwd = self._eplus_working_dir + '/output',
                        stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE,
                        preexec_fn=os.setsid)
         
        
    def _assembleMsg(self, version, flag, nDb, nIn, nBl, curSimTim, Dblist):
        """
        Assemble the send msg to the EnergyPlus based on the protocal.
        Send msg must a blank space seperated string, [verison, flag, nDb
        , nIn, nBl, curSimTim, float, float, float ....]
        
        Return:
            The send msg.
        """
        ret = '';
        ret += '%d'%(version);
        ret += ' ';
        ret += '%d'%(flag);
        ret += ' ';
        ret += '%d'%(nDb);
        ret += ' ';
        ret += '%d'%(nIn);
        ret += ' ';
        ret += '%d'%(nBl);
        ret += ' ';
        ret += '%20.15e'%(curSimTim);
        ret += ' ';
        for i in range(len(Dblist)):
            ret += '%20.15e'%(Dblist[i]);
            ret += ' ';
        ret += '\n';
        
        return ret;
    
    def _disassembleMsg(self, rcv):
        rcv = rcv.split(' ');
        version = int(rcv[0]);
        flag = int(rcv[1]);
        nDb = int(rcv[2]);
        nIn = int(rcv[3]);
        nBl = int(rcv[4]);
        curSimTim = float(rcv[5]);
        Dblist = [];
        for i in range(6, len(rcv) - 1):
            Dblist.append(float(rcv[i]));
        
        return (version, flag, nDb, nIn, nBl, curSimTim, Dblist);






    

    
"""

while True:   
   rcv = c.recv(1024).decode();
   logging.info('Got message: ' + rcv);
   version, flag, nDb, nIn, nBl, curSimTim, Dblist = disassembleMsg(rcv);
   tosend = assembleMsg(version, flag, 0, nIn, nBl, curSimTim, []);
   c.send(tosend.encode());
import gym
import core.eplus_env.eplus8_6;
env = gym.make('Eplus-v0')

"""