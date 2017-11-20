#Wrap the EnergyPLus simulator into the Openai gym env
import socket              
import os
import time
import copy
import signal
import _thread
import logging
import subprocess
import threading
import pandas as pd

from shutil import copyfile
from gym import Env, spaces
from gym.envs.registration import register
from xml.etree.ElementTree import Element, SubElement, Comment, tostring

from ..util.logger import Logger 
from ..util.time import (get_hours_to_now, get_time_string, get_delta_seconds, 
                         WEEKDAY_ENCODING)
from ..util.time_interpolate import get_time_interpolate


WEATHER_FORECAST_COLS_SELECT = {'tmy3': [0, 2, 8, 9],
                                'actW': [0, 1, 5, 6]}
YEAR = 1991 # Non leap year
CWD = os.getcwd();
LOG_LEVEL_MAIN = 'INFO';
LOG_LEVEL_EPLS = 'INFO'
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";
ACTION_SIZE = 2 * 5;


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
    incl_forecast: bool
      Whether to include forecasted weather in the state observation. 
    forecast_hour: int
      How many steps for the weather forecast. 
    env_name: str
      The environment name. 

    Attributes
    ----------
    """

    def __init__(self, eplus_path, weather_path, bcvtb_path, variable_path, idf_path, env_name,
                 min_max_limits, incl_forecast = False, forecastSource = 'tmy3', forecastFilePath = None,
                 forecast_hour = 12, act_repeat = 1):
        self._env_name = env_name;
        self._thread_name = threading.current_thread().getName();
        self.logger_main = Logger().getLogger('EPLUS_ENV_%s_%s_ROOT'%(env_name, self._thread_name), 
                                            LOG_LEVEL_MAIN, LOG_FMT);
        
        # Set the environment variable for bcvtb
        os.environ['BCVTB_HOME'] = bcvtb_path;
        
        # Create a socket for communication with the EnergyPlus
        self.logger_main.debug('Creating socket for communication...')
        s = socket.socket()
        host = socket.gethostname() # Get local machine name
        s.bind((host, 0))           # Bind to the host and any available port
        sockname = s.getsockname();
        port = sockname[1];         # Get the port number
        s.listen(60)                # Listen on request
        self.logger_main.debug('Socket is listening on host %s port %d'%(sockname));
  
        self._env_working_dir_parent = self._get_eplus_working_folder(CWD, '-%s-res'%(env_name));
        os.makedirs(self._env_working_dir_parent);
        self._host = host;
        self._port = port;
        self._socket = s;
        self._eplus_path = eplus_path
        self._weather_path = weather_path
        self._variable_path = variable_path
        self._idf_path = idf_path
        self._episode_existed = False;
        (self._eplus_run_st_mon, self._eplus_run_st_day,
         self._eplus_run_ed_mon, self._eplus_run_ed_day,
         self._eplus_run_st_weekday,
         self._eplus_run_stepsize) = self._get_eplus_run_info(idf_path);
        self._eplus_run_stepsize = 3600 / self._eplus_run_stepsize 
                                                            # Stepsize in second
        self._eplus_one_epi_len = self._get_one_epi_len(self._eplus_run_st_mon,
                                                        self._eplus_run_st_day,
                                                        self._eplus_run_ed_mon,
                                                        self._eplus_run_ed_day);
        self._weatherForecastSrc = forecastSource;
        self._incl_forecast = incl_forecast;
        self._forecast_hour = forecast_hour;
        if incl_forecast:
            self._weather = self._get_weather_info(self._eplus_run_st_mon,
                                                   self._eplus_run_st_day,
                                                   self._eplus_run_ed_mon,
                                                   self._eplus_run_ed_day,
                                                   self._eplus_run_stepsize, 
                                                   self._weatherForecastSrc);
        self._epi_num = 0;
        self._act_repeat = act_repeat;

        """legacy env
        env_5702x_82_list = {'IW-v570202', 'IW-eval-v570202', 'IW-v570203', 'IW-eval-v570203',
                           'IW-v570204', 'IW-eval-v570204', 'IW-v82'};
        if (('Eplus-v0' == env_name) or ('Eplus-forecast-v0' == env_name) \
            or ('Eplus-eval-v0' == env_name) or ('Eplus-eval-multiagent-v0' == env_name)):
            self._min_max_limits = [(-16.7, 26.0),
                                (  0.0, 100.0),
                                (  0.0, 23.1),
                                (  0.0, 360.0),
                                (  0.0, 389.0),
                                (  0.0, 905.0),
                                ( 15.0, 30.0),
                                ( 15.0, 30.0),
                                ( 15.0, 30.0),
                                ( 15.0, 30.0),
                                (  0.0, 100.0),
                                (  0.5, 1.0),
                                (  0.0, 100.0),
                                (  0.0, 1.0),
                                (  0.0, 33000.0)];

        elif (('Eplus-v1' == env_name) or ('Eplus-forecast-v1' == env_name) \
            or ('Eplus-eval-v1' == env_name) or ('Eplus-eval-multiagent-v1' == env_name) \
            or ('Eplus-multiagent-v1' == env_name)):
            self._min_max_limits = [(-16.7, 26.0),
                                (  0.0, 100.0),
                                (  0.0, 23.1),
                                (  0.0, 360.0),
                                (  0.0, 389.0),
                                (  0.0, 905.0),
                                ( 15.0, 30.0),
                                ( 15.0, 30.0),
                                ( 15.0, 30.0),
                                (  0.0, 100.0),
                                (  0.0, 100.0),
                                (  0.0, 1.0),
                                (  0.0, 33000.0)];

        elif (('Eplus-v3' == env_name) or ('Eplus-forecast-v3' == env_name) \
            or ('Eplus-eval-v3' == env_name) or ('Eplus-eval-multiagent-v3' == env_name) \
            or ('Eplus-multiagent-v3' == env_name)):
            self._min_max_limits = [(-16.7, 26.0),
                                (  0.0, 100.0),
                                (  0.0, 23.1),
                                (  0.0, 360.0),
                                (  0.0, 389.0),
                                (  0.0, 905.0),
                                ( 15.0, 30.0),
                                ( 15.0, 30.0),
                                ( 15.0, 30.0),
                                (  0.0, 100.0),
                                (  0.0, 100.0),
                                (  0.0, 33000.0)];

        elif (('IW-v57' == env_name) or ('IW-eval-v57' == env_name)): ### Change

            self._min_max_limits = [(-13.0, 26.0), # OA
                                    ( 0.0, 100.0), # RH
                                    ( 0.0, 11.0),  # WS
                                    ( 0.0, 360.0), # WD
                                    ( 0.0, 378.0), # DifS
                                    ( 0.0, 1000),  # DirS 
                                    ( -30.0, 30.0),  # OAESSPs
                                    ( 20.0, 75.0), # SWTSSP
                                    ( 10.0, 30.0), # IATSSP
                                    ( 10.0, 30.0), # IAT
                                    ( 0.0, 85.0)]  # HTDMD ;

        elif (('IW-v5702' == env_name) or ('IW-eval-v5702' == env_name)): ### Change

            self._min_max_limits = [(-13.0, 26.0), # OA
                                    ( 0.0, 100.0), # RH
                                    ( 0.0, 11.0),  # WS
                                    ( 0.0, 360.0), # WD
                                    ( 0.0, 378.0), # DifS
                                    ( 0.0, 1000),  # DirS 
                                    ( -30.0, 30.0),  # OAESSPs
                                    ( 0.0, 100.0), # PPD
                                    ( 10.0, 30.0), # IATSSP
                                    ( 10.0, 30.0), # IAT
                                    ( 10.0, 30.0), # IAT Logics
                                    ( 0.0,  1.0), # Occupy flag
                                    ( 0.0, 85.0)]  # HTDMD ;

        
        elif (env_name in env_5702x_82_list): ### Change
            self._min_max_limits = [(-13.0, 26.0), # OA
                                    ( 0.0, 100.0), # RH
                                    ( 0.0, 11.0),  # WS
                                    ( 0.0, 360.0), # WD
                                    ( 0.0, 378.0), # DifS
                                    ( 0.0, 1000),  # DirS 
                                    ( -30.0, 30.0),  # OAESSPs
                                    ( 0.0, 100.0), # PPD
                                    ( 18.0, 25.0), # IATSSP
                                    ( 18.0, 25.0), # IAT
                                    ( 18.0, 25.0), # IAT Logics
                                    ( 0.0,  1.0), # Occupy flag
                                    ( 0.0, 85.0)]  # HTDMD ;
        """
        self._min_max_limits = min_max_limits;

 
    def _reset(self):
        """Reset the environment.

        This method does the followings:
        1: Make a new EnergyPlus working directory
        2: Copy .idf and variables.cfg file to the working directory
        3: Create the socket.cfg file in the working directory
        4: Create the EnergyPlus subprocess
        5: Establish the connection with EnergyPlus
        6: Read the first sensor data from the EnergyPlus
        
        Return: (float, [float], boolean) or (float, [float], [[float]], boolean)
            Return a tuple with length 3 or 4, depending on whether to generate
            the weather forecast. The index 0 is current_simulation_time in second, 
            index 1 is EnergyPlus results in 1-D python list requested by the 
            variable.cfg, index 3 (if generate the future weather forecast)
            is a 2-D python list of weather forecast with rows the weather of 
            one step and cols the weather variables (like oa or rh) - the 
            order of the weather variables is the same as the .epw file, 
            index 4 is the boolean indicating whether episode terminal.
        """
        ret = [];
        
        # End the last episode if exists
        if self._episode_existed:
            self._end_episode()
            self.logger_main.debug('Last EnergyPlus process has been closed. ')
            self._epi_num += 1;
        
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
                                  'variables.cfg'); # Variable file must be with this name
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
        eplus_logger = Logger().getLogger('EPLUS_ENV_%s_%s-EPLUSPROCESS_EPI_%d'
                                        %(self._env_name, self._thread_name, self._epi_num),
                                        LOG_LEVEL_EPLS, LOG_FMT);
        _thread.start_new_thread(self._log_subprocess_info,
                                (eplus_process.stdout,
                                 eplus_logger));
        _thread.start_new_thread(self._log_subprocess_err,
                                (eplus_process.stderr,
                                 eplus_logger));                       
            
        # Establish connection with EnergyPlus
        conn, addr = self._socket.accept()     # Establish connection with client.
        self.logger_main.debug('Got connection from %s at port %d.'%(addr));
        # Start the first data exchange
        rcv_1st = conn.recv(2048).decode(encoding = 'ISO-8859-1');
        self.logger_main.debug('Got the first message successfully: ' + rcv_1st);
        version, flag, nDb, nIn, nBl, curSimTim, Dblist \
                                                = self._disassembleMsg(rcv_1st);
        ret.append(curSimTim);
        ret.append(Dblist);
        # Remember the message header, useful when send data back to EnergyPlus
        self._eplus_msg_header = [version, flag];
        self._curSimTim = curSimTim;
        
        # Read the weather forecast
        if self._incl_forecast:
            wea_forecast = self._get_weather_forecast(curSimTim); 
            ret[-1].extend(wea_forecast);
        
        # Check if episode terminates
        is_terminal = False;
        if curSimTim >= self._eplus_one_epi_len:
            is_terminal = True;
        ret.append(is_terminal);
        # Change some attributes
        self._conn = conn;
        self._eplus_working_dir = eplus_working_dir;
        self._episode_existed = True;
        
        # Process for episode terminal
        if is_terminal:
            self._end_episode();
            
        return tuple(ret)

    def _step(self, action):
        """Execute the specified action.
        
        This method does the followings:
        1: Send a list of float to EnergyPlus
        2: Recieve EnergyPlus results for the next step (state)

        Parameters
        ----------
        action: python list of float
          Control actions that will be passed to the EnergyPlus

        Return: (float, [float], boolean) or (float, [float], [[float]], boolean)
                or None (only if the environment has reached the terminal)
            Return a tuple with length 3 or 4, depending on whether to generate
            the weather forecast. The index 0 is current_simulation_time in second, 
            index 1 is EnergyPlus results in 1-D python list requested by the 
            variable.cfg, index 3 (if generate the future weather forecast)
            is a 2-D python list of weather forecast with rows the weather of 
            one step and cols the weather variables (like oa or rh) - the 
            order of the weather variables is the same as the .epw file, 
            index 4 is the boolean indicating whether episode terminal.
        """
        # Check terminal
        if self._curSimTim >= self._eplus_one_epi_len:
            return None;
        ret = [];
        # Send to the EnergyPlus
        act_repeat_i = 0;
        is_terminal = False;
        curSimTim = self._curSimTim;
        integral_item_list = []; # Now just hard code to the energy, the last item in state observation
        while act_repeat_i < self._act_repeat and (not is_terminal):
            self.logger_main.debug('Perform one step.')
            header = self._eplus_msg_header;
            runFlag = 0 # 0 is normal flag
            tosend = self._assembleMsg(header[0], runFlag, len(action), 0,
                                       0, curSimTim, action);
            self._conn.send(tosend.encode());
            # Recieve from the EnergyPlus
            rcv = self._conn.recv(2048).decode(encoding = 'ISO-8859-1');
            self.logger_main.debug('Got message successfully: ' + rcv);
            # Process received msg        
            version, flag, nDb, nIn, nBl, curSimTim, Dblist \
                                        = self._disassembleMsg(rcv);
            integral_item_list.append(Dblist[-1]); # Hard code that the last item is the integral item
            if curSimTim >= self._eplus_one_epi_len:
                is_terminal = True;
            act_repeat_i += 1;
        # Construct the return. The return is the state observation of the last step plus the integral item
        ret.append(curSimTim);
        Dblist[-1] = 1.0 * sum(integral_item_list)/len(integral_item_list);
        ret.append(Dblist);
        # Read the weather forecast
        if self._incl_forecast:
            wea_forecast = self._get_weather_forecast(curSimTim);
            ret[-1].extend(wea_forecast);
        # Add terminal status
        ret.append(is_terminal);
        # Change some attributes
        self._curSimTim = curSimTim;
        return ret;
        

    def _render(self, mode='human', close=False):
        pass;
    
    def _create_eplus(self, eplus_path, weather_path, 
                      idf_path, out_path, eplus_working_dir):
        
        eplus_process = subprocess.Popen('%s -w %s -d %s %s'
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
        self._end_episode();
        self._socket.shutdown(socket.SHUT_RDWR);
        self._socket.close();
        
        
        
    def _end_episode(self):
        """
        This method terminates the current EnergyPlus subprocess 
        and run the EnergyPlus output processing function (ReadVarsESO).
        
        This method is usually called by the reset() function before it
        resets the EnergyPlus environment.
        """
        # Send the final msg to EnergyPlus
        #header = self._eplus_msg_header;
        #tosend = self._assembleMsg(header[0], 1.0, ACTION_SIZE, 0,
        #                            0, self._curSimTim, 
        #                           [24 for i in range(ACTION_SIZE)]);
        #self.logger_main.debug('Send final msg to Eplus.');
        #self._conn.send(tosend.encode());
        # Recieve the final msg from Eplus
        #rcv = self._conn.recv(2048).decode(encoding = 'ISO-8859-1');
        #self.logger_main.debug('Final msh from Eplus: %s', rcv)
        #self._conn.send(tosend.encode()); # Send again, don't know why
        
        #time.sleep(0.2) # Rest for a while so EnergyPlus finish post processing
        # Remove the connection
        self._conn.close();
        self._conn = None;
        # Process the output
        #self._run_eplus_outputProcessing();
        time.sleep(1);# Sleep the thread so EnergyPlus has time to do the
                      # post processing

        # Kill subprocess
        os.killpg(self._eplus_process.pid, signal.SIGTERM);
        
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
    
    def _get_eplus_run_info(self, idf_path):
        """
        This method read the .idf file and find the running start month, start
        date, end month, end date and the step size.
        
        Args:
            idf_path: String
                The .idf file path.
        
        Return: (int, int, int, int, int, int)
            (start month, start date, end month, end date, start weekday, 
            step size)
        """
        ret = [];
        
        with open(idf_path, encoding = 'ISO-8859-1') as idf:
            contents = idf.readlines();
        
        # Run period
        tgtIndex = None;
        
        for i in range(len(contents)):
            line = contents[i];
            effectiveContent = line.strip().split('!')[0] # Ignore contents after '!'
            effectiveContent = effectiveContent.strip().split(',')[0]
                                                          # Remove tailing ','
            if effectiveContent.lower() == 'runperiod':
                tgtIndex = i;
                break;
        
        for i in range(2, 6):
            ret.append(int(contents[tgtIndex + i].strip()
                                                 .split('!')[0]
                                                 .strip()
                                                 .split(',')[0]
                                                 .strip()
                                                 .split(';')[0]));
        # Start weekday
        ret.append(WEEKDAY_ENCODING[contents[tgtIndex + i + 1].strip()
                                                          .split('!')[0]
                                                          .strip()
                                                          .split(',')[0]
                                                          .strip()
                                                          .split(';')[0]
                                                          .strip()
                                                          .lower()]);
        # Step size
        line_count = 0;
        for line in contents:
            effectiveContent = line.strip().split('!')[0] # Ignore contents after '!'
            effectiveContent = effectiveContent.strip().split(',');
            if effectiveContent[0].strip().lower() == 'timestep':
                if len(effectiveContent) > 1 and len(effectiveContent[1]) > 0:
                    ret.append(int(effectiveContent[1]
                                   .split(';')[0]
                                   .strip()));
                else:
                    ret.append(int(contents[line_count + 1].strip()
                                                  .split('!')[0]
                                                  .strip()
                                                  .split(',')[0]
                                                  .strip()
                                                  .split(';')[0]));
                break;
            line_count += 1;
            
        return tuple(ret);
            
    def _get_weather_info(self, eplus_run_st_mon, eplus_run_st_day, eplus_run_ed_mon, 
                        eplus_run_ed_day, eplus_run_stepsize, weatherForecastSrc):
        """
        This function read the .epw file and extract the relevant section
        (defined by the .idf runperiod) to a pd.DataFrame;
        
        Args:
            eplus_run_st_mon, eplus_run_st_day,
            eplus_run_ed_mon, eplus_run_ed_day: String
                EnergyPlus run start month, start day, end month, end day.
        
        Return: pd.DataFrame
            A pd.DataFrame with weather info, 2-D, index is time, each row
            is the weather for a time step, each col is a weather variable.
        """
        # Set some info based on tmy3 or actual weather file
        if weatherForecastSrc == 'tmy3':
            lineRowBias = 8; # The weather data starts from line 9 in the .epw file, line 2 in the real weather file
            lineColBias = 6; # The Weather data starts from column 7 in the .epw file, col 2 in the real weather file
            weatherFileStepSize = 1;
            tgt_idxs = WEATHER_FORECAST_COLS_SELECT['tmy3']
            stHour = '01:00:00';

        else:
            lineRowBias = 1;
            lineColBias = 1;
            weatherFileStepSize = int(3600/eplus_run_stepsize);
            tgt_idxs = WEATHER_FORECAST_COLS_SELECT['actW']
            stHour = '01:00:00' if weatherFileStepSize == 1 else\
                    '00:%02d:00'%(60/weatherFileStepSize);

        
        # Get start line number
        hour_by_start = get_hours_to_now(eplus_run_st_mon, eplus_run_st_day);
        file_line_by_start = hour_by_start * weatherFileStepSize;
        stLine = file_line_by_start + lineRowBias; 
        # Get end line number
        hour_by_end = get_hours_to_now(eplus_run_ed_mon, eplus_run_ed_day) + 24;
        file_line_by_end = hour_by_end * weatherFileStepSize;
        enLine = file_line_by_end + lineRowBias ;
        # Read data into the python list
        weather_list = [];
        weatherForecastFile = self._weather_path if weatherForecastSrc == 'tmy3' \
                            else weatherForecastSrc;
        with open(weatherForecastFile) as f:
            weather = f.readlines();
        for line_i in range(stLine, enLine):
            this_line = weather[line_i].split('\n')[0].split(',')[lineColBias:];    
            this_line = [float(this_line[tgt_idx]) for tgt_idx in tgt_idxs];
            weather_list.append(this_line);
        # Create the pandas dataframe
        weather_df = pd.DataFrame(weather_list);
        timeidx = pd.date_range('%d/%d/%d %s'%(eplus_run_st_mon, 
                                               eplus_run_st_day,
                                               YEAR, stHour),
                                periods = (hour_by_end - hour_by_start) * weatherFileStepSize,
                                freq = '%dMin'%(60/weatherFileStepSize));
        weather_df.set_index(timeidx, inplace = True);
        
        return weather_df;
        
    def _get_weather_forecast(self, curSimTim):
        """
        This method gets the future steps' weather information from the 
        weather file. 
        
        Return: 2-D python list.
            Index 0 is the weather information for one step;
            Index 1 is the weather variables with the same order as the
            .epw file. 
        """
        forecastStepSize = 3600; # seconds
        forecastTimeList = [];
        ret = [];
        
        for i in range(1, self._forecast_hour + 1):
            forecastTimeList.append(get_time_string(YEAR,
                                                    self._eplus_run_st_mon,
                                                    self._eplus_run_st_day,
                                curSimTim + i * forecastStepSize));
        for time in forecastTimeList:
            weatherAtTime = get_time_interpolate(self._weather, time);
            ret.extend(weatherAtTime.tolist());
            
        return ret;
            
    def _get_one_epi_len(self, st_mon, st_day, ed_mon, ed_day):
        """
        Get the length of one episode (One EnergyPlus process run to the end).
        
        Args:
            st_mon, st_day, ed_mon, ed_day: int
                The EnergyPlus simulation start month, start day, end month, 
                end day.
        
        Return: int
            The simulation time step that the simulation ends. 
        """
        return get_delta_seconds(YEAR, st_mon, st_day, ed_mon, ed_day);
    
    @property
    def min_max_limits(self):
        """
        Return the min_max_limits for all state features. 
        
        Return: python list of tuple.
            In the order of the state features, and the index 0 of the tuple
            is the minimum value, index 1 is the maximum value. 
        """
        return copy.deepcopy(self._min_max_limits);
    
    @property
    def start_year(self):
        """
        Return the EnergyPlus simulaton year.
        
        Return: int
        """
        return YEAR;
    
    @property
    def start_mon(self):
        """
        Return the EnergyPlus simulaton start month.
        
        Return: int
        """
        return self._eplus_run_st_mon;
    
    @property
    def start_day(self):
        """
        Return the EnergyPlus simulaton start day of the month.
        
        Return: int
        """
        return self._eplus_run_st_day;
    
    @property
    def start_weekday(self):
        """
        Return the EnergyPlus simulaton start weekday. 0 is Monday, 6 is Sunday.
        
        Return: int
        """
        return self._eplus_run_st_weekday;

    
