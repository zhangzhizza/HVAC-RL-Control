#Wrap the EnergyPLus simulator into the Openai gym env
import socket              
import os
import time
import signal
import _thread
import logging
import subprocess
import pandas as pd

from shutil import copyfile
from gym import Env, spaces
from gym.envs.registration import register
from xml.etree.ElementTree import Element, SubElement, Comment, tostring

from ..util.logger import Logger 
from ..util.time import (get_hours_to_now, get_time_string, get_delta_seconds, 
                         WEEKDAY_ENCODING)
from ..util.time_interpolate import get_time_interpolate


YEAR = 1991 # Non leap year
CWD = os.getcwd();
LOG_LEVEL = 'INFO';
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";
LOGGER = Logger();
ACTION_SIZE = 2;

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
                 variable_path, idf_path,
                 incl_forecast = False, forecast_step = 36):
            
        self.logger_main = LOGGER.getLogger('EPLUS_ENV_ROOT', LOG_LEVEL, LOG_FMT);
        
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
        self._incl_forecast = incl_forecast;
        self._forecast_step = forecast_step;
        if incl_forecast:
            self._weather = self._get_weather_info(self._eplus_run_st_mon,
                                                   self._eplus_run_st_day,
                                                   self._eplus_run_ed_mon,
                                                   self._eplus_run_ed_day);
        self._epi_num = 0;
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
                                (  0.0, 1.0),
                                (  0.0, 1.0),
                                (  0.0, 33000.0)];

        
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
        eplus_logger = LOGGER.getLogger('ENERGYPLUS-EPI_%d'%self._epi_num,
                                        LOG_LEVEL, LOG_FMT);
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
        ret.append(curSimTim);
        ret.append(Dblist);
        # Remember the message header, useful when send data back to EnergyPlus
        self._eplus_msg_header = [version, flag];
        self._curSimTim = curSimTim;
        
        # Read the weather forecast
        if self._incl_forecast:
            wea_forecast = self._get_weather_forecast(curSimTim); 
            ret.append(wea_forecast);
        
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
        
        ret.append(curSimTim);
        ret.append(Dblist);
        # Read the weather forecast
        if self._incl_forecast:
            wea_forecast = self._get_weather_forecast(curSimTim);
            ret.append(wea_forecast);
        # Check if episode terminates
        is_terminal = False;
        if curSimTim >= self._eplus_one_epi_len:
            is_terminal = True;
        ret.append(is_terminal);
        # Change some attributes
        self._curSimTim = curSimTim;
        
        return ret;
        

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
        header = self._eplus_msg_header;
        tosend = self._assembleMsg(header[0], header[1], ACTION_SIZE, 0,
                                    0, self._curSimTim, 
                                   [24 for i in range(ACTION_SIZE)]);
        self._conn.send(tosend.encode());
        # Recieve the final msg from Eplus
        rcv = self._conn.recv(1024).decode();
        self._conn.send(tosend.encode()); # Send again, don't know why
        
        time.sleep(2) # Rest for a while so EnergyPlus finish post processing
        # Remove the connection
        self._conn.close();
        self._conn = None;
        # Process the output
        self._run_eplus_outputProcessing();
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
                if len(effectiveContent) > 0:
                    ret.append(int(effectiveContent[1]
                                   .split(';')[0]
                                   .strip()));
                else:
                    ret.append(int(contents[i + 1].strip()
                                                  .split('!')[0]
                                                  .strip()
                                                  .split(',')[0]
                                                  .strip()
                                                  .split(';')[0]));
                break;
            line_count += 1;
            
        return tuple(ret);
            
    def _get_weather_info(self, eplus_run_st_mon, eplus_run_st_day, 
                          eplus_run_ed_mon, eplus_run_ed_day):
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
        
        # Get start line number
        hour_by_start = get_hours_to_now(eplus_run_st_mon, eplus_run_st_day);
        stLine = hour_by_start + 8; # The weather data starts from line 9 in the .epw file
        # Get end line number
        hour_by_end = get_hours_to_now(eplus_run_ed_mon, eplus_run_ed_day) + 23;
        enLine = hour_by_end + 8 ;
        # Read data into the python list
        weather_list = [];
        with open(self._weather_path) as f:
            weather = f.readlines();
        for line_i in range(stLine, enLine + 1):
            this_line = weather[line_i].split('\n')[0].split(',')[6:];
                    # Weather data starts from 7th column
            this_line = [float(item) for item in this_line];
            weather_list.append(this_line);
        # Create the pandas dataframe
        weather_df = pd.DataFrame(weather_list);
        timeidx = pd.date_range('%d/%d/%d 01:00:00'%(eplus_run_st_mon, 
                                                     eplus_run_st_day,
                                                     YEAR),
                                periods = hour_by_end - hour_by_start + 1,
                                freq = 'H');
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
        forecastTimeList = [];
        ret = [];
        
        for i in range(1, self._forecast_step + 1):
            forecastTimeList.append(get_time_string(YEAR,
                                                    self._eplus_run_st_mon,
                                                    self._eplus_run_st_day,
                                curSimTim + i * self._eplus_run_stepsize));
        for time in forecastTimeList:
            weatherAtTime = get_time_interpolate(self._weather, time);
            ret.append(weatherAtTime);
            
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
        return self._min_max_limits;
    
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