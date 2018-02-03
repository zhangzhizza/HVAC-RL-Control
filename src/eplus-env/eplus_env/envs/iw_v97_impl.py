#Wrap the EnergyPLus simulator into the Openai gym env
import socket              
import os
import time
import ast
import copy
import signal
import _thread
import logging
import subprocess
import threading
import pandas as pd
import numpy as np
import requests
import json
import csv

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.poolmanager import PoolManager

from shutil import copyfile
from datetime import datetime
from pysolar.solar import get_altitude
from gym import Env, spaces
from gym.envs.registration import register
from xml.etree.ElementTree import Element, SubElement, Comment, tostring

from ..util.logger import Logger 
from ..util.time import (get_hours_to_now, get_time_string, get_delta_seconds, 
                         WEEKDAY_ENCODING, getSecondFromStartOfYear)
from ..util.time_interpolate import get_time_interpolate
from ..util.solarCalculator import getSolarBreakDown


WEATHER_FORECAST_COLS_SELECT = {'tmy3': [0, 2, 8, 9],
                                'actW': [0, 1, 5, 6]}
YEAR = 2017 # Non leap year
CWD = os.getcwd();
LOG_LEVEL_MAIN = 'DEBUG';
LOG_LEVEL_ENV = 'INFO'
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";
ACTION_SIZE = 2 * 5;


class IW_IMP_V97(Env):
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

    def __init__(self, site_server_ip, rd_port, wt_port, env_name, defaultObValues, localLat, localLong, ctrl_step_size_s, 
                 min_max_limits, incl_forecast = False, forecastRandMode = 'normal', forecastRandStd = 0.15,
                 forecastSource = 'tmy3', forecastFilePath = None, forecast_hour = 12, act_repeat = 1):
        self._env_name = env_name;
        self._thread_name = threading.current_thread().getName();
        self.logger_main = Logger().getLogger('ENV_%s_%s'%(env_name, self._thread_name), 
                                            LOG_LEVEL_MAIN, LOG_FMT);
        self._site_server_ip = site_server_ip;
        self._rd_port = rd_port;
        self._wt_port = wt_port;
        self._host = site_server_ip;
        self._localLat = localLat;
        self._localLong = localLong;
        self._defaultObValues = defaultObValues;
        stDate = datetime.now();
        self._st_mon = stDate.month;
        self._st_day = stDate.day;
        self._st_weekday = stDate.weekday(); # 0 is Mondays
        self._ctrl_step_size_s = ctrl_step_size_s;
        self._weatherForecastSrc = forecastSource;
        self._forecastRandMode = 'normal';
        self._forecastRandStd = 0.15;
        self._incl_forecast = incl_forecast;
        self._forecast_hour = forecast_hour;
        if incl_forecast:
            pass;
        self._act_repeat = act_repeat;
        self._min_max_limits = min_max_limits;

 
    def _reset(self):
        """Reset the environment.

        This method does the followings:
        1: Establish the connection with the site BAS data collection server
        6: Read the first sensor data from the site server
        
        Return: (float, [float], boolean)
            Index 0 is current_simulation_time in second, 
            index 1 is observations in 1-D python list with order
            [OAT-TOC, OAH, WS-TOMS, WD, SOLDIF-CAL, SOLDIR-CAL, HWOEN-TOC, PPD-CAL, MULSSP-TOC, IAT-TOC, IATSSP-TOC, OCCP-CAL, HTDMD-KW],
            index 4 is the boolean indicating whether episode terminal.
        """
        ret = []; 
        nowDatetime = datetime.now()                 
        # Read from BAS
        readData = self._readFromBASHelper('getAll');
        # readData: OAT-F, OAH-%, WS-MHP, WD, HWOEN-F, MULSSP-F, IAT-F, IATSSP-F, HTDMD-KW, 15 Values for AMV
        (readDataUnitChanged, solDif, solDir, ppdCal, occpMode) = self._processRawReadDataAndQueryPi(readData, nowDatetime);
        # Set a lower bound for thermal comfort based on the indoor air temperature (IAT must be > 20 C) in case no body report AMV 
        # State ob: OAT-TOC, OAH, WS-TOMS, WD, SOLDIF-CAL, SOLDIR-CAL, HWOEN-TOC, PPD-CAL, MULSSP-TOC, IAT-TOC, IATSSP-TOC, OCCP-CAL, HTDMD-TOKWH
        prcdStateOb = [readDataUnitChanged[0], readDataUnitChanged[1], readDataUnitChanged[2], readDataUnitChanged[3], solDif, solDir,\
                       readDataUnitChanged[4], ppdCal, readDataUnitChanged[5], readDataUnitChanged[6], readDataUnitChanged[7], occpMode, readDataUnitChanged[8]];
        self.logger_main.info('State observation is: %s'%(prcdStateOb));
        curObTim = getSecondFromStartOfYear(nowDatetime);
        self.logger_main.info('Current time in second since the start of the year is: %d'%(curObTim));
        ret.append(curObTim);
        ret.append(prcdStateOb);
        # Read the weather forecast
        if self._incl_forecast:
            pass;    
        # Check if episode terminates
        is_terminal = False;
        ret.append(is_terminal);
        # Process for episode terminal
        if is_terminal:
            self._end_episode();
        # Setup log file
        self._env_working_dir = self._get_working_folder(CWD, '-%s-res'%(self._env_name));
        os.makedirs(self._env_working_dir);
        self._logfilename = os.path.join(self._env_working_dir, 'ob_log.csv');  
        toWriteLog = self._getToWriteLog(nowDatetime, prcdStateOb, readData[-15:]);
        logheaders=['time','OAT C','OAH %', 'WS M/S', 'WD', 'SOLDIF W/M2', 'SOLDIR W/M2', 'HWOEN C', 'PPD(PRCD) %', 'MULSSP C', \
                    'IAT C', 'IATSSP C', 'OCCP', 'HTDMD KW']
        for i in range(15):
            # 15 amv values
            logheaders.append('AMVRAW_%s'%(i));
        with open(self._logfilename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(logheaders)
            writer.writerow(toWriteLog);

        return tuple(ret);

    def _getToWriteLog(self, nowDatetime, prcdStateOb, rawAMVs):
        ret = [];
        ret.append(str(nowDatetime));
        ret.extend(prcdStateOb);
        ret.extend(rawAMVs);
        return ret;

    def _get_working_folder(self, parent_dir, dir_sig = '-run'):
        """Return working folder path string

        Assumes folders in the parent_dir have suffix -run{run
        number}. Finds the highest run number and sets the output folder
        to that number + 1. 

        Parameters
        ----------
        parent_dir: str
        dir_sig: str, working directory suffix name

        Returns
        -------
        parent_dir/run_dir
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

        parent_dir = os.path.join(parent_dir, 'env')
        parent_dir = parent_dir + '%s%d'%(dir_sig, experiment_id)
        return parent_dir

    def _readFromBASHelper(self, readCode, isShowInfoLog = True):
        # Establish connection with site BAS data collection server
        rd_s = socket.socket(); # Read sensor data socket
        rd_s.connect((self._site_server_ip, self._rd_port));
        if isShowInfoLog:
            self.logger_main.info('Built connection with the BAS SDC read server.')
        rd_s.sendall(bytearray(readCode, encoding = 'utf-8')) # Send get request to the site read server
        # Start the data exchange with the site read server
        rcv_from_bas_server = rd_s.recv(2048).decode(encoding = 'utf-8')
        self.logger_main.debug('Got the message successfully from the BAS SDC read server: ' + rcv_from_bas_server);
        flag, retMsg, nDb, readData = self._disassembleRSMsg(rcv_from_bas_server);
        if flag == 0:
            self.logger_main.warning('Read from BACnet server error with message: %s, use default observations instead'%(retMsg));
            if readCode.lower() == 'getall':
                readData = self._defaultObValues;
            elif readCode.lower() == 'getamv':
                readData = self._defaultObValues[9:];
            elif readCode.lower() == 'getenergy':
                readData = [self._defaultObValues[8]];
            elif readCode.lower() == 'getiarh':
                readData = [50]
            else:
                pass
        else:
            # Process the None values because of float conversion error
            defaultObValuesShiftIdx = 0;
            if readCode.lower() == 'getall':
                defaultObValuesShiftIdx = 0;
            elif readCode.lower() == 'getamv':
                defaultObValuesShiftIdx = 9;
            elif readCode.lower() == 'getenergy':
                defaultObValuesShiftIdx = 8;
            elif readCode.lower() == 'getiarh':
                defaultObValuesShiftIdx = -1 # Not in the default values
            else:
                pass;
            if None in readData:
                if defaultObValuesShiftIdx > 0:
                    for i in range(len(readData)):
                        if readData[i] == None:
                            readData[i] = self._defaultObValues[i + defaultObValuesShiftIdx];
                else:
                    if readCode.lower() == 'getiarh':
                        readData[0] = 50;
                self.logger_main.warning('Data read from BACnet server contains None values, raw data is: %s, '\
                                         'processed data using the default values is: %s'%(rcv_from_bas_server, readData));
        if isShowInfoLog:
            self.logger_main.info('Data read from BAS SDC read server: %s'%(readData));
        return readData;

    def _processRawReadDataAndQueryPi(self, readData, nowDatetime):
        # Change unit of the readData
        readDataUnitChanged = self._getPrcdUnitChangedReadData(readData);
        iat = readDataUnitChanged[6];
        met = 1.2 #met
        airVel = 0.1 #m/s
        clo = 1.0 #clo
        # Get occp mode 
        self.logger_main.debug('Now is %s'%(nowDatetime));
        occpMode = self._getCurrentOccpMode(nowDatetime)
        # Get PPD
        ppdCal = self._getOnePPDFromAllAMV(readDataUnitChanged[9:]);
        if ppdCal == 0: # Indicating no one votes or every one is comfortable
            # IARH is needed for fanger PPD
            iarhFromBAS = self._readFromBASHelper('getIARH', False);
            pmvFanger, ppdFanger = fangerPMV(iat, iat, iarhFromBAS, airVel, met, clo);
            self.logger_main.warning('Collected PPD is 0.0, so Fanger-PPD: %s is used'%(ppdFanger))
            ppdCal = ppdFanger;
        # Get soldif and soldir
        solTotal, solTotalMsgBack = self._getSolTotalNow();
        self.logger_main.debug('Global solar radiation from PI is: %s'%(solTotal));
        if solTotal == None:
            self.logger_main.warning('Query from PI for global solRad failed, use value 0.0 instead, raw return from PI is: %s'%(solTotalMsgBack));
            solTotal = 0.0;
        nowSolAlt = get_altitude(self._localLat, self._localLong, nowDatetime); # In degree
        solDir, solDif = getSolarBreakDown(solTotal, nowSolAlt);
        self.logger_main.debug('Direct solar rad is: %s, diffuse solar rad is: %s, solar altitude is: %s'%(solDir, solDif, nowSolAlt));
        return (readDataUnitChanged, solDif, solDir, ppdCal, occpMode)



    def _step(self, action):
        """Execute the specified action.
        
        This method does the followings:
        1: Send a control actions to BAS
        2: Recieve observations from BAS

        Parameters
        ----------
        action: python list of float
          Control actions

        Return: (float, [float], boolean)
            Index 0 is current_simulation_time in second, 
            index 1 is observations in 1-D python list with order
            [OAT-TOC, OAH, WS-TOMS, WD, SOLDIF-CAL, SOLDIR-CAL, HWOEN-TOC, PPD-CAL, MULSSP-TOC, IAT-TOC, IATSSP-TOC, OCCP-CAL, HTDMD-KW],
            index 4 is the boolean indicating whether episode terminal.
        """
        ret = [];
        # Before send, check IW AMV
        amvFromBAS = self._readFromBASHelper('getAMV');
        ppdCal_base = self._getOnePPDFromAllAMV(amvFromBAS);
        self.logger_main.info('Base PPD before step is: %s'%(ppdCal_base))
        # Send to BAS
        actionUnitChanged = self._getPrcdUnitChangedActions(action);
        is_terminal = False;
        integral_energy_list = [];
        wt_s = socket.socket(); # Write action socket
        wt_s.connect((self._site_server_ip, self._wt_port));
        self.logger_main.info('Built connection with the BAS SDC write server.')
        toSend = ['write', len(actionUnitChanged)];
        toSend.extend(actionUnitChanged);
        wt_s.sendall(bytearray(str(toSend), encoding = 'utf-8')) # Send to write server [codeString, #ofData, data1, data2, ...]
        self.logger_main.info('Sent action %s to the BAS'%(toSend[2:]))
        wt_s_recv = wt_s.recv(2048).decode(encoding = 'utf-8'); # Receive is [flag, msg]
        wt_s_recv = ast.literal_eval(wt_s_recv);
        if wt_s_recv[0] == 0:
            self.logger_main.warning('Write to BAS has error: %s'%wt_s_recv[1]);
        else:
            self.logger_main.info('Write to BAS success!');
        # Read from BAS
        # Wait for control step finishes, except if IW PPD increases
        controlStepLenS = self._ctrl_step_size_s * self._act_repeat;
        baseTime = time.time();
        self.logger_main.info('Wait for %d seconds to finish this control step'%(controlStepLenS));
        loopCount = 0
        queryFreq = 10 #Seconds
        energyQueryFreq = 10 #Seconds
        energyQueryWait = 120 #Seconds
        while (time.time() - baseTime) < controlStepLenS:
            # Check ppd
            amvFromBAS = self._readFromBASHelper('getAMV', False);
            ppdCal_now = self._getOnePPDFromAllAMV(amvFromBAS);
            if ppdCal_now > ppdCal_base:
                # Jump out of the while and finish this step
                self.logger_main.info('IW AMV profile changed to: %s, now PPD is %s (old PPD is %s), jump out of the step loop'
                                      %(amvFromBAS, ppdCal_now, ppdCal_base));
                break;
            # Check energy frequently because we want the average energy demand of the last control step
            # But check until after 2 min to let energy demand stable
            if loopCount != 0 and loopCount%(energyQueryFreq/queryFreq) == 0 and loopCount > energyQueryWait/queryFreq:
                energyFromBAS = self._readFromBASHelper('getenergy', True)
                energy_now = energyFromBAS[0]; # in kW
                integral_energy_list.append(energy_now);
                self.logger_main.info('Energy demand is: %s'%(energy_now));
            self.logger_main.debug('Remaining time to finish this step: %ds'%(controlStepLenS - (time.time() - baseTime)))
            time.sleep(queryFreq) # Check PPD every 5 seconds
            loopCount += 1;
        # Do the reading from BAS
        nowDatetime = datetime.now()
        readData = self._readFromBASHelper('getAll');
        # readData: OAT-F, OAH-%, WS-MHP, WD, HWOEN-F, MULSSP-F, IAT-F, IATSSP-F, HTDMD-KW, 15 Values for AMV
        (readDataUnitChanged, solDif, solDir, ppdCal, occpMode) = self._processRawReadDataAndQueryPi(readData, nowDatetime, self._iat_thres);
        integral_energy_list.append(readDataUnitChanged[8]);
        # State ob: OAT-TOC, OAH, WS-TOMS, WD, SOLDIF-CAL, SOLDIR-CAL, HWOEN-TOC, PPD-CAL, MULSSP-TOC, IAT-TOC, IATSSP-TOC, OCCP-CAL, HTDMD-TOKWH
        prcdStateOb = [readDataUnitChanged[0], readDataUnitChanged[1], readDataUnitChanged[2], readDataUnitChanged[3], solDif, solDir,\
                       readDataUnitChanged[4], ppdCal, readDataUnitChanged[5], readDataUnitChanged[6], readDataUnitChanged[7], occpMode, \
                       sum(integral_energy_list)/len(integral_energy_list)];
        self.logger_main.info('State observation is: %s'%(prcdStateOb));
        curObTim = getSecondFromStartOfYear(nowDatetime);
        self.logger_main.info('Current time in second since the start of the year is: %d'%(curObTim));
        ret.append(curObTim);
        ret.append(prcdStateOb);
        # Read the weather forecast
        if self._incl_forecast:
            pass;
        # Add terminal status
        ret.append(is_terminal);
        # Log observations to file
        toWriteLog = self._getToWriteLog(nowDatetime, prcdStateOb, readData[-15:]);
        with open(self._logfilename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(toWriteLog);
        return ret;
        

    def _render(self, mode='human', close=False):
        pass;

    def _getPrcdUnitChangedActions(self, rawAct):
        retAct = copy.deepcopy(rawAct);
        retAct[0] = retAct[0] * 1.8 + 32;
        return retAct;

    def _getCurrentOccpMode(self, nowDatetime):
        occpMode = 0;
        nowWeekday = nowDatetime.weekday(); # 0 is Monday
        nowHour = nowDatetime.hour; 
        # In Weekday
        if nowWeekday < 5:
            if nowHour >= 7 and nowHour <= 19:
                occpMode = 1;
            else:
                occpMode = 0;
        else:
            if nowHour >= 8 and nowHour <= 18:
                occpMode = 1;
            else:
                occpMode = 0;
        return occpMode;

    def _getOnePPDFromAllAMV(self, allAMV):
        """
            ret: float
                0-100
        """
        # 15 AMV in total, 0 to 4 scale, 0 is cold, 4 is hot, 2 is neutral
        allAMVScaleAbs = [abs(amv - 2.0) for amv in allAMV];
        onePPD = 100 * sum(allAMVScaleAbs) / (2.0 * len(allAMV));
        return onePPD;

    def _getSolTotalNow(self):
        class HostNameIgnoringAdapter(HTTPAdapter):
            def init_poolmanager(self, connections, maxsize, block=False):
                self.poolmanager = PoolManager(num_pools=connections,
                                       maxsize=maxsize,
                                       block=block,
                                       assert_hostname=False)
        s = requests.Session() 
        s.mount('https://', HostNameIgnoringAdapter())
        solPiUrl = 'https://128.2.109.159/piwebapi/streams/P0-MYhSMORGkyGTe9bdohw0AVSsAAAV0lOLTYyTlBVMkJWTDIwXFBISVBQU19XRUFUSEVSX1NPTEFSX1JBRElBVElPTl9BVi5QUkVTRU5UX1ZBTFVF/value' 
        r = s.get(solPiUrl, auth=('CMU_Students', 'WorkHard!ChangeWorld'));
        r = json.loads(r.text);
        if r['Good']:
            msgBack = 'Query success!'
            value = r['Value']
        else:
            msgBack = r;
            value = None;
        return [value, msgBack];

    
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
        
        
    
    def _disassembleRSMsg(self, rcv):
        """
        Disassemble the read server back message
        """
        rcv = ast.literal_eval(rcv);
        flag = rcv[0];
        retMsg = rcv[1];
        count = rcv[2];
        readData = [];
        if flag == 1:
            for i in range(3, 3 + count):
                try:
                    readData.append(float(rcv[i]));
                except ValueError as err:
                    self.logger_main.warning('Encountered error [%s] during disassembling the SDC read server return msg (msg: [%s]),'
                                            ' this record is changed to None'%(err, rcv[i]));
                    readData.append(None);
        return (flag, retMsg, count, readData);

    def _getPrcdUnitChangedReadData(self, rawData):
        # rawData: OAT-F, OAH-%, WS-MHP, WD, HWOEN-F, MULSSP-F, IAT-F, IATSSP-F, HTDMD-KW, 15 Values for AMV
        prcdData = [];
        prcdData.append(self._fToC(rawData[0]))
        prcdData.append(rawData[1])
        prcdData.append(self._mphToMs(rawData[2]))
        prcdData.append(rawData[3])
        prcdData.append(self._fToC(rawData[4]))
        prcdData.append(self._fToC(rawData[5]))
        prcdData.append(self._fToC(rawData[6]))
        prcdData.append(self._fToC(rawData[7]))
        prcdData.append(rawData[8])
        for i in range(15):
            prcdData.append(rawData[9 + i]);

        return prcdData;

    def _fToC(self, degF):
        return (degF - 32)/1.8;

    def _mphToMs(self, spdMph):
        return spdMph * 0.44704;
    

    def _addNormalRandomToForecast(self, rawForecast, forecastRandStd, min_max_limits):
        """
        Randomness is added by raw*(1 + dev), where dev is sampled from normal distribution with mean 0, std forecastRandStd.
        """
        # Sample from normal distribution for dev
        randomBase = np.random.normal(0, forecastRandStd, len(min_max_limits));
        # Caculate the randomed forecast
        randomedForecastRaw = rawForecast * (1 + randomBase);
        # Clip the randomed forecast by its limits
        min_max_limits = np.array(min_max_limits);
        randomedForecastCliped = np.clip(randomedForecastRaw, min_max_limits[:, 0], min_max_limits[:, 1]);

        return randomedForecastCliped.tolist();
    
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
        return self._st_mon;
    
    @property
    def start_day(self):
        """
        Return the EnergyPlus simulaton start day of the month.
        
        Return: int
        """
        return self._st_day;
    
    @property
    def start_weekday(self):
        """
        Return the EnergyPlus simulaton start weekday. 0 is Monday, 6 is Sunday.
        
        Return: int
        """
        return self._st_weekday;

    
