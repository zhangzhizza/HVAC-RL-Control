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
from ..util.pmvCalculator import fangerPMV

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
                 forecastSource = 'tmy3', forecastFilePath = None, forecast_hour = 12, act_repeat = 1, isPPDBk = True,
                 clo = 1.0, met = 1.2, airVel = 0.1, isMullSspLowerLimit =  False, useCSLWeather = False):
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
        self._isPPDBk = isPPDBk;
        self._clo = clo;
        self._met = met;
        self._airVel = airVel;
        self._oat = None;
        self._isMullSspLowerLimit = isMullSspLowerLimit;
        self._useCSLWeather = useCSLWeather;
 
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
        # readData: OAT-F, OAH-%, WS-MHP, WD, HWOEN-F, MULSSP-F, IAT-F, IATSSP-F, HTDMD-KW, 17 Values for AMV
        (readDataUnitChanged, solDif, solDir, ppdCal, occpMode) = self._processRawReadDataAndQueryPi(readData, nowDatetime, self._isPPDBk);
        # Set a lower bound for thermal comfort based on the indoor air temperature (IAT must be > 20 C) in case no body report AMV 
        # State ob: OAT-TOC, OAH, WS-TOMS, WD, SOLDIF-CAL, SOLDIR-CAL, HWOEN-TOC, PPD-CAL, MULSSP-TOC, IAT-TOC, IATSSP-TOC, OCCP-CAL, HTDMD-TOKWH
        prcdStateOb = [readDataUnitChanged[0], readDataUnitChanged[1], readDataUnitChanged[2], readDataUnitChanged[3], solDif, solDir,\
                       readDataUnitChanged[4], ppdCal, readDataUnitChanged[5], readDataUnitChanged[6], readDataUnitChanged[7], occpMode, readDataUnitChanged[8]];
        self._oat = readDataUnitChanged[0];
        self._iat = readDataUnitChanged[6];
        self._ocp = occpMode;
        curObTim = getSecondFromStartOfYear(nowDatetime);
        self.logger_main.info('RESET State observation in reset is: %s, current time in seconds since start of the year is: %s'%(prcdStateOb, curObTim));
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
        toWriteLog = self._getToWriteLog(nowDatetime, prcdStateOb, readData[9:]);
        logheaders=['time','OAT C','OAH %', 'WS M/S', 'WD', 'SOLDIF W/M2', 'SOLDIR W/M2', 'HWOEN C', 'PPD(PRCD) %', 'MULSSP C', \
                    'IAT C', 'IATSSP C', 'OCCP', 'HTDMD KW']
        for i in range(len(readData[9:])):
            # 17 amv values
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
            self.logger_main.info('Read from SDC server with readCode: %s'%readCode)
        rd_s.sendall(bytearray(readCode, encoding = 'utf-8')) # Send get request to the site read server
        # Start the data exchange with the site read server
        rcv_from_bas_server = rd_s.recv(4096).decode(encoding = 'utf-8')
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
            elif readCode.lower() == 'getiat':
                readData = [self._defaultObValues[6]]
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
            elif readCode.lower() == 'getiat':
                defaultObValuesShiftIdx = 6;
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

    def _processRawReadDataAndQueryPi(self, readData, nowDatetime, isPPDBk):
        # Change weather info source
        if self._useCSLWeather:
            self.logger_main.debug("Use CSL source weather data")
            cslValues, cslPiMsgBack = self._getWeatherFromCSL();
            if None in cslValues:
                self.logger_main.warning('Query from PI for CSL weather data failed, use BACnet values instead, raw return from PI is: %s'%(cslPiMsgBack));
            else:
                oatcsl, oahcsl, wscsl, wdcsl = cslValues;
                readData[0] = oatcsl;
                readData[1] = oahcsl;
                readData[2] = wscsl;
                readData[3] = wdcsl;
        # Change unit of the readData, readData: OAT-F, OAH-%, WS-MHP, WD, HWOEN-F, MULSSP-F, IAT-F, IATSSP-F, HTDMD-KW, 17 Values for AMV
        readDataUnitChanged = self._getPrcdUnitChangedReadData(readData);
        # Get occp mode 
        self.logger_main.debug('Now is %s'%(nowDatetime));
        occpMode = self._getCurrentOccpMode(nowDatetime)
        # Get PPD
        if isPPDBk:
            ppdCal = self._getPPDFangerBK(readDataUnitChanged[9:], occpMode, nowDatetime);
        else:
            ppdCal = self._getPPDAMV(readDataUnitChanged[9:]);
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

    def _processActions(self, rawAction, oat, iat, ocp):
        """
            Keep the raw action if one the following conditions is met:
                1, OAT > oatThres1
                2, IAT > iatThres1
                3, OCP == 0
        """
        oatThres1 = 5; # C
        oatThres2 = 0; # C
        iatThres1 = 22.5; # C
        if oat >= oatThres1:
            return rawAction;
        elif ocp == 0:
            return rawAction;
        else:
            if iat >= iatThres1:
                return rawAction;
            else: 
                if oat >= oatThres2:
                    minMullSsp = 30;
                    self.logger_main.info('STEP Ourdoor is between %sC and %sC, IAT is below %s and occupancy flag is %s, '
                                          'so minimum MullSSP is: %s'%(oatThres2, oatThres1, iatThres1, ocp, minMullSsp));
                else:
                    minMullSsp = 35;
                    self.logger_main.info('STEP Ourdoor is below %sC, IAT is below %s and occupancy flag is %s, '
                                          'so minimum MullSSP is: %s'%(oatThres2, iatThres1, ocp, minMullSsp));
                rawAction[0] = max(rawAction[0], minMullSsp);
            return rawAction;


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
        ppdCal_base = self._getPPDAMV(amvFromBAS);
        # Send to BAS
        if self._isMullSspLowerLimit:
            action = self._processActions(action, self._oat, self._iat, self._ocp);
        actionUnitChanged = self._getPrcdUnitChangedActions(action);
        is_terminal = False;
        integral_energy_list = [];
        wt_s = socket.socket(); # Write action socket
        wt_s.connect((self._site_server_ip, self._wt_port));
        toSend = ['write', len(actionUnitChanged)];
        toSend.extend(actionUnitChanged);
        wt_s.sendall(bytearray(str(toSend), encoding = 'utf-8')) # Send to write server [codeString, #ofData, data1, data2, ...]
        self.logger_main.info('STEP base AMV-PPD before step is: %s, action sent to BAS is: %s'%(ppdCal_base, toSend[2:]))
        wt_s_recv = wt_s.recv(4096).decode(encoding = 'utf-8'); # Receive is [flag, msg]
        wt_s_recv = ast.literal_eval(wt_s_recv);
        if wt_s_recv[0] == 0:
            self.logger_main.warning('Write to BAS has error: %s'%wt_s_recv[1]);
        else:
            self.logger_main.info('STEP Write to BAS success!');
        # Read from BAS
        # Wait for control step finishes, except if IW PPD increases
        controlStepLenS = self._ctrl_step_size_s * self._act_repeat;
        baseTime = time.time();
        self.logger_main.info('STEP Wait for %d seconds to finish this control step'%(controlStepLenS));
        loopCount = 0
        queryFreq = 10 #Seconds
        energyQueryFreq = 10 #Seconds
        energyQueryWait = 120 #Seconds
        energy_now = None;
        while (time.time() - baseTime) < controlStepLenS:
            # Check ppd
            amvFromBAS = self._readFromBASHelper('getAMV', False);
            ppdCal_now = self._getPPDAMV(amvFromBAS);
            if ppdCal_now > ppdCal_base:
                # Jump out of the while and finish this step
                self.logger_main.info('STEP IW AMV profile changed to: %s, now PPD-AMV is %s (old PPD-AMV is %s), jump out of the step loop'
                                      %(amvFromBAS, ppdCal_now, ppdCal_base));
                break;
            # Check energy frequently because we want the average energy demand of the last control step
            # But check until after 2 min to let energy demand stable
            if loopCount != 0 and loopCount%(energyQueryFreq/queryFreq) == 0 and loopCount > energyQueryWait/queryFreq:
                energyFromBAS = self._readFromBASHelper('getenergy', True)
                energy_now = energyFromBAS[0]; # in kW
                integral_energy_list.append(energy_now);
            self.logger_main.info('STEP Remaining time to finish this step: %ds, PPD-AMV now is: %s, energy demand is: %s'
                                    %(controlStepLenS - (time.time() - baseTime), ppdCal_now, energy_now))
            time.sleep(queryFreq) # Check PPD every 5 seconds
            loopCount += 1;
        # Do the reading from BAS
        nowDatetime = datetime.now()
        readData = self._readFromBASHelper('getAll');
        # readData: OAT-F, OAH-%, WS-MHP, WD, HWOEN-F, MULSSP-F, IAT-F, IATSSP-F, HTDMD-KW, 17 Values for AMV
        (readDataUnitChanged, solDif, solDir, ppdCal, occpMode) = self._processRawReadDataAndQueryPi(readData, nowDatetime, self._isPPDBk);
        integral_energy_list.append(readDataUnitChanged[8]);
        # State ob: OAT-TOC, OAH, WS-TOMS, WD, SOLDIF-CAL, SOLDIR-CAL, HWOEN-TOC, PPD-CAL, MULSSP-TOC, IAT-TOC, IATSSP-TOC, OCCP-CAL, HTDMD-TOKWH
        prcdStateOb = [readDataUnitChanged[0], readDataUnitChanged[1], readDataUnitChanged[2], readDataUnitChanged[3], solDif, solDir,\
                       readDataUnitChanged[4], ppdCal, readDataUnitChanged[5], readDataUnitChanged[6], readDataUnitChanged[7], occpMode, \
                       sum(integral_energy_list)/len(integral_energy_list)];
        # Remember last step oa
        self._oat = readDataUnitChanged[0];
        self._iat = readDataUnitChanged[6];
        self._ocp = occpMode;
        curObTim = getSecondFromStartOfYear(nowDatetime);
        self.logger_main.info('STEP State observation is: %s, current time in sceonds since the start of the year is: %s'%(prcdStateOb, curObTim));
        ret.append(curObTim);
        ret.append(prcdStateOb);
        # Read the weather forecast
        if self._incl_forecast:
            pass;
        # Add terminal status
        ret.append(is_terminal);
        # Log observations to file
        toWriteLog = self._getToWriteLog(nowDatetime, prcdStateOb, readData[9:]);
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
    def _getPPDFangerBK(self, allAMV, occp, nowDatetime):
        """
            ret: float
                0-100
        """
        amvScale = 7;
        amvBase = 7//2
        # 17 AMV in total, 0 to 6 scale, 0 is cold, 6 is hot, 3 is neutral
        allAMVScale = [(amv - amvBase) for amv in allAMV];
        # Fill the fanger PMV for those AMV 0 
        # IARH and IAT is needed for fanger PPD
        met = self._met #met
        airVel = self._airVel #m/s
        clo = self._clo #clo
        mrtAdj = 1.0 #C
        if occp == 0 or nowDatetime.weekday() >= 5: # Non-occupied mode and weekends
            met = 1.2;
            clo = 1.0;
            airVel = 0.1;
            mrtAdj = 0.0; #C
        iarhFromBAS = self._readFromBASHelper('getIARH', False)[0]; 
        iatFromBAS = self._readFromBASHelper('getIAT', False)[0];# F
        iatFromBAS = (iatFromBAS - 32)/1.8
        pmvFanger, ppdFanger = fangerPMV(iatFromBAS, iatFromBAS - mrtAdj, iarhFromBAS, airVel, met, clo);
        changedAmvIdxList = [];
        for i in range(len(allAMVScale)):
            if allAMVScale[i] == 0: 
                allAMVScale[i] = pmvFanger;
                changedAmvIdxList.append(i);
        onePPD = 100 * sum(map(abs, allAMVScale))/(amvBase * len(allAMVScale))
        if len(changedAmvIdxList)>0:
            self.logger_main.warning('Collected AMV for idx: %s is 0.0, so Fanger-PMV: %s is used, resulting PPD is: %s'%(changedAmvIdxList, pmvFanger, onePPD))
        return onePPD;

    def _getPPDAMV(self, allAMV):
        """
            ret: float
                0-100
        """
        amvScale = 7;
        amvBase = 7//2
        # 17 AMV in total, 0 to 6 scale, 0 is cold, 6 is hot, 3 is neutral
        allAMVScale = [(amv - amvBase) for amv in allAMV];
        onePPD = 100 * sum(map(abs, allAMVScale))/(amvBase * len(allAMVScale))
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


    def _getWeatherFromCSL(self):
        class HostNameIgnoringAdapter(HTTPAdapter):
            def init_poolmanager(self, connections, maxsize, block=False):
                self.poolmanager = PoolManager(num_pools=connections,
                                       maxsize=maxsize,
                                       block=block,
                                       assert_hostname=False)
        piIds = ['P0-MYhSMORGkyGTe9bdohw0AWysAAAV0lOLTYyTlBVMkJWTDIwXFBISVBQU19XRUFUSEVSX09BVC5QUkVTRU5UX1ZBTFVF',
                 'P0-MYhSMORGkyGTe9bdohw0AXCsAAAV0lOLTYyTlBVMkJWTDIwXFBISVBQU19XRUFUSEVSX09BSC5QUkVTRU5UX1ZBTFVF',
                 'P0-MYhSMORGkyGTe9bdohw0AVysAAAV0lOLTYyTlBVMkJWTDIwXFBISVBQU19XRUFUSEVSX1dJTkRfU1BFRURfQVYuUFJFU0VOVF9WQUxVRQ',
                 'P0-MYhSMORGkyGTe9bdohw0AUysAAAV0lOLTYyTlBVMkJWTDIwXFBISVBQU19XRUFUSEVSX1dJTkRfRElSRUNUSU9OX0FWLlBSRVNFTlRfVkFMVUU']
        ret = [];
        msgBacks = [];
        for piid in piIds:
            s = requests.Session() 
            s.mount('https://', HostNameIgnoringAdapter())
            piUrl = 'https://128.2.109.159/piwebapi/streams/%s/value'%piid 
            r = s.get(piUrl, auth=('CMU_Students', 'WorkHard!ChangeWorld'));
            r = json.loads(r.text);
            if r['Good']:
                msgBack = 'Query success!'
                value = r['Value']
            else:
                msgBack = r;
                value = None;
            ret.append(value);
            msgBacks.append(msgBack)
        return (ret, msgBacks)
    
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
        # rawData: OAT-F, OAH-%, WS-MHP, WD, HWOEN-F, MULSSP-F, IAT-F, IATSSP-F, HTDMD-KW, 17 Values for AMV
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
        for i in range(len(rawData[9:])):
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

    
