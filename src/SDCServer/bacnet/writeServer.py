import socket
import subprocess
import copy
import ast
import os
import time
import xml.etree.ElementTree as ET

from util.logger import Logger
from SDCServer.bacnet.bacenum import bacenumMap

FD = os.path.dirname(os.path.realpath(__file__));
LOG_LEVEL = 'DEBUG';
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";     

class WriteServer(object):

	def runWriteServer(self, port, writeConfig, bacwpPath):
		# Set the logger
		logger_main = Logger().getLogger('SDC_WriteServer', LOG_LEVEL, LOG_FMT, log_file_path = '%s/log/writeServer_%s.log'%(FD, time.time()));
		# Read config file
		configMap = self.readXmlConfg(writeConfig);
		wpCmds = self.assembleBacwpCmd(configMap);
		for wpCmd in wpCmds:
			wpCmd.insert(0, bacwpPath);
		logger_main.debug('wpCmds:' + str(wpCmds))
		# Create the socket
		s = socket.socket();
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.bind(('0.0.0.0', port))        
		s.listen(5)
		self._s = s;
		logger_main.info('Server starts at ' + (':'.join(str(e) for e in s.getsockname())))                 
		while True:
			c, addr = s.accept()
			addr = (':'.join(str(e) for e in addr));   
			logger_main.info('Got connection from ' + addr);
			recv = c.recv(2048).decode(encoding = 'utf-8') # Recieve a list: [codeString, #ofData, data1, data2, ...]
			recv = ast.literal_eval(recv);
			code = recv[0];
			dataCount = recv[1];
			toWriteData = recv[2:];
			if code == 'write':
				logger_main.info('Recived WRITE request from %s, number of data to write is %d, data to write are %s'%(addr, dataCount, str(toWriteData)))
				# Perform Bacwp
				sendBack = []; # Send back [flag, msg], flag = 1 is ok
				if dataCount != len(wpCmds):
					errMsg = "Number of data to write should be %d, but gets %d, so bacwp is not performed"%(len(wpCmds), dataCount);
					logger_main.warning(errMsg);
					sendBack.append(0);
					sendBack.append(errMsg);
				else:
					errMsg = 'Errors encountered when perform bacwp:';
					noErrLen = len(errMsg)
					for cmd_i in range(len(wpCmds)):
						wpCmdThisTime = copy.deepcopy(wpCmds[cmd_i]);
						wpCmdThisTime.append(str(toWriteData[cmd_i]));
						logger_main.debug('Command to write through BACnet stack: %s'%wpCmdThisTime)
						writeResponse = subprocess.run(wpCmdThisTime, stdout=subprocess.PIPE).stdout.decode();
						logger_main.debug('Raw response from the BACnet stack: %s'%writeResponse);
						writeResponse = writeResponse.strip();
						if "WriteProperty Acknowledged" not in writeResponse:
							errMsg += ' For No.%d bacwp, BACnet server responded \'%s\''%(cmd_i, writeResponse);
					if len(errMsg) == noErrLen:
						sucMsg = 'Write success!';
						sendBack.append(1); # All writes success!
						sendBack.append(sucMsg);
						logger_main.info(sucMsg);
					else:
						sendBack.append(0);
						sendBack.append(errMsg);
						logger_main.warning(errMsg);
				c.sendall(bytearray(str(sendBack), encoding = 'utf-8'));
			else:		
				c.sendall(bytearray(str([0]), encoding = 'utf-8'))

	def closeServer(self):
		self._s.close();

	def readXmlConfg(self, writeConfig):
		deviceInstanceDict = {};
		tree = ET.parse(writeConfig)
		root = tree.getroot()
		# Device level info
		for deviceLevel in root:
			deviceId = deviceLevel.attrib['Instance']
			deviceInstanceDict[deviceId] = [];
			# Instance level info
			for instanceLevel in deviceLevel:
				thisInstanceInfo = [];
				instanceType = instanceLevel.attrib['Type'];
				instanceId = instanceLevel.attrib['Instance'];
				thisInstanceInfo.append(bacenumMap[instanceType]);
				thisInstanceInfo.append(instanceId);
				# Instance's child level, only one child
				propertyName = instanceLevel[0].attrib['Name'];
				priority = instanceLevel[0].attrib['Priority'];
				tag = instanceLevel[0].attrib['ApplicationTag'];
				index = instanceLevel[0].attrib['Index'];
				thisInstanceInfo.append(bacenumMap[propertyName]);
				thisInstanceInfo.append(priority);
				thisInstanceInfo.append(index);
				thisInstanceInfo.append(bacenumMap[tag]);
				deviceInstanceDict[deviceId].append(thisInstanceInfo);
		return deviceInstanceDict;

	def assembleBacwpCmd(self, configMap):
		cmds = [];
		for key in configMap.keys():
			deviceWpRequests = copy.deepcopy(configMap[key]);
			for deviceWpRequestEach in deviceWpRequests:
				deviceWpRequestEach.insert(0, key);
				cmds.append(deviceWpRequestEach);
		return cmds;


		








