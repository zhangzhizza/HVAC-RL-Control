import socket
import subprocess
import copy
import time
import os
import xml.etree.ElementTree as ET

from util.logger import Logger
from SDCServer.bacnet.bacenum import bacenumMap

FD = os.path.dirname(os.path.realpath(__file__));
LOG_LEVEL = 'DEBUG';
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";     

class ReadServer(object):

	def runReadServer(self, port, readConfig, bacrpmPath):
		# Set the logger
		logger_main = Logger().getLogger('SDC_ReadServer', LOG_LEVEL, LOG_FMT, log_file_path = '%s/log/readServer_%s.log'%(FD, time.time()));
		# Read config file
		(configMap, readCount) = self.readXmlConfg(readConfig);
		rpCmds = self.assembleBacrpCmd(configMap);
		for rpCmd in rpCmds:
			rpCmd.insert(0, bacrpmPath);
		logger_main.debug('rpCmds:' + str(rpCmds))
		# Create the socket
		s = socket.socket();
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.bind(('0.0.0.0', port))        
		s.listen(5)
		logger_main.info('Server starts at ' + (':'.join(str(e) for e in s.getsockname())))                 
		while True:
			c, addr = s.accept()
			addr = (':'.join(str(e) for e in addr));   
			logger_main.info('Got connection from ' + addr);
			recv = c.recv(1024).decode(encoding = 'utf-8')
			if recv.lower() == 'get':
				logger_main.info('Recived GET request from ' + addr)
				# Perform Bacrp
				bacrpmRes = [];
				bacnetStackRawRet = [];
				flag = 1;
				retMsg = 'Read success!';
				for rpCmd in rpCmds:
					rawResult = subprocess.run(rpCmd, stdout=subprocess.PIPE).stdout.decode();
					bacnetStackRawRet.append(rawResult);
					logger_main.debug('Raw return from BACnet stack is ' + rawResult);
					prcdResult = self.processRawBacrpmRet(rawResult);
					bacrpmRes.extend(prcdResult);
				if len(bacrpmRes) != readCount:
					retMsg = 'Should read %d values, but gets %d values, the raw output from the BACnet stack is: %s'%(readCount, len(bacrpmRes));
					logger_main.warning(retMsg);
					logger_main.warning('The BACnet stack raw return is: %s'%bacnetStackRawRet);
					flag = 0;	
				else:
					logger_main.info(retMsg);
				bacrpmRes.insert(0, len(bacrpmRes));
				bacrpmRes.insert(0, retMsg);
				bacrpmRes.insert(0, flag);
				c.sendall(bytearray(str(bacrpmRes), encoding = 'utf-8'));
			else:		
				c.sendall(bytearray(str([0, 0]), encoding = 'utf-8'))

	def readXmlConfg(self, readConfig):
		deviceInstanceDict = {};
		tree = ET.parse(readConfig)
		root = tree.getroot()
		readCount = 0;
		for deviceLevel in root:
			deviceId = deviceLevel.attrib['Instance']
			deviceInstanceDict[deviceId] = [];
			for instanceLevel in deviceLevel:
				instanceType = instanceLevel.attrib['Type'];
				instanceId = instanceLevel.attrib['Instance'];
				deviceInstanceDict[deviceId].append(bacenumMap[instanceType]);
				deviceInstanceDict[deviceId].append(instanceId);
				propertyName = instanceLevel[0].attrib['Name'];
				deviceInstanceDict[deviceId].append(bacenumMap[propertyName]);
				readCount += 1;
		return (deviceInstanceDict, readCount);

	def assembleBacrpCmd(self, configMap):
		cmds = [];
		for key in configMap.keys():
			deviceRpRequests = copy.deepcopy(configMap[key]);
			deviceRpRequests.insert(0, key);
			cmds.append(deviceRpRequests);
		return cmds;

	def processRawBacrpmRet(self, rawResult):
		prcdRes = []
		rawResult = rawResult.replace('\n', '').replace('\r', '').replace(' ','');
		colonIdx = rawResult.find(':');
		rightBraceIdx = rawResult.find('}');
		while colonIdx != -1:
			prcdRes.append(rawResult[colonIdx + 1:rightBraceIdx]);
			colonIdx = rawResult.find(':', colonIdx + 1);
			rightBraceIdx = rawResult.find('}', rightBraceIdx + 1);

		return prcdRes;



		








