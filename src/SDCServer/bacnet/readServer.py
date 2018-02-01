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
		(configList, readCountAll) = self.readXmlConfg(readConfig);
		rpCmds = copy.deepcopy(configList);
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
			bacrpmRes = [];
			bacnetStackRawRet = [];
			flag = 1;
			retMsg = 'Read success!';
			toReadCmds = [];
			readCount = 0;
			if recv.lower() == 'getall':
				logger_main.info('Recived GETALL request from ' + addr)
				toReadCmds = rpCmds;
				readCount = readCountAll;	
			elif recv.lower() == 'getamv':
				logger_main.info('Recived GETAMV request from ' + addr)
				toReadCmds = [rpCmds[-1]];
				readCount = (len(toReadCmds[0]) - 2)/3;
			elif recv.lower() == 'getenergy':
				logger_main.info('Recieved GETENERGY request from ' + addr)
				toReadCmds = [rpCmds[3]]
				readCount = 1;
			else:
				logger_main.warning('Recieved unrecognized request from %s: %s'%(addr, recv.lower()));
				toReadCmds = []
				readCount = 0;
				retMsg = 'Unrecognized request'
				flag = 0;
			# Perform read
			for rpCmd in toReadCmds:
				rawResult = subprocess.run(rpCmd, stdout=subprocess.PIPE).stdout.decode();
				bacnetStackRawRet.append(rawResult);
				logger_main.debug('Raw return from BACnet stack is ' + rawResult);
				prcdResult = self.processRawBacrpmRet(rawResult);
				bacrpmRes.extend(prcdResult);
			if len(bacrpmRes) != readCount:
				retMsg = 'Should read %d values, but gets %d values, the raw output from the BACnet stack is: %s'%(readCount, len(bacrpmRes), bacnetStackRawRet);
				logger_main.warning(retMsg);
				logger_main.warning('The BACnet stack raw return is: %s'%bacnetStackRawRet);
				flag = 0;	
			else:
				logger_main.info(retMsg);
			bacrpmRes.insert(0, len(bacrpmRes));
			bacrpmRes.insert(0, retMsg);
			bacrpmRes.insert(0, flag);
			c.sendall(bytearray(str(bacrpmRes), encoding = 'utf-8'));

	def readXmlConfg(self, readConfig):
		deviceInstanceList = [];
		tree = ET.parse(readConfig)
		root = tree.getroot()
		readCount = 0;
		for deviceLevel in root:
			deviceId = deviceLevel.attrib['Instance']
			thisDeviceList = [];
			thisDeviceList.append(deviceId);
			for instanceLevel in deviceLevel:
				instanceType = instanceLevel.attrib['Type'];
				instanceId = instanceLevel.attrib['Instance'];
				thisDeviceList.append(bacenumMap[instanceType]);
				thisDeviceList.append(instanceId);
				propertyName = instanceLevel[0].attrib['Name'];
				thisDeviceList.append(bacenumMap[propertyName]);
				readCount += 1;
			deviceInstanceList.append(thisDeviceList);
		return (deviceInstanceList, readCount);

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



		








