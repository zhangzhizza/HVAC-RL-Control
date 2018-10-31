import socket
import subprocess
import copy
import time
import os
import xml.etree.ElementTree as ET

from util.logger import Logger
from SDCServer.bacnet.bacenum import bacenumMap, bacenumMap_inv

FD = os.path.dirname(os.path.realpath(__file__));
LOG_LEVEL = 'DEBUG';
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";     

class WorkerClient(object):

	def runWorkerClient(self, port):
		# Set the logger
		logger_main = Logger().getLogger('worker_client_logger', LOG_LEVEL, LOG_FMT, 
			log_file_path = '%s/log/%s.log'%(FD, time.time()));
		# Create the socket
		s = socket.socket();
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.bind(('0.0.0.0', port))        
		s.listen(5)
		logger_main.info('Socket starts at ' + (':'.join(str(e) for e in s.getsockname())))                 
		while True:
			c, addr = s.accept()
			addr = (':'.join(str(e) for e in addr));   
			logger_main.info('Got connection from ' + addr);
			recv = c.recv(1024).decode(encoding = 'utf-8')
			if recv.lower() == 'getstatus': 
				logger_main.info('Recived GETALL request from ' + addr)
			# Perform tasks
			# 
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

	def processRawBacrpmRet(self, rawResult, rpCmd):
		"""
			rpCmd: python list of str, ['./bacrpm', 'device id', 'object type', 'instance id', 'property', 'device id', ....]
		"""
		prcdRes = []
		rawResult = rawResult.replace('\n', '').replace('\r', '').replace(' ','');
		rpCmd_idx = 4 # The first property ID
		curSearch_idx = 0;
		while rpCmd_idx < len(rpCmd):
			propertyName = bacenumMap_inv[rpCmd[rpCmd_idx]].lower();
			propertyNameIdx = rawResult.find(propertyName, curSearch_idx);
			rightBraceIdx = rawResult.find('}', propertyNameIdx);
			rpCmd_idx += 3;
			if propertyNameIdx == -1 or rightBraceIdx == -1:
				continue;
			else:
				prcdRes.append(rawResult[propertyNameIdx + len(propertyName) + 1: rightBraceIdx]);
				curSearch_idx = rightBraceIdx;
		return prcdRes;



		








