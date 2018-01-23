import socket
import subprocess
import copy
import xml.etree.ElementTree as ET

from util.logger import Logger
from SDCServer.bacnet.bacenum import bacenumMap

LOG_LEVEL = 'DEBUG';
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";     

def runReadServer(port, readConfig, bacrpmPath):
	# Set the logger
	logger_main = Logger().getLogger('SDC_ReadServer', LOG_LEVEL, LOG_FMT);
	# Read config file
	configMap = readXmlConfg(readConfig);
	rpCmds = assembleBacrpCmd(configMap);
	for rpCmd in rpCmds:
		rpCmd.insert(0, bacrpmPath);
	logger_main.debug('rpCmds:' + str(rpCmds))
	# Create the socket
	s = socket.socket();
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
			for rpCmd in rpCmds:
				rawResult = subprocess.run(rpCmd, stdout=subprocess.PIPE).stdout.decode();
				logger_main.debug('Raw return from BACnet stack is ' + rawResult);
				prcdResult = processRawBacrpmRet(rawResult);
				bacrpmRes.extend(prcdResult);
			bacrpmRes.insert(0, len(bacrpmRes));
			bacrpmRes.insert(0, 1); # Flag, 1 is ok
			c.sendall(bytearray(str(bacrpmRes), encoding = 'utf-8'));
		else:		
			c.sendall(bytearray(str([0, 0]), encoding = 'utf-8'))

def readXmlConfg(readConfig):
	deviceInstanceDict = {};
	tree = ET.parse(readConfig)
	root = tree.getroot()
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
	return deviceInstanceDict;

def assembleBacrpCmd(configMap):
	cmds = [];
	for key in configMap.keys():
		deviceRpRequests = copy.deepcopy(configMap[key]);
		deviceRpRequests.insert(0, key);
		cmds.append(deviceRpRequests);
	return cmds;

def processRawBacrpmRet(rawResult):
	prcdRes = []
	rawResult = rawResult.replace('\n', '').replace('\r', '').replace(' ','');
	colonIdx = rawResult.find(':');
	rightBraceIdx = rawResult.find('}');
	while colonIdx != -1:
		prcdRes.append(rawResult[colonIdx + 1:rightBraceIdx]);
		colonIdx = rawResult.find(':', colonIdx + 1);
		rightBraceIdx = rawResult.find('}', rightBraceIdx + 1);

	return prcdRes;



		








