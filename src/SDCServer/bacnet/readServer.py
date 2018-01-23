import socket
import xml.etree.ElementTree as ET

from ....util.logger import Logger

LOG_LEVEL = 'INFO';
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";
ACTION_SIZE = 2 * 5;           

def runReadServer(port, readConfig):
	# Set the logger
	logger_main = Logger().getLogger('SDC_ReadServer', LOG_LEVEL, LOG_FMT);
	# Read config file
	configMap = readXmlConfg(readConfig);
	# Create the socket
	s = socket.socket();
	s.bind(('0.0.0.0', port))        
	s.listen(5)
	logger_main.info('Server starts at ' + (':'.join(str(e) for e in s.getsockname())))                 
	while True:
		c, addr = s.accept()    
		logger_main.info('Got connection from', addr)
		recv = c.recv(1024).decode(encoding = 'utf-8')
		if recv.lower() == 'get':
			logger_main.info('Recived GET request from', addr)
			c.sendall(bytearray(str([1,2,3,4]), encoding = 'utf-8'));
		else:		
			c.sendall(b'I don not understand')

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
			deviceInstanceDict[deviceId].append(bacenumMap[instanceTypens]);
			deviceInstanceDict[deviceId].append(instanceId);
			propertyName = instanceLevel[0].attrib['Name'];
			deviceInstanceDict[deviceId].append(propertyName);
	return deviceInstanceDict;




