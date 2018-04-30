
import argparse

from SDCServer.bacnet.writeServer import WriteServer;
from SDCServer.bacnet.readServer import ReadServer;

import threading

def main():
	parser = argparse.ArgumentParser(description='Run SDC Server')
	parser.add_argument('--read_port', default=61221, type=int, help='Read server listening port');
	parser.add_argument('--write_port', default=61222, type=int, help='Write server listening port');
	parser.add_argument('--read_config', default='SDCServer/bacnet/IW9701_ReadConfig.cfg', type=str, help='Read server configuration file');
	parser.add_argument('--write_config', default='SDCServer/bacnet/IW9701_WriteConfig.cfg', type=str, help='Write server configuration file')
	args = parser.parse_args();
	# Run
	writeServerIns = WriteServer();
	readServerIns = ReadServer();
	runWriteServer = lambda: writeServerIns.runWriteServer(args.write_port, args.write_config);
	writeThread = threading.Thread(target = (runWriteServer));
	writeThread.start();
	runReadServer = lambda: readServerIns.runReadServer(args.read_port, args.read_config);
	readThread = threading.Thread(target = (runReadServer));
	readThread.start();

if __name__ == '__main__':
    main()

# Write server config
port_wt = 61222;
writeConfig = 'SDCServer/bacnet/IW9701_WriteConfig.cfg'
bacwpPath = './SDCServer/bacnet/bacnet-stack-0.8.5/bin/bacwp'

# Read server config
port_rd = 61221;
readConfig = 'SDCServer/bacnet/IW9701_ReadConfig.cfg'
bacrpmPath = './SDCServer/bacnet/bacnet-stack-0.8.5/bin/bacrpm'

# Run
writeServerIns = WriteServer();
readServerIns = ReadServer();
runWriteServer = lambda: writeServerIns.runWriteServer(port_wt, writeConfig, bacwpPath);
writeThread = threading.Thread(target = (runWriteServer));
writeThread.start();
runReadServer = lambda: readServerIns.runReadServer(port_rd, readConfig, bacrpmPath);
readThread = threading.Thread(target = (runReadServer));
readThread.start();

