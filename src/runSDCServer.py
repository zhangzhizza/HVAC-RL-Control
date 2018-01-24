from SDCServer.bacnet.writeServer import WriteServer;
from SDCServer.bacnet.readServer import ReadServer;

import threading

# Write server config
port_wt = 61222;
writeConfig = 'SDCServer/bacnet/testWriteConfig.cfg'
bacwpPath = './SDCServer/bacnet/bacnet-stack-0.8.5/bin/bacwp'

# Read server config
port_rd = 61221;
readConfig = 'SDCServer/bacnet/testReadConfig.cfg'
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