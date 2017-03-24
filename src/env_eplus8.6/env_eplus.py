import socket               # Import socket module
import os
import _thread
import logging

logging.getLogger().setLevel(logging.INFO);

def create_eplus():
    os.system('../EnergyPlus-8-6-0/energyplus -w pittsburgh.epw -d output -r 5ZoneAutoDXVAV.idf');
    
def disassembleMsg(rcv):
    rcv = rcv.split(' ');
    version = int(rcv[0]);
    flag = int(rcv[1]);
    nDb = int(rcv[2]);
    nIn = int(rcv[3]);
    nBl = int(rcv[4]);
    curSimTim = float(rcv[5]);
    Dblist = [];
    for i in range(6, len(rcv) - 1):
        Dblist.append(float(rcv[i]));
    return (version, flag, nDb, nIn, nBl, curSimTim, Dblist);
    
def assembleMsg(version, flag, nDb, nIn, nBl, curSimTim, Dblist):
    ret = '';
    ret += '%d'%(version);
    ret += ' ';
    ret += '%d'%(flag);
    ret += ' ';
    ret += '%d'%(nDb);
    ret += ' ';
    ret += '%d'%(nIn);
    ret += ' ';
    ret += '%d'%(nBl);
    ret += ' ';
    ret += '%20.15e'%(curSimTim);
    ret += ' ';
    for i in range(len(Dblist)):
        ret += '%20.15e'%(Dblist[i]);
        ret += ' ';
    ret += '\n';
    return ret;
    
    
s = socket.socket()         # Create a socket object
host = socket.gethostname() # Get local machine name
port = 59215                # Reserve a port for your service.
s.bind((host, port))        # Bind to the port
s.listen(60)                 # Now wait for client connection.
logging.info('Creating EnergyPlus simulation environment...')
_thread.start_new_thread(create_eplus, ());
c, addr = s.accept()     # Establish connection with client.
logging.info('Got connection from %s at port %d'%addr);

while True:   
   rcv = c.recv(1024).decode();
   logging.info('Got message: ' + rcv);
   version, flag, nDb, nIn, nBl, curSimTim, Dblist = disassembleMsg(rcv);
   tosend = assembleMsg(version, flag, 0, nIn, nBl, curSimTim, []);
   c.send(tosend.encode());

