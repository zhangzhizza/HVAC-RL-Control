#Wrap the EnergyPLus simulator into the Openai gym env
import socket              
import os
import _thread
import logging

from shutil import copyfile
from gym import Env, spaces
from gym.envs.registration import register
from xml.etree.ElementTree import Element, SubElement, Comment, tostring


logging.basicConfig(format='[%(asctime)s] %(levelname)s:%(message)s'
                    , level=logging.INFO);

class EplusEnv(Env):
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

    Attributes
    ----------
    """

    def __init__(self, eplus_path, 
                 weather_path, bcvtb_path, 
                 variable_path, idf_path):
        
        # Set the environment variable for bcvtb
        os.environ['BCVTB_HOME'] = bcvtb_path;
        
        # Create a socket for communication with eplus_path
        logging.info('Creating socket for communication...')
        s = socket.socket()
        host = socket.gethostname() # Get local machine name
        s.bind((host, 0))           # Bind to the host and any available port
        sockname = s.getsockname();
        port = sockname[1];         # Get the port number
        s.listen(60)                # Listen on request
        logging.info('Socket is listening on ', sockname);
        
        # Create EnergyPlus simulaton process
        logging.info('Creating EnergyPlus simulation environment...')
        eplus_working_dir = self._get_eplus_working_folder('.');
                                    # Create the Eplus working directory
        eplus_working_idf_path = (eplus_working_dir + 
                                  '/' + 
                                  self._get_file_name(idf_path));
        eplus_working_var_path = (eplus_working_dir + 
                                  '/' + 
                                  self._get_file_name(variable_path));
        copyfile(idf_path, eplus_working_idf_path);
                                    # Copy the idf file to the working directory
        copyfile(variable_path, eplus_working_var_path);
                                    # Copy the variable.cfg file to the working dir
        self._create_socket_cfg(host, 
                                port,
                                eplus_working_dir); 
                                    # Create the socket.cfg file in the working dir
        logging.info('EnergyPlus working directory is in ', eplus_output_dir);
        _thread.start_new_thread(create_eplus, 
                                 (weather_path, eplus_working_idf_path));
        
        #Establish connection with EnergyPlus
        c, addr = s.accept()     # Establish connection with client.
        logging.info('Got connection from %s at port %d.'%addr);
        
    def _reset(self):
        """Reset the environment.

        The server should always start on Queue 1.

        Returns
        -------
        (int, int, int, int)
          A tuple representing the current state with meanings
          (current queue, num items in 1, num items in 2, num items in
          3).
        """

    def _step(self, action):
        """Execute the specified action.

        Parameters
        ----------
        action: int
          A number in range [0, 3]. Represents the action.

        Returns
        -------
        (state, reward, is_terminal, debug_info)
          State is the tuple in the same format as the reset
          method. Reward is a floating point number. is_terminal is a
          boolean representing if the new state is a terminal
          state. debug_info is a dictionary. You can fill debug_info
          with any additional information you deem useful.
        """


    def _render(self, mode='human', close=False):
        pass;
    
    def _create_eplus(self, weather_path, idf_path):
        subprocess.call('../EnergyPlus-8-6-0/energyplus -w %s -d %s -r %s'
                        %(weather_path, 'output', idf_path));
    
    def _get_eplus_working_folder(self, parent_dir):
        """Return Eplus output folder. Author: CMU-10703 Spring 2017 TA

        Assumes folders in the parent_dir have suffix -run{run
        number}. Finds the highest run number and sets the output folder
        to that number + 1. 

        Parameters
        ----------
        parent_dir: str
        Parent dir of the Eplus output directory.

        Returns
        -------
        parent_dir/run_dir
        Path to Eplus save directory.
        """
        os.makedirs(parent_dir, exist_ok=True)
        experiment_id = 0
        for folder_name in os.listdir(parent_dir):
            if not os.path.isdir(os.path.join(parent_dir, folder_name)):
                continue
            try:
                folder_name = int(folder_name.split('-run')[-1])
                if folder_name > experiment_id:
                    experiment_id = folder_name
            except:
                pass
        experiment_id += 1

        parent_dir = os.path.join(parent_dir, env_name)
        parent_dir = parent_dir + '-run{}'.format(experiment_id)
        return parent_dir

    def _create_socket_cfg(self, host, port, write_dir):
        top = Element('BCVTB-client');
        ipc = SubElement(top, 'ipc');
        socket = SubElement(ipc, 'socket'
                            ,{'port':str(port),
                              'hostname':host,})
        xml_str = tostring(top, encoding='ISO-8859-1'
                         , xml_declaration=True
                         , pretty_print=True);
        with open(write_dir + 'socket.cfg', 'w+') as socket_file:
            socket_file.write(xml_str);
        
    
    def _get_file_name(self, file_path):
        path_list = file_path.split;
        return path_list[-1];

register(
    id='Eplus_test',
    entry_point='env_eplus8-6.env_eplus:EplusEnv',
    kwargs={'eplus_path':'EnergyPlus-8-6-0/energyplus',
            'weather_path':'Pittsburgh.epw',
            'bcvtb_path':'bcvtb/',
            'variable_path':'variables.cfd',
            'idf_path':'5ZoneAutoDXVAV.idf'});




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
    
"""

while True:   
   rcv = c.recv(1024).decode();
   logging.info('Got message: ' + rcv);
   version, flag, nDb, nIn, nBl, curSimTim, Dblist = disassembleMsg(rcv);
   tosend = assembleMsg(version, flag, 0, nIn, nBl, curSimTim, []);
   c.send(tosend.encode());

"""