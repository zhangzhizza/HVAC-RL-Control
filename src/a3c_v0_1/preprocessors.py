import random
import datetime
import numpy as np

from a3c_v0_1.state_index import *
from util.time import get_time_from_seconds
from numpy import linalg as LA

def process_raw_state_1(simTime, state, start_year, start_mon, 
                          start_day, start_weekday):
    """
    Raw state processing 1. Do the following things:
    1. Insert day of the week and hour of the day to start of the state list;
    
    Args:
        simTime: python list, 1-D 
            Delta seconds from the start time, each item in the list
            corresponds to a row in the state. 
        state: python list, 1-D or 2-D
            The raw observation from the environment. It can be only one 
            sample (1-D) or multiple samples (2-D, where each row is a sample).
        start_year: int 
            Start year.
        start_mon: int 
            Start month.
        start_day: int.
            The day of the month at the start time. 
        start_weekday: int 
            The start weekday. 0 is Monday and 6 is Sunday.
    Return: python list, 1-D or 2-D
        Deepcopy of the processed state. It can be only one sample (1-D) or 
        multiple samples (2-D, where each row is a sample).
    """
    ret = [];
    state = np.array(state);
    # Reshape the state to 2-D if it is 1-D
    if len(state.shape) == 1:
        state = state.reshape(1, -1);
    # Manipulate the state
    for i in range(state.shape[0]):
        state_i_list = state[i,:].tolist();
        nowWeekday, nowHour = get_time_from_seconds(simTime[i], start_year, 
                                                    start_mon, start_day, 
                                                    start_weekday);
        nowWeekday = 1 if nowWeekday <= 4 else 0;
        state_i_list.insert(0, nowHour);     # Add weekday and hour infomation
        state_i_list.insert(0, nowWeekday);
        #state_i_list[ZPCT_RAW_IDX + 2] = 1 if state_i_list[ZPCT_RAW_IDX + 2] > 0 \
        #                                   else 0; # Occupancy count --> occupancy
        if state.shape[0] > 1:
            ret.append(state_i_list);
        else:
            ret = state_i_list;
    return ret;

def process_raw_state_2(state_prcd_1, min_max_limits):
    """
    Raw state processing 2. Do the following things:
    1. Standarderlize the state using the min max normalization;
    
    Args:
        state_prcd_1: python list, 1-D or 2-D
            The processed state by process_raw_state_1. It can be only one 
            sample (1-D) or multiple samples (2-D, where each row is a sample)
        min_max_limits: python list, 2-D, 2*m where m is the number of state 
                        features
            The minimum and maximum possible values for each state feature. 
            The first row is the minimum values and the second row is the maximum
            values.
            
    Return: python list, 1-D or 2-D
        Min max normalized state. It can be only one sample (1-D) or multiple 
        samples (2-D, where each row is a sample).
    """
    state_prcd_1 = np.array(state_prcd_1);
    min_max_limits = np.array(min_max_limits);
    # Do min-max normalization
    std_state = (state_prcd_1 - min_max_limits[0,:])/(min_max_limits[1,:] -
                                                      min_max_limits[0,:]);
    return std_state.tolist()
    
def process_raw_state_cmbd(raw_state, simTime, start_year, start_mon, 
                          start_date, start_day, min_max_limits):
    """
    Process the raw state by calling process_raw_state_1 and process_raw_state_2
    in order.
    
    Args:
        raw_state: python list, 1-D or 2-D
            The raw observation from the environment. It can be only one 
            sample (1-D) or multiple samples (2-D, where each row is a sample).
        simTime: python list, 1-D 
            Delta seconds from the start time, each item in the list
            corresponds to a row in the state. 
        start_year: int 
            Start year.
        start_mon: int 
            Start month.
        start_date: int.
            The day of the month at the start time. 
        start_day: int 
            The start weekday. 0 is Monday and 6 is Sunday.
        min_max_limits: python list, 2-D, 2*m where m is the number of state 
                        features
            The minimum and maximum possible values for each state feature. 
            The first row is the minimum values and the second row is the maximum
            values.
    
    Return: python list, 1-D or 2-D
        Processed min-max normalized (0 to 1) state It can be only one sample 
        (1-D) or multiple samples (2-D, where each row is a sample).
        State feature order:
        
    """
    state_after_1 = process_raw_state_1(simTime, raw_state, start_year, start_mon, 
                                        start_date, start_day);
    state_after_2 = process_raw_state_2(state_after_1, min_max_limits)
    
    return state_after_2;
    
class HistoryPreprocessor:
    """Keeps the last k states.

    Useful for seeing the trend of the change, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Args:
        history_length: int
            Number of previous states to prepend to state being processed.

    """

    def __init__(self, history_length, forecast_dim):
        self._history_length = history_length;
        self._flag_start_net = True;
        self._stacked_return_net = None;
        self._forecast_dim = forecast_dim;

    def process_state_for_network(self, state):
        """Take the current state and return a stacked state with current
        state and the history states. 
        
        Args:
            state: python 1-D list.
                Expect python 1-D list representing the current state.
        
        Return: np.ndarray, dim = 1*m where m is the state_dim * history_length
            Stacked states.
        """
        forecast_state = None;
        if self._forecast_dim > 0:
            ob_state = state[0: len(state) - self._forecast_dim] # Delete the forecast states
            forecast_state = state[-self._forecast_dim: ] # Get the forecast state
            state = ob_state;
        state = np.array(state).reshape(1, -1);
        state_dim = state.shape[-1];
        if self._flag_start_net:
            self._stacked_return_net = np.zeros((self._history_length, state_dim));
            self._stacked_return_net[-1,:] = state;
            self._flag_start_net = False;
        else:
            for i in range(self._history_length - 1):
                self._stacked_return_net[i, :] = \
                    self._stacked_return_net[i+1, :];
            self._stacked_return_net[-1, :] = state;

        ret = np.copy(self._stacked_return_net.flatten().reshape(1, -1));
        if self._forecast_dim > 0:
            ret = np.append(ret, np.array(forecast_state).reshape(1, -1), 1);
        return ret;


    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self._flag_start_net = True;

    def get_config(self):
        return {'history_length': self.history_length}
 