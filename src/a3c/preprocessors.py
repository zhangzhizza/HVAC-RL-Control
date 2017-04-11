import random
import datetime
import numpy as np

from a3c.state_index import *

#def get_env_states_stats(env, actions, start_year,
                         #start_mon, start_date, start_day):
    #"""
    #Run the env for one episode under the random policy to compute the mean
    #and standard deviation of the states.
    
    #Args:
        #env: gym.env 
            #The gym environment.
        #actions: python list of tuples
            #A python list of tuples, where each item in the list represents one 
            #action choice. 
        #start_year: int 
            #Start year.
        #start_mon: int 
            #Start month.
        #start_date: int.
            #The day of the month at the start time.
        #start_day: int 
            #The start weekday. 0 is Monday and 6 is Sunday.
            
    #Return: (np.ndarray, np.ndarray)
        #Two 1-D numpy array, each has length of the raw state dim + 2 (weekday
        #and hour of the day). The first array is the mean of each state feature,
        #and the second array is the standard deviation of each state feature. 
    
    #"""
    #states_hist = [];
    ## Init env reset
    #time_this, ob_this, is_terminal = env.reset();
    #ob_this = process_raw_state_1([time_this], ob_this, start_year, start_mon, 
                          #start_date, start_day)
    
    #states_hist.append(ob_this);
    ## Interact with the env until terminal
    #while not is_terminal:
        #actIdx = random.randint(0, len(actions) - 1);
        #actRandom = actions[actIdx];
        #htStpt_this = ob_this[8 + 2] # Heating stpt
        #clStpt_this = ob_this[9 + 2] # Cooling stpt
        #resStpt, effeAct = get_legal_action(htStpt_this, clStpt_this, 
                                            #actRandom, (15, 30));
        #time_next, ob_next, is_terminal = env.step([resStpt[0], resStpt[1]]);
        #ob_next = process_raw_state_1([time_next], ob_next, start_year, start_mon, 
                          #start_date, start_day);
        #states_hist.append(ob_next);
        #ob_this = ob_next;
    ## Calculate the statistics
    #sample_mean = np.mean(np.array(states_hist), axis = 0);
    #sample_stdv = np.std(np.array(states_hist), axis = 0);
    
    #return (sample_mean, sample_stdv);
        

def get_time_from_seconds(second, start_year, start_mon, 
                          start_date, start_day):
    """
    Get the day of the week and hour of the day given the delta seconds and
    the start time.
    
    Args:
        second: int
            Delta seconds from the start time.
        start_year: int 
            Start year.
        start_mon: int 
            Start month.
        start_date: int.
            The day of the month at the start time.
        start_day: int 
            The start weekday. 0 is Monday and 6 is Sunday.
    
    Return: (int, int)
        The hour of the day and the weekday of now (hour of the day ranges from
        0 to 23, and weekday ranges from 0 to 6 where 0 is Monday).
    """
    startTime = datetime.datetime(start_year, start_mon, start_date, 0, 0, 0);
    nowTime = startTime + datetime.timedelta(0, second);
    delta_days = (nowTime - startTime).days;
    nowWeekday = (start_day + delta_days) % 7;
    nowHour = nowTime.hour;
    return (nowWeekday, nowHour);

def process_raw_state_1(simTime, state, start_year, start_mon, 
                          start_day, start_weekday):
    """
    Raw state processing 1. Do the following things:
    1. Insert day of the week and hour of the day to start of the state list;
    2. Change the occupant count to the occupancy status (0 or 1).
    
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
        State feature order:
            0. Day of the week (0-6, 0 is Monday)
            1. Hour of the day (0-23)
            2. Site Outdoor Air Drybulb Temperature (C)
            3. Site Outdoor Air Relative Humidity (%)
            4. Site Wind Speed (m/s)
            5. Site Wind Direction (degree from north)
            6. Site Diffuse Solar Radiation Rate per Area (W/m2)
            7. Site Direct Solar Radiation Rate per Area (W/m2)
            8. Zone Thermostat Heating Setpoint Temperature (C) 
            9. Zone Thermostat Cooling Setpoint Temperature (C) 
            10. Zone Air Temperature (C)
            11. Zone Thermal Comfort Mean Radiant Temperature (C)
            12. Zone Air Relative Humidity (%)
            13. Zone Thermal Comfort Clothing Value (clo) 
            14. Zone Thermal Comfort Fanger Model PPD
            15. Zone People Occupancy status (0 or 1)
            16. Facility Total HVAC Electric Demand Power (W)
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
        state_i_list.insert(0, nowHour);     # Add weekday and hour infomation
        state_i_list.insert(0, nowWeekday);
        state_i_list[ZPCT_RAW_IDX + 2] = 1 if state_i_list[ZPCT_RAW_IDX + 2] > 0 \
                                           else 0; # Occupancy count --> occupancy
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
            0. Day of the week
            1. Hour of the day
            2. Site Outdoor Air Drybulb Temperature
            3. Site Outdoor Air Relative Humidity
            4. Site Wind Speed
            5. Site Wind Direction
            6. Site Diffuse Solar Radiation Rate per Area
            7. Site Direct Solar Radiation Rate per Area
            8. Zone Thermostat Heating Setpoint Temperature
            9. Zone Thermostat Cooling Setpoint Temperature 
            10. Zone Air Temperature
            11. Zone Thermal Comfort Mean Radiant Temperature
            12. Zone Air Relative Humidity
            13. Zone Thermal Comfort Clothing Value
            14. Zone Thermal Comfort Fanger Model PPD
            15. Zone People Occupancy status 
            16. Facility Total HVAC Electric Demand Power
    """
    state_after_1 = process_raw_state_1(simTime, raw_state, start_year, start_mon, 
                                        start_date, start_day);
    state_after_2 = process_raw_state_2(state_after_1, min_max_limits)
    
    return state_after_2;
    
    
def get_legal_action(htStpt, clStpt, action_raw, stptLmt):
    """
    Check whether the action is legal, which is that the resulting cooling
    setpoint must be higher or equal to the resulting heating setpoint; also, 
    both heating and cooling setpoint must be within the range of stptLmt.
    
    The stptLmt will be firstly checked. The resuling heating setpoint and 
    cooling setpoint will be truncated by the stptLmt. Then clStpt > htStpt
    rule will be checked. If violated, the original htStpt and clStpt will be
    returned; else, the resulting htStpt and clStpt will be returned. 
    
    Args:
        htStpt: float
            Heating setpoint of the current observation.
        clStpt: float
            Cooling setpoint of the current observation.
        action_raw: (float, float)
            The raw action planned to be taken to the current heating and 
            cooling setpoint.
        stptLmt: (float, float)
            The low limit (included) and high limit (included) for the heating
            and cooling setpoint. 
        
    Return: ((float, float), (float, float))
        A tuple with length 2. The index 0 is a tuple of resulting heating and
        cooling setpoint, and the index 1 is a tuple of resulting effective 
        action.
    """
    res_htStpt = max(min(htStpt + action_raw[0], stptLmt[1]), stptLmt[0]);
    res_clStpt = max(min(clStpt + action_raw[1], stptLmt[1]), stptLmt[0]);
    if res_clStpt < res_htStpt:
        return ((htStpt, clStpt),(0.0, 0.0));
    else:
        return ((res_htStpt, res_clStpt),
                (res_htStpt - htStpt, res_clStpt - clStpt)); 
    
def get_reward(normalized_hvac_energy, normalized_ppd, e_weight, p_weight,
               occupancy_status):
    """
    Get the reward from hvac energy and pmv. If occupancy status is 0 (not 
    occupied), then the PPD will be 0.0; else, PPD is the original normalized
    PPD. 
    
    Args:
        normalized_hvac_energy: float
            Normalized HVAC energy ranging from 0 to 1.
        normalized_ppd: float
            Normalized PPD ranging from 0 to 1.
        e_weight: float
            The weight to HVAC energy consumption.
        p_weight: float
            The weight to PPD. 
        occupancy_status: float
            The occupancy status, 1 or 0.
    Return: float
        The combined reward. 
    """
    if occupancy_status == 0.0:
        effect_normalized_ppd = 0.0;
    else:
        effect_normalized_ppd = normalized_ppd;
        
    return - (e_weight * normalized_hvac_energy + p_weight * effect_normalized_ppd);
    
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

    def __init__(self, history_length=1):
        self._history_length = history_length;
        self._flag_start_net = True;
        self._stacked_return_net = None;

    def process_state_for_network(self, state):
        """Take the current state and return a stacked state with current
        state and the history states. 
        
        Args:
            state: python 1-D list.
                Expect python 1-D list representing the current state.
        
        Return: np.ndarray, dim = 1*m where m is the state_dim * history_length
            Stacked states.
        """
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
            
        return np.copy(self._stacked_return_net.flatten().reshape(1, -1));


    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self._flag_start_net = True;

    def get_config(self):
        return {'history_length': self.history_length}
 