import random
import datetime
import numpy as np

def get_env_states_stats(env, actions, start_year,
                         start_mon, start_date, start_day):
    """
    Run the env for one episode under the random policy to compute the mean
    and standard deviation of the states.
    
    Args:
        env: gym.env 
            The gym environment.
        actions: python list of tuples
            A python list of tuples, where each item in the list represents one 
            action choice. 
        start_year: int 
            Start year.
        start_mon: int 
            Start month.
        start_date: int.
            The day of the month at the start time.
        start_day: int 
            The start weekday. 0 is Monday and 6 is Sunday.
            
    Return: (np.ndarray, np.ndarray)
        Two 1-D numpy array, each has length of the raw state dim + 2 (weekday
        and hour of the day). The first array is the mean of each state feature,
        and the second array is the standard deviation of each state feature. 
    
    """
    states_hist = [];
    # Init env reset
    time_this, ob_this, is_terminal = env.reset();
    ob_this = process_raw_state_1([time_this], ob_this, start_year, start_mon, 
                          start_date, start_day)
    
    states_hist.append(ob_this);
    # Interact with the env until terminal
    while not is_terminal:
        actIdx = random.randint(0, len(actions) - 1);
        actRandom = actions[actIdx];
        htStpt_this = ob_this[8 + 2] # Heating stpt
        clStpt_this = ob_this[9 + 2] # Cooling stpt
        resStpt, effeAct = get_legal_action(htStpt_this, clStpt_this, actRandom);
        time_next, ob_next, is_terminal = env.step([resStpt[0], resStpt[1]]);
        ob_next = process_raw_state_1([time_next], ob_next, start_year, start_mon, 
                          start_date, start_day);
        states_hist.append(ob_next);
        ob_this = ob_next;
    # Calculate the statistics
    sample_mean = np.mean(np.array(states_hist), axis = 0);
    sample_stdv = np.std(np.array(states_hist), axis = 0);
    
    return (sample_mean, sample_stdv);
        

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
                          start_date, start_day):
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
        start_date: int.
            The day of the month at the start time. 
        start_day: int 
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
            8. Zone Air Temperature (C)
            9. Zone Air Relative Humidity (%)
            10. Zone Thermostat Heating Setpoint Temperature (C)
            11. Zone Thermostat Cooling Setpoint Temperature (C)
            12. Zone Thermal Comfort Fanger Model PMV
            13. Zone People Occupancy status (0 or 1)
            14. Facility Total HVAC Electric Demand Power (W)
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
                                                    start_mon, start_date, 
                                                    start_day);
        state_i_list.insert(0, nowHour);     # Add weekday and hour infomation
        state_i_list.insert(0, nowWeekday);
        state_i_list[11 + 2] = 1 if state_i_list[11 + 2] > 0 else 0; 
                                             # Occupancy count --> occupancy
        if state.shape[0] > 1:
            ret.append(state_i_list);
        else:
            ret = state_i_list;
    return ret;

def process_raw_state_2(state_prcd_1, sample_mean, sample_stdv):
    """
    Raw state processing 2. Do the following things:
    1. Standarderlize the state by the sample mean and sample stdv:
        std = (raw - mean)/stdv;
    2. Do 1 except for PMV and occupancy status. For PMV, divide the raw value
       by 3: 
        PMV <-- PMV/3.0
       For occupancy count, 1 --> 1, and 0 --> -1.
    
    Args:
        state_prcd_1: python list, 1-D or 2-D
            The processed state by process_raw_state_1. It can be only one 
            sample (1-D) or multiple samples (2-D, where each row is a sample)
        sample_mean: np.ndarray
            The 1-D numpy array with the length the same as state_prcd_1, 
            showing the mean of the each state feature.
        sample_stdv: np.ndarray
            The 1-D numpy array with the length the same as state_prcd_1,
            showing the standard deviation of the each state feature.
    Return: python list, 1-D or 2-D
        Processed state. It can be only one sample (1-D) or multiple samples 
        (2-D, where each row is a sample).
    """
    state_prcd_1 = np.array(state_prcd_1);
    # Reshape the state to 2-D if it is 1-D
    if len(state_prcd_1.shape) == 1:
        state_prcd_1 = state_prcd_1.reshape(1, -1);
    # Get a copy of occupancy status and PMV
    
    .......
    # Do standarderlization
    std_state = (np.array(state_prcd_1) - sample_mean)/sample_stdv;
    # 
    
    
    
    
def get_legal_action(htStpt, clStpt, action):
    """
    Check whether the action is legal, which is that the resulting cooling
    setpoint must be higher or equal to the resulting heating setpoint. 
    
    If the original action is legal, return the resulting heating and cooling
    setpoint; else, return original heating and cooling setpoint. In either 
    case, the effective action will also be returned.
    
    Args:
        htStpt: float
            Heating setpoint of the current observation.
        clStpt: float
            Cooling setpoint of the current observation.
        action: (float, float)
            The action taken to the current heating and cooling setpoint.
        
    Return: ((float, float), (float, float))
        A tuple with length 2. The index 0 is a tuple of resulting heating and
        cooling setpoint, and the index 1 is a tuple of resulting effective 
        action.
    """
    res_htStpt = htStpt + action[0];
    res_clStpt = clStpt + action[1];
    if res_clStpt < res_htStpt:
        return ((htStpt, clStpt),(0.0, 0.0));
    else:
        return ((res_htStpt, res_clStpt),action);
    