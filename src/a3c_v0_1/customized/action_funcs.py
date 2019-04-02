
import copy
import numpy as np
from a3c_v0_1.customized.action_limits import * 

def iat_stpt_smlRefBld(action_raw, stptLmt, ob_this_raw):
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
    HTSP_RAW_IDX = 6; 
    CLSP_RAW_IDX = 7;
    htStpt = ob_this_raw[HTSP_RAW_IDX];
    clStpt = ob_this_raw[CLSP_RAW_IDX];
    res_htStpt = max(min(htStpt + action_raw[0], stptLmt[1]), stptLmt[0]);
    res_clStpt = max(min(clStpt + action_raw[1], stptLmt[1]), stptLmt[0]);
    if res_clStpt < res_htStpt:
        return ((htStpt, clStpt),(0.0, 0.0));
    else:
        return ((res_htStpt, res_clStpt),
                (res_htStpt - htStpt, res_clStpt - clStpt)); 

def mull_stpt_iw(action_raw, stptLmt, ob_this_raw):
    """
    Check whether the action is legal, which is both oat ssp and swt ssp 
    must be within the range of stptLmt.
    
    Args:
        action_raw: (float, float)
            The raw action planned to be taken to the current heating and 
            cooling setpoint.
        stptLmt: (float, float)
            The low limit (included) and high limit (included) for the heating
            and cooling setpoint. 
        ob_this_raw: 
        
    Return: ((float, float), (float, float))
        A tuple with length 2. The index 0 is a tuple of resulting heating and
        cooling setpoint, and the index 1 is a tuple of resulting effective 
        action.
    """
    OAE_RAW_IDX = 6; 
    SWT_RAW_IDX = 7;
    oae_ssp_cur = ob_this_raw[OAE_RAW_IDX];
    swt_ssp_cur = ob_this_raw[SWT_RAW_IDX];
    res_oae_ssp = max(min(oae_ssp_cur + action_raw[0], stptLmt[0][1]), stptLmt[0][0]);
    res_swt_ssp = max(min(swt_ssp_cur + action_raw[1], stptLmt[1][1]), stptLmt[1][0]);
    return ((res_oae_ssp, res_swt_ssp),
                (res_oae_ssp - oae_ssp_cur, res_swt_ssp - swt_ssp_cur)); 

def mull_stpt_oaeTrans_iw(action_raw, stptLmt, ob_this_raw):
    """
    Transfer the mull op to OAE setpoint.
    Check whether the action is legal, which is swt ssp must be within the range of stptLmt.
    
    Args:
        action_raw: (float, float)
            The raw action planned to be taken to the current heating and 
            cooling setpoint.
        stptLmt: (float, float)
            The low limit (included) and high limit (included) for the heating
            and cooling setpoint. 
        ob_this_raw: 
        
    Return: ((float, float), (float, float))
        A tuple with length 2. The index 0 is a tuple of resulting heating and
        cooling setpoint, and the index 1 is a tuple of resulting effective 
        action.
    """
    OAT_RAW_IDX = 0;
    SWT_RAW_IDX = 7;
    oat_cur = ob_this_raw[OAT_RAW_IDX]
    swt_ssp_cur = ob_this_raw[SWT_RAW_IDX];
    # Transfer the mull op from 1/0 to OAE setpoint
    if action_raw[0] == 0.0:
        res_oae_ssp = oat_cur - 5.0; # If OAE setpoint < next step OAT, mull op is off
    else:
        res_oae_ssp = oat_cur + 5.0; # If OAE setpoint > next step OAT, mull op is on
    # Get the next step SWT ssp
    res_swt_ssp = swt_ssp_cur + action_raw[1];
    res_oae_ssp = max(min(res_oae_ssp, stptLmt[0][1]), stptLmt[0][0]);
    res_swt_ssp = max(min(res_swt_ssp, stptLmt[1][1]), stptLmt[1][0]);
    return ((res_oae_ssp, res_swt_ssp),
                (action_raw[0], res_swt_ssp - swt_ssp_cur)) 

def mull_stpt_noExpTurnOffMullOP(action_raw, stptLmt, ob_this_raw):
    """
    Transfer the mull op to OAE setpoint.
    Check whether the action is legal, which is swt ssp must be within the range of stptLmt.
    
    Args:
        action_raw: (float, float)
            The raw action planned to be taken to the current heating and 
            cooling setpoint.
        stptLmt: (float, float)
            The low limit (included) and high limit (included) for the heating
            and cooling setpoint. 
        ob_this_raw: 
        
    Return: ((float, float), (float, float))
        A tuple with length 2. The index 0 is a tuple of resulting heating and
        cooling setpoint, and the index 1 is a tuple of resulting effective 
        action.
    """
    OAT_RAW_IDX = 0;
    SWT_RAW_IDX = 7;
    oat_cur = ob_this_raw[OAT_RAW_IDX]
    swt_ssp_cur = ob_this_raw[SWT_RAW_IDX];
    # Get the next step SWT ssp
    res_swt_ssp = swt_ssp_cur + action_raw[0];
    # Determine whether should turn off heating
    if res_swt_ssp < stptLmt[0][0]:
        res_oae_ssp = oat_cur - 5.0; # If res_swt_ssp < lower limit, set OAE setpoint < next step OAT, mull op is off
    else:
        res_oae_ssp = oat_cur + 5.0; # If res_swt_ssp >= lower limit, set OAE setpoint > next step OAT, mull op is on
    # Set all action into limits
    res_swt_ssp = max(min(res_swt_ssp, stptLmt[1][1]), stptLmt[1][0]);
    res_oae_ssp = max(min(res_oae_ssp, stptLmt[0][1]), stptLmt[0][0]);

    return ((res_oae_ssp, res_swt_ssp),
                (action_raw[0], res_swt_ssp - swt_ssp_cur)) 

def stpt_directSelect(action_raw, action_raw_idx, raw_state_limits, stptLmt, ob_this_raw, logger, is_show_debug):
    """
    Transfer the mull op to OAE setpoint.
    Check whether the action is legal, which is swt ssp must be within the range of stptLmt.
    
    Args:
        action_raw: (float, float)
            The raw action planned to be taken to the current heating and 
            cooling setpoint.
        stptLmt: (float, float)
            The low limit (included) and high limit (included) for the heating
            and cooling setpoint. 
        ob_this_raw: 
        
    Return: ((float, float), (float, float))
        A tuple with length 2. The index 0 is a tuple of resulting heating and
        cooling setpoint, and the index 1 is a tuple of resulting effective 
        action.
    """
    OAT_RAW_IDX = 0;
    oat_cur = ob_this_raw[OAT_RAW_IDX]
    # Get the next step SWT ssp
    res_swt_ssp = action_raw[0];
    # Determine whether should turn off heating
    if res_swt_ssp < stptLmt[1][0]:
        res_oae_ssp = oat_cur - 5.0; # If res_swt_ssp < lower limit, set OAE setpoint < next step OAT, mull op is off
    else:
        res_oae_ssp = oat_cur + 5.0; # If res_swt_ssp >= lower limit, set OAE setpoint > next step OAT, mull op is on
    # Set all action into limits
    res_oae_ssp = max(min(res_oae_ssp, stptLmt[0][1]), stptLmt[0][0]);

    return ((res_oae_ssp, res_swt_ssp),
                (action_raw_idx))

def stpt_directSelect_sspOnly(action_raw, action_raw_idx, raw_state_limits, stptLmt, ob_this_raw, logger, is_show_debug):
    """ 
    Return: ((float), (float))
       
    """

    # Get the next step SWT ssp
    res_swt_ssp = action_raw[0];

    return (([res_swt_ssp]),
                (action_raw_idx))  

def stpt_directSelect_withHeuristics(action_raw, action_raw_idx, raw_state_limits, stptLmt, ob_this_raw, logger, is_show_debug):
    """
    Transfer the mull op to OAE setpoint.
    Check whether the action is legal, which is swt ssp must be within the range of stptLmt.
    
    Args:
        action_raw: (float, float)
            The raw action planned to be taken to the current heating and 
            cooling setpoint.
        stptLmt: (float, float)
            The low limit (included) and high limit (included) for the heating
            and cooling setpoint. 
        ob_this_raw: 
        
    Return: ((float, float), (float, float))
        A tuple with length 2. The index 0 is a tuple of resulting heating and
        cooling setpoint, and the index 1 is a tuple of resulting effective 
        action.
    """
    OAT_RAW_IDX = 0;
    PPD_RAW_IDX = 7;
    IAT_RAW_IDX = 9;
    IATLG_RAW_IDX = 10;
    OCP_RAW_IDX = 11;
    oat_cur = ob_this_raw[OAT_RAW_IDX]
    ppd_cur = ob_this_raw[PPD_RAW_IDX]
    iat_cur = ob_this_raw[IAT_RAW_IDX]
    iatlg_cur = ob_this_raw[IATLG_RAW_IDX]
    ocp_cur = ob_this_raw[OCP_RAW_IDX]
    # If during unoccupied hour (IAT - IATLG) < -3, if during occupied hour PPD > 0.2
    if ((iat_cur - iatlg_cur) < -3.0 and ocp_cur == 0) or ((ppd_cur > 30 and ocp_cur == 1 and (iat_cur < iatlg_cur))):
        res_oae_ssp = oat_cur + 5.0;
        res_swt_ssp = stptLmt[1][1];
        effectiveActIdx = 10;
    else:
        # Get the next step SWT ssp
        res_swt_ssp = action_raw[0];
        # Determine whether should turn off heating
        if res_swt_ssp < stptLmt[1][0]:
            res_oae_ssp = oat_cur - 5.0; # If res_swt_ssp < lower limit, set OAE setpoint < next step OAT, mull op is off
        else:
            res_oae_ssp = oat_cur + 5.0; # If res_swt_ssp >= lower limit, set OAE setpoint > next step OAT, mull op is on
        effectiveActIdx = action_raw_idx
    # Set all action into limits
    res_oae_ssp = max(min(res_oae_ssp, stptLmt[0][1]), stptLmt[0][0]);

    return ((res_oae_ssp, res_swt_ssp),
                (effectiveActIdx)) 

def iw_iat_stpt_noExpHeatingOp(action_raw, stptLmt, ob_this_raw):
    """
    Transfer the mull op to OAE setpoint.
    Check whether the action is legal, which is swt ssp must be within the range of stptLmt.
    
    Args:
        action_raw: (float, float)
            The raw action planned to be taken to the current heating and 
            cooling setpoint.
        stptLmt: (float, float)
            The low limit (included) and high limit (included) for the heating
            and cooling setpoint. 
        ob_this_raw: 
        
    Return: ((float, float), (float, float))
        A tuple with length 2. The index 0 is a tuple of resulting heating and
        cooling setpoint, and the index 1 is a tuple of resulting effective 
        action.
    """
    OAT_RAW_IDX = 0;
    IATSSP_RAW_IDX = 8;
    oat_cur = ob_this_raw[OAT_RAW_IDX]
    iatssp_cur = ob_this_raw[IATSSP_RAW_IDX];
    # Get the next step IAT SSP
    res_iat_ssp = iatssp_cur + action_raw[0];
    # Determine whether should turn off heating
    if res_iat_ssp <= stptLmt[1][0]:
        res_oae_ssp = oat_cur - 5.0; # If res_swt_ssp < lower limit, set OAE setpoint < next step OAT, mull op is off
    else:
        res_oae_ssp = oat_cur + 5.0; # If res_swt_ssp >= lower limit, set OAE setpoint > next step OAT, mull op is on
    # Set all action into limits
    res_oae_ssp = max(min(res_oae_ssp, stptLmt[0][1]), stptLmt[0][0]);
    res_iat_ssp = max(min(res_iat_ssp, stptLmt[1][1]), stptLmt[1][0]);

    return ((res_oae_ssp, res_iat_ssp),
                (action_raw[0], res_iat_ssp)) 


def directPass(action_raw, action_raw_idx, raw_state_limits, stptLmt, ob_this_raw, logger, is_show_debug):
    """
    Pass the raw action as the output. 
    
    Args:
        action_raw: (float, )
            The raw action planned to be taken.
        action_raw_idx: int
            The index of the action in the action space.
        stptLmt: [[float, float], [float, float], ...]
            The low limit (included) and high limit (included) for each type of the actions.
        ob_this_raw: [float]
            The raw observation.
        
    Return: tuple
        A tuple with length 2. The index 0 is a tuple of resulting action, 
        and the index 1 is a tuple of resulting action idx.
    """

    return (action_raw, action_raw_idx)

def act_func_part3_v1(action_raw, action_raw_idx, raw_state_limits, stptLmt, ob_this_raw, logger, is_show_debug):
    """
    Change the action to open all chillers if the chw temp is too high.
    This is to prevent eplus simulation diverge error.
    
    Args:
        action_raw: (float, )
            The raw action planned to be taken.
        action_raw_idx: int
            The index of the action in the action space.
        stptLmt: [[float, float], [float, float], ...]
            The low limit (included) and high limit (included) for each type of the actions.
        ob_this_raw: [float]
            The raw observation.
        
    Return: tuple
        A tuple with length 2. The index 0 is a tuple of resulting action, 
        and the index 1 is a tuple of resulting action idx.
    """
    CHW_TEMP_IDX = 11;
    CHW_TEMP_STPT_IDX = 12;

    chw_temp = ob_this_raw[CHW_TEMP_IDX];
    chw_temp_stpt = ob_this_raw[CHW_TEMP_STPT_IDX];
    org_action_raw = copy.deepcopy(action_raw);
    if (chw_temp - chw_temp_stpt) > 10.0:
        action_raw = [0, 0, 0, 0, 1];
        if np.random.uniform() < 0.2:
            logger.warning('The original action %s is changed to %s to prevent too high chilled water temperature (%s C)!'
                        %(org_action_raw, action_raw, chw_temp))
    return (action_raw, action_raw_idx)

def act_func_part3_pit_det_v1(action_raw, action_raw_idx, raw_state_limits, stptLmt, ob_this_raw, logger, is_show_debug):
    """
    Limit the action to the one that meet the cooling demand
    
    Args:
        action_raw: (float, )
            The raw action planned to be taken.
        action_raw_idx: int
            The index of the action in the action space.
        stptLmt: [[float, float], [float, float], ...]
            The low limit (included) and high limit (included) for each type of the actions.
        ob_this_raw: [float]
            The raw observation.
        
    Return: tuple
        A tuple with length 2. The index 0 is a tuple of resulting action, 
        and the index 1 is a tuple of resulting action idx.
    """
    CLG_DMD_IDX = 14;
    CHILLER1_CAP = 1079600 # W
    CHILLER2_CAP = 1079600 # W
    CHILLER3_CAP = 541500 # W

    act_choice_0 = [1,0,0,0,0];
    act_choice_1 = [0,1,0,0,0];
    act_choice_2 = [0,0,1,0,0];
    act_choice_3 = [0,0,0,1,0];
    act_choice_4 = [0,0,0,0,1];

    act_0_max_cap = CHILLER3_CAP; # 1 small chiller
    act_1_max_cap = CHILLER1_CAP; # 1 big chiller
    act_2_max_cap = CHILLER1_CAP + CHILLER3_CAP; # 1 small 1 big
    act_3_max_cap = CHILLER1_CAP + CHILLER2_CAP; # 2 bigs
    act_4_max_cap = CHILLER1_CAP + CHILLER2_CAP + CHILLER3_CAP; # all chillers
    clg_demand = ob_this_raw[CLG_DMD_IDX];
    org_action_raw = copy.deepcopy(action_raw);
    org_action_raw_idx = action_raw_idx;
    # Check the current cooling demand in which range
    if clg_demand <= act_0_max_cap:
        action_ret = org_action_raw;
    elif act_1_max_cap >= clg_demand > act_0_max_cap:
        if org_action_raw_idx < 1:
            action_ret = act_choice_1;
        else:
            action_ret = org_action_raw;
    elif act_2_max_cap >= clg_demand > act_1_max_cap:
        if org_action_raw_idx < 2:
            action_ret = act_choice_2;
        else:
            action_ret = org_action_raw;
    elif act_3_max_cap >= clg_demand > act_2_max_cap:
        if org_action_raw_idx < 3:
            action_ret = act_choice_3;
        else:
            action_ret = org_action_raw;
    elif act_4_max_cap >= clg_demand > act_3_max_cap:
        if org_action_raw_idx < 4:
            action_ret = act_choice_4;
        else:
            action_ret = org_action_raw;
    else:
        action_ret = act_choice_4;
    action_ret_idx = org_action_raw_idx; 

    if action_ret != org_action_raw:
        if is_show_debug:
            logger.debug('Action function: raw action %s has been changed to %s for '
                        'the demand %s W.'%(org_action_raw, action_ret, clg_demand));
            
    return (action_ret, action_ret_idx);

def act_func_part3_pit_sto_v1(action_raw, action_raw_idx, raw_state_limits, stptLmt, ob_this_raw, logger, is_show_debug):
    """
    Limit the action to a random one that meet the cooling demand
    
    Args:
        action_raw: (float, )
            The raw action planned to be taken.
        action_raw_idx: int
            The index of the action in the action space.
        stptLmt: [[float, float], [float, float], ...]
            The low limit (included) and high limit (included) for each type of the actions.
        ob_this_raw: [float]
            The raw observation.
        
    Return: tuple
        A tuple with length 2. The index 0 is a tuple of resulting action, 
        and the index 1 is a tuple of resulting action idx.
    """
    CLG_DMD_IDX = 14;
    CHILLER1_CAP = 1079600 # W
    CHILLER2_CAP = 1079600 # W
    CHILLER3_CAP = 541500 # W

    act_choice_0 = [1,0,0,0,0];
    act_choice_1 = [0,1,0,0,0];
    act_choice_2 = [0,0,1,0,0];
    act_choice_3 = [0,0,0,1,0];
    act_choice_4 = [0,0,0,0,1];
    act_num = 5;
    act_choices = [act_choice_0, act_choice_1, act_choice_2, 
                    act_choice_3, act_choice_4];  
    act_0_max_cap = CHILLER3_CAP; # 1 small chiller
    act_1_max_cap = CHILLER1_CAP; # 1 big chiller
    act_2_max_cap = CHILLER1_CAP + CHILLER3_CAP; # 1 small 1 big
    act_3_max_cap = CHILLER1_CAP + CHILLER2_CAP; # 2 bigs
    act_4_max_cap = CHILLER1_CAP + CHILLER2_CAP + CHILLER3_CAP; # all chillers
    clg_demand = ob_this_raw[CLG_DMD_IDX];
    org_action_raw = copy.deepcopy(action_raw);
    org_action_raw_idx = action_raw_idx;
    # Check the current cooling demand in which range
    if clg_demand <= act_0_max_cap:
        action_ret_idx = org_action_raw_idx;
        action_ret = org_action_raw;
    elif act_1_max_cap >= clg_demand > act_0_max_cap:
        if org_action_raw_idx < 1:
            action_ret_idx = np.random.randint(1, act_num);
            action_ret = act_choices[action_ret_idx];
        else:
            action_ret_idx = org_action_raw_idx;
            action_ret = org_action_raw;
    elif act_2_max_cap >= clg_demand > act_1_max_cap:
        if org_action_raw_idx < 2:
            action_ret_idx = np.random.randint(2, act_num);
            action_ret = act_choices[action_ret_idx];
        else:
            action_ret_idx = org_action_raw_idx;
            action_ret = org_action_raw;
    elif act_3_max_cap >= clg_demand > act_2_max_cap:
        if org_action_raw_idx < 3:
            action_ret_idx = np.random.randint(3, act_num);
            action_ret = act_choices[action_ret_idx];
        else:
            action_ret_idx = org_action_raw_idx;
            action_ret = org_action_raw;
    elif act_4_max_cap >= clg_demand > act_3_max_cap:
        if org_action_raw_idx < 4:
            action_ret_idx = np.random.randint(4, act_num);
            action_ret = act_choices[action_ret_idx];
        else:
            action_ret_idx = org_action_raw_idx;
            action_ret = org_action_raw;
    else:
        action_ret_idx = org_action_raw_idx;
        action_ret = act_choice_4;

    if action_raw_idx != action_ret_idx:
        if is_show_debug:
            logger.debug('Action function: raw action %s has been changed to %s for '
                        'the demand %s W.'%(action_raw_idx, action_ret_idx, clg_demand));
    return (action_ret, action_ret_idx);

def act_func_part3_bej_det_v1(action_raw, action_raw_idx, raw_state_limits, stptLmt, ob_this_raw, logger, is_show_debug):
    """
    Limit the action to the one that meet the cooling demand
    
    Args:
        action_raw: (float, )
            The raw action planned to be taken.
        action_raw_idx: int
            The index of the action in the action space.
        stptLmt: [[float, float], [float, float], ...]
            The low limit (included) and high limit (included) for each type of the actions.
        ob_this_raw: [float]
            The raw observation.
        
    Return: tuple
        A tuple with length 2. The index 0 is a tuple of resulting action, 
        and the index 1 is a tuple of resulting action idx.
    """
    CLG_DMD_IDX = 14;
    CHILLER1_CAP = 1294100 # W
    CHILLER2_CAP = 1294100 # W
    CHILLER3_CAP = 685700 # W

    act_choice_0 = [1,0,0,0,0];
    act_choice_1 = [0,1,0,0,0];
    act_choice_2 = [0,0,1,0,0];
    act_choice_3 = [0,0,0,1,0];
    act_choice_4 = [0,0,0,0,1];

    act_0_max_cap = CHILLER3_CAP; # 1 small chiller
    act_1_max_cap = CHILLER1_CAP; # 1 big chiller
    act_2_max_cap = CHILLER1_CAP + CHILLER3_CAP; # 1 small 1 big
    act_3_max_cap = CHILLER1_CAP + CHILLER2_CAP; # 2 bigs
    act_4_max_cap = CHILLER1_CAP + CHILLER2_CAP + CHILLER3_CAP; # all chillers
    clg_demand = ob_this_raw[CLG_DMD_IDX];
    org_action_raw = copy.deepcopy(action_raw);
    org_action_raw_idx = action_raw_idx;
    # Check the current cooling demand in which range
    if clg_demand <= act_0_max_cap:
        action_ret = org_action_raw;
    elif act_1_max_cap >= clg_demand > act_0_max_cap:
        if org_action_raw_idx < 1:
            action_ret = act_choice_1;
        else:
            action_ret = org_action_raw;
    elif act_2_max_cap >= clg_demand > act_1_max_cap:
        if org_action_raw_idx < 2:
            action_ret = act_choice_2;
        else:
            action_ret = org_action_raw;
    elif act_3_max_cap >= clg_demand > act_2_max_cap:
        if org_action_raw_idx < 3:
            action_ret = act_choice_3;
        else:
            action_ret = org_action_raw;
    elif act_4_max_cap >= clg_demand > act_3_max_cap:
        if org_action_raw_idx < 4:
            action_ret = act_choice_4;
        else:
            action_ret = org_action_raw;
    else:
        action_ret = act_choice_4;
    action_ret_idx = org_action_raw_idx; 

    if action_ret != org_action_raw:
        if is_show_debug:
            logger.debug('Action function: raw action %s has been changed to %s for '
                        'the demand %s W.'%(org_action_raw, action_ret, clg_demand));
            
    return (action_ret, action_ret_idx);

def act_func_part3_bej_sto_v1(action_raw, action_raw_idx, raw_state_limits, stptLmt, ob_this_raw, logger, is_show_debug):
    """
    Limit the action to a random one that meet the cooling demand
    
    Args:
        action_raw: (float, )
            The raw action planned to be taken.
        action_raw_idx: int
            The index of the action in the action space.
        stptLmt: [[float, float], [float, float], ...]
            The low limit (included) and high limit (included) for each type of the actions.
        ob_this_raw: [float]
            The raw observation.
        
    Return: tuple
        A tuple with length 2. The index 0 is a tuple of resulting action, 
        and the index 1 is a tuple of resulting action idx.
    """
    CLG_DMD_IDX = 14;
    CHILLER1_CAP = 1294100 # W
    CHILLER2_CAP = 1294100 # W
    CHILLER3_CAP = 685700 # W

    act_choice_0 = [1,0,0,0,0];
    act_choice_1 = [0,1,0,0,0];
    act_choice_2 = [0,0,1,0,0];
    act_choice_3 = [0,0,0,1,0];
    act_choice_4 = [0,0,0,0,1];
    act_num = 5;
    act_choices = [act_choice_0, act_choice_1, act_choice_2, 
                    act_choice_3, act_choice_4];  
    act_0_max_cap = CHILLER3_CAP; # 1 small chiller
    act_1_max_cap = CHILLER1_CAP; # 1 big chiller
    act_2_max_cap = CHILLER1_CAP + CHILLER3_CAP; # 1 small 1 big
    act_3_max_cap = CHILLER1_CAP + CHILLER2_CAP; # 2 bigs
    act_4_max_cap = CHILLER1_CAP + CHILLER2_CAP + CHILLER3_CAP; # all chillers
    clg_demand = ob_this_raw[CLG_DMD_IDX];
    org_action_raw = copy.deepcopy(action_raw);
    org_action_raw_idx = action_raw_idx;
    # Check the current cooling demand in which range
    if clg_demand <= act_0_max_cap:
        action_ret_idx = org_action_raw_idx;
        action_ret = org_action_raw;
    elif act_1_max_cap >= clg_demand > act_0_max_cap:
        if org_action_raw_idx < 1:
            action_ret_idx = np.random.randint(1, act_num);
            action_ret = act_choices[action_ret_idx];
        else:
            action_ret_idx = org_action_raw_idx;
            action_ret = org_action_raw;
    elif act_2_max_cap >= clg_demand > act_1_max_cap:
        if org_action_raw_idx < 2:
            action_ret_idx = np.random.randint(2, act_num);
            action_ret = act_choices[action_ret_idx];
        else:
            action_ret_idx = org_action_raw_idx;
            action_ret = org_action_raw;
    elif act_3_max_cap >= clg_demand > act_2_max_cap:
        if org_action_raw_idx < 3:
            action_ret_idx = np.random.randint(3, act_num);
            action_ret = act_choices[action_ret_idx];
        else:
            action_ret_idx = org_action_raw_idx;
            action_ret = org_action_raw;
    elif act_4_max_cap >= clg_demand > act_3_max_cap:
        if org_action_raw_idx < 4:
            action_ret_idx = np.random.randint(4, act_num);
            action_ret = act_choices[action_ret_idx];
        else:
            action_ret_idx = org_action_raw_idx;
            action_ret = org_action_raw;
    else:
        action_ret_idx = org_action_raw_idx;
        action_ret = act_choice_4;

    if action_raw_idx != action_ret_idx:
        if is_show_debug:
            logger.debug('Action function: raw action %s has been changed to %s for '
                        'the demand %s W.'%(action_raw_idx, action_ret_idx, clg_demand));
    return (action_ret, action_ret_idx);

def act_func_part3_shg_det_v1(action_raw, action_raw_idx, raw_state_limits, stptLmt, ob_this_raw, logger, is_show_debug):
    """
    Limit the action to the one that meet the cooling demand
    
    Args:
        action_raw: (float, )
            The raw action planned to be taken.
        action_raw_idx: int
            The index of the action in the action space.
        stptLmt: [[float, float], [float, float], ...]
            The low limit (included) and high limit (included) for each type of the actions.
        ob_this_raw: [float]
            The raw observation.
        
    Return: tuple
        A tuple with length 2. The index 0 is a tuple of resulting action, 
        and the index 1 is a tuple of resulting action idx.
    """
    CLG_DMD_IDX = 14;
    CHILLER1_CAP = 1656300 # W
    CHILLER2_CAP = 1656300 # W
    CHILLER3_CAP = 868600 # W

    act_choice_0 = [1,0,0,0,0];
    act_choice_1 = [0,1,0,0,0];
    act_choice_2 = [0,0,1,0,0];
    act_choice_3 = [0,0,0,1,0];
    act_choice_4 = [0,0,0,0,1];

    act_0_max_cap = CHILLER3_CAP; # 1 small chiller
    act_1_max_cap = CHILLER1_CAP; # 1 big chiller
    act_2_max_cap = CHILLER1_CAP + CHILLER3_CAP; # 1 small 1 big
    act_3_max_cap = CHILLER1_CAP + CHILLER2_CAP; # 2 bigs
    act_4_max_cap = CHILLER1_CAP + CHILLER2_CAP + CHILLER3_CAP; # all chillers
    clg_demand = ob_this_raw[CLG_DMD_IDX];
    org_action_raw = copy.deepcopy(action_raw);
    org_action_raw_idx = action_raw_idx;
    # Check the current cooling demand in which range
    if clg_demand <= act_0_max_cap:
        action_ret = org_action_raw;
    elif act_1_max_cap >= clg_demand > act_0_max_cap:
        if org_action_raw_idx < 1:
            action_ret = act_choice_1;
        else:
            action_ret = org_action_raw;
    elif act_2_max_cap >= clg_demand > act_1_max_cap:
        if org_action_raw_idx < 2:
            action_ret = act_choice_2;
        else:
            action_ret = org_action_raw;
    elif act_3_max_cap >= clg_demand > act_2_max_cap:
        if org_action_raw_idx < 3:
            action_ret = act_choice_3;
        else:
            action_ret = org_action_raw;
    elif act_4_max_cap >= clg_demand > act_3_max_cap:
        if org_action_raw_idx < 4:
            action_ret = act_choice_4;
        else:
            action_ret = org_action_raw;
    else:
        action_ret = act_choice_4;
    action_ret_idx = org_action_raw_idx; 

    if action_ret != org_action_raw:
        if is_show_debug:
            logger.debug('Action function: raw action %s has been changed to %s for '
                        'the demand %s W.'%(org_action_raw, action_ret, clg_demand));
            
    return (action_ret, action_ret_idx);

def act_func_part3_shg_sto_v1(action_raw, action_raw_idx, raw_state_limits, stptLmt, ob_this_raw, logger, is_show_debug):
    """
    Limit the action to a random one that meet the cooling demand
    
    Args:
        action_raw: (float, )
            The raw action planned to be taken.
        action_raw_idx: int
            The index of the action in the action space.
        stptLmt: [[float, float], [float, float], ...]
            The low limit (included) and high limit (included) for each type of the actions.
        ob_this_raw: [float]
            The raw observation.
        
    Return: tuple
        A tuple with length 2. The index 0 is a tuple of resulting action, 
        and the index 1 is a tuple of resulting action idx.
    """
    CLG_DMD_IDX = 14;
    CHILLER1_CAP = 1656300 # W
    CHILLER2_CAP = 1656300 # W
    CHILLER3_CAP = 868600 # W

    act_choice_0 = [1,0,0,0,0];
    act_choice_1 = [0,1,0,0,0];
    act_choice_2 = [0,0,1,0,0];
    act_choice_3 = [0,0,0,1,0];
    act_choice_4 = [0,0,0,0,1];
    act_num = 5;
    act_choices = [act_choice_0, act_choice_1, act_choice_2, 
                    act_choice_3, act_choice_4];  
    act_0_max_cap = CHILLER3_CAP; # 1 small chiller
    act_1_max_cap = CHILLER1_CAP; # 1 big chiller
    act_2_max_cap = CHILLER1_CAP + CHILLER3_CAP; # 1 small 1 big
    act_3_max_cap = CHILLER1_CAP + CHILLER2_CAP; # 2 bigs
    act_4_max_cap = CHILLER1_CAP + CHILLER2_CAP + CHILLER3_CAP; # all chillers
    clg_demand = ob_this_raw[CLG_DMD_IDX];
    org_action_raw = copy.deepcopy(action_raw);
    org_action_raw_idx = action_raw_idx;
    # Check the current cooling demand in which range
    if clg_demand <= act_0_max_cap:
        action_ret_idx = org_action_raw_idx;
        action_ret = org_action_raw;
    elif act_1_max_cap >= clg_demand > act_0_max_cap:
        if org_action_raw_idx < 1:
            action_ret_idx = np.random.randint(1, act_num);
            action_ret = act_choices[action_ret_idx];
        else:
            action_ret_idx = org_action_raw_idx;
            action_ret = org_action_raw;
    elif act_2_max_cap >= clg_demand > act_1_max_cap:
        if org_action_raw_idx < 2:
            action_ret_idx = np.random.randint(2, act_num);
            action_ret = act_choices[action_ret_idx];
        else:
            action_ret_idx = org_action_raw_idx;
            action_ret = org_action_raw;
    elif act_3_max_cap >= clg_demand > act_2_max_cap:
        if org_action_raw_idx < 3:
            action_ret_idx = np.random.randint(3, act_num);
            action_ret = act_choices[action_ret_idx];
        else:
            action_ret_idx = org_action_raw_idx;
            action_ret = org_action_raw;
    elif act_4_max_cap >= clg_demand > act_3_max_cap:
        if org_action_raw_idx < 4:
            action_ret_idx = np.random.randint(4, act_num);
            action_ret = act_choices[action_ret_idx];
        else:
            action_ret_idx = org_action_raw_idx;
            action_ret = org_action_raw;
    else:
        action_ret_idx = org_action_raw_idx;
        action_ret = act_choice_4;

    if action_raw_idx != action_ret_idx:
        if is_show_debug:
            logger.debug('Action function: raw action %s has been changed to %s for '
                        'the demand %s W.'%(action_raw_idx, action_ret_idx, clg_demand));
    return (action_ret, action_ret_idx);

def act_func_part3_sgp_det_v1(action_raw, action_raw_idx, raw_state_limits, stptLmt, ob_this_raw, logger, is_show_debug):
    """
    Limit the action to the one that meet the cooling demand
    
    Args:
        action_raw: (float, )
            The raw action planned to be taken.
        action_raw_idx: int
            The index of the action in the action space.
        stptLmt: [[float, float], [float, float], ...]
            The low limit (included) and high limit (included) for each type of the actions.
        ob_this_raw: [float]
            The raw observation.
        
    Return: tuple
        A tuple with length 2. The index 0 is a tuple of resulting action, 
        and the index 1 is a tuple of resulting action idx.
    """
    CLG_DMD_IDX = 14;
    CHILLER1_CAP = 1294100 # W
    CHILLER2_CAP = 1294100 # W
    CHILLER3_CAP = 685700 # W

    act_choice_0 = [1,0,0,0,0];
    act_choice_1 = [0,1,0,0,0];
    act_choice_2 = [0,0,1,0,0];
    act_choice_3 = [0,0,0,1,0];
    act_choice_4 = [0,0,0,0,1];

    act_0_max_cap = CHILLER3_CAP; # 1 small chiller
    act_1_max_cap = CHILLER1_CAP; # 1 big chiller
    act_2_max_cap = CHILLER1_CAP + CHILLER3_CAP; # 1 small 1 big
    act_3_max_cap = CHILLER1_CAP + CHILLER2_CAP; # 2 bigs
    act_4_max_cap = CHILLER1_CAP + CHILLER2_CAP + CHILLER3_CAP; # all chillers
    clg_demand = ob_this_raw[CLG_DMD_IDX];
    org_action_raw = copy.deepcopy(action_raw);
    org_action_raw_idx = action_raw_idx;
    # Check the current cooling demand in which range
    if clg_demand <= act_0_max_cap:
        action_ret = org_action_raw;
    elif act_1_max_cap >= clg_demand > act_0_max_cap:
        if org_action_raw_idx < 1:
            action_ret = act_choice_1;
        else:
            action_ret = org_action_raw;
    elif act_2_max_cap >= clg_demand > act_1_max_cap:
        if org_action_raw_idx < 2:
            action_ret = act_choice_2;
        else:
            action_ret = org_action_raw;
    elif act_3_max_cap >= clg_demand > act_2_max_cap:
        if org_action_raw_idx < 3:
            action_ret = act_choice_3;
        else:
            action_ret = org_action_raw;
    elif act_4_max_cap >= clg_demand > act_3_max_cap:
        if org_action_raw_idx < 4:
            action_ret = act_choice_4;
        else:
            action_ret = org_action_raw;
    else:
        action_ret = act_choice_4;
    action_ret_idx = org_action_raw_idx; 

    if action_ret != org_action_raw:
        if is_show_debug:
            logger.debug('Action function: raw action %s has been changed to %s for '
                        'the demand %s W.'%(org_action_raw, action_ret, clg_demand));
            
    return (action_ret, action_ret_idx);

def act_func_part3_sgp_sto_v1(action_raw, action_raw_idx, raw_state_limits, stptLmt, ob_this_raw, logger, is_show_debug):
    """
    Limit the action to a random one that meet the cooling demand
    
    Args:
        action_raw: (float, )
            The raw action planned to be taken.
        action_raw_idx: int
            The index of the action in the action space.
        stptLmt: [[float, float], [float, float], ...]
            The low limit (included) and high limit (included) for each type of the actions.
        ob_this_raw: [float]
            The raw observation.
        
    Return: tuple
        A tuple with length 2. The index 0 is a tuple of resulting action, 
        and the index 1 is a tuple of resulting action idx.
    """
    CLG_DMD_IDX = 14;
    CHILLER1_CAP = 1294100 # W
    CHILLER2_CAP = 1294100 # W
    CHILLER3_CAP = 685700 # W

    act_choice_0 = [1,0,0,0,0];
    act_choice_1 = [0,1,0,0,0];
    act_choice_2 = [0,0,1,0,0];
    act_choice_3 = [0,0,0,1,0];
    act_choice_4 = [0,0,0,0,1];
    act_num = 5;
    act_choices = [act_choice_0, act_choice_1, act_choice_2, 
                    act_choice_3, act_choice_4];  
    act_0_max_cap = CHILLER3_CAP; # 1 small chiller
    act_1_max_cap = CHILLER1_CAP; # 1 big chiller
    act_2_max_cap = CHILLER1_CAP + CHILLER3_CAP; # 1 small 1 big
    act_3_max_cap = CHILLER1_CAP + CHILLER2_CAP; # 2 bigs
    act_4_max_cap = CHILLER1_CAP + CHILLER2_CAP + CHILLER3_CAP; # all chillers
    clg_demand = ob_this_raw[CLG_DMD_IDX];
    org_action_raw = copy.deepcopy(action_raw);
    org_action_raw_idx = action_raw_idx;
    # Check the current cooling demand in which range
    if clg_demand <= act_0_max_cap:
        action_ret_idx = org_action_raw_idx;
        action_ret = org_action_raw;
    elif act_1_max_cap >= clg_demand > act_0_max_cap:
        if org_action_raw_idx < 1:
            action_ret_idx = np.random.randint(1, act_num);
            action_ret = act_choices[action_ret_idx];
        else:
            action_ret_idx = org_action_raw_idx;
            action_ret = org_action_raw;
    elif act_2_max_cap >= clg_demand > act_1_max_cap:
        if org_action_raw_idx < 2:
            action_ret_idx = np.random.randint(2, act_num);
            action_ret = act_choices[action_ret_idx];
        else:
            action_ret_idx = org_action_raw_idx;
            action_ret = org_action_raw;
    elif act_3_max_cap >= clg_demand > act_2_max_cap:
        if org_action_raw_idx < 3:
            action_ret_idx = np.random.randint(3, act_num);
            action_ret = act_choices[action_ret_idx];
        else:
            action_ret_idx = org_action_raw_idx;
            action_ret = org_action_raw;
    elif act_4_max_cap >= clg_demand > act_3_max_cap:
        if org_action_raw_idx < 4:
            action_ret_idx = np.random.randint(4, act_num);
            action_ret = act_choices[action_ret_idx];
        else:
            action_ret_idx = org_action_raw_idx;
            action_ret = org_action_raw;
    else:
        action_ret_idx = org_action_raw_idx;
        action_ret = act_choice_4;

    if action_raw_idx != action_ret_idx:
        if is_show_debug:
            logger.debug('Action function: raw action %s has been changed to %s for '
                        'the demand %s W.'%(action_raw_idx, action_ret_idx, clg_demand));
    return (action_ret, action_ret_idx);

def act_func_part4_v1(action_raw, action_raw_idx, raw_state_limits, stptLmt, ob_this_raw, logger, is_show_debug):
    """
    Increment the current stpt by the action within the limit.
    
    Args:
        action_raw: (float, )
            The raw action planned to be taken.
        action_raw_idx: int
            The index of the action in the action space.
        stptLmt: [[float, float], [float, float], ...]
            The low limit (included) and high limit (included) for each type of the actions.
        ob_this_raw: [float]
            The raw observation.
        
    Return: tuple
        A tuple with length 2. The index 0 is a tuple of resulting action, 
        and the index 1 is a tuple of resulting action idx.
    """
    IAT_STPT_IDX = 4;

    iat_stpt_ob = ob_this_raw[IAT_STPT_IDX];
    iat_stpt_act_raw = iat_stpt_ob + action_raw[0];
    action_ret = [max(min(iat_stpt_act_raw, stptLmt[0][1]), stptLmt[0][0])];
    action_ret_idx = action_raw_idx;

    return (action_ret, action_ret_idx);

def cslDxCool_ahuStptIncmt(action_raw, action_raw_idx, raw_state_limits, stptLmt, ob_this_raw, logger, is_show_debug):
    """
    Pass the raw action as the output. 
    
    Args:
        action_raw: (float, )
            The raw action planned to be taken.
        action_raw_idx: int
            The index of the action in the action space.
        stptLmt: [[float, float], [float, float], ...]
            The low limit (included) and high limit (included) for each type of the actions.
        ob_this_raw: [float]
            The raw observation.
        
    Return: tuple
        A tuple with length 2. The index 0 is a tuple of resulting action, 
        and the index 1 is a tuple of resulting action idx.
    """
    AHUSTPT_IDX = -1;
    lastAhuStpt = ob_this_raw[AHUSTPT_IDX];
    thisAhuStpt = max(min(lastAhuStpt + action_raw[0], stptLmt[0][1]), stptLmt[0][0]);

    return ((thisAhuStpt, ), action_raw_idx)

act_func_dict = {'1':[mull_stpt_iw, act_limits_iw_1],
                '2':[mull_stpt_oaeTrans_iw, act_limits_iw_2],
                '3':[mull_stpt_noExpTurnOffMullOP, act_limits_iw_2],
                '4':[stpt_directSelect, act_limits_iw_2],
                '5':[iw_iat_stpt_noExpHeatingOp, act_limits_iw_3],
                '6':[iw_iat_stpt_noExpHeatingOp, act_limits_iw_4],
                '7':[stpt_directSelect, act_limits_iw_4],
                '8':[stpt_directSelect, act_limits_iw_5],
                '9':[stpt_directSelect_withHeuristics, act_limits_iw_5],
                '10':[stpt_directSelect_sspOnly, act_limits_iw_6],
                'cslDxActCool_1':[directPass, act_limits_cslDxCool_1],
                'cslDxActCool_2':[cslDxCool_ahuStptIncmt, act_limits_cslDxCool_1],
                'part1_v1':[directPass, act_limits_part1_v1],
                'part2_v1':[directPass, act_limits_part2_v1],
                'part2_v2':[directPass, act_limits_part2_v2],
                'part2_v3':[directPass, act_limits_part2_v3],
                'part2_v4':[directPass, act_limits_part2_v4],
                'part3_v1':[act_func_part3_v1, act_limits_part3_v1],
                'part3_pit_det_v1':[act_func_part3_pit_det_v1, act_limits_part3_v1],
                'part3_pit_sto_v1':[act_func_part3_pit_sto_v1, act_limits_part3_v1],
                'part3_bej_det_v1':[act_func_part3_bej_det_v1, act_limits_part3_v1],
                'part3_bej_sto_v1':[act_func_part3_bej_sto_v1, act_limits_part3_v1],
                'part3_shg_det_v1':[act_func_part3_shg_det_v1, act_limits_part3_v1],
                'part3_shg_sto_v1':[act_func_part3_shg_sto_v1, act_limits_part3_v1],
                'part3_sgp_det_v1':[act_func_part3_sgp_det_v1, act_limits_part3_v1],
                'part3_sgp_sto_v1':[act_func_part3_sgp_sto_v1, act_limits_part3_v1],
                'part4_v1':[act_func_part4_v1, act_limits_part4_v1]}