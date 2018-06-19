

from a3c_v0_1.action_limits import * 

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

def stpt_directSelect(action_raw, action_raw_idx, stptLmt, ob_this_raw):
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

def stpt_directSelect_sspOnly(action_raw, action_raw_idx, stptLmt, ob_this_raw):
    """ 
    Return: ((float), (float))
       
    """

    # Get the next step SWT ssp
    res_swt_ssp = action_raw[0];

    return (([res_swt_ssp]),
                (action_raw_idx))  

def stpt_directSelect_withHeuristics(action_raw, action_raw_idx, stptLmt, ob_this_raw):
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


def directPass(action_raw, action_raw_idx, stptLmt, ob_this_raw):
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
                'cslDxActCool_1':[directPass, act_limits_cslDxCool_1]}