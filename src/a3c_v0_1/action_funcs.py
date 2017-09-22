
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