from a3c_v0_1.state_index import *
from numpy import linalg as LA

TIMESTATE_LEN = 2;

def ppd_energy_reward_smlRefBld(ob_next_prcd, e_weight, p_weight, mode, ppd_penalty_limit):
    """
    Get the reward from hvac energy and pmv. If occupancy status is 0 (not 
    occupied), then the PPD will be 0.0; else, PPD is the original normalized
    PPD. 
    
    Args:
        ob_next_prcd:

        e_weight: float
            The weight to HVAC energy consumption.
        p_weight: float
            The weight to PPD. 
        mode:

        ppd_penalty_limit:

    Return: float
        The combined reward. 
    """
    HVACE_RAW_IDX = 12;
    ZPPD_RAW_IDX = 10;
    ZPCT_RAW_IDX = 11;
    normalized_hvac_energy = ob_next_prcd[HVACE_RAW_IDX + TIMESTATE_LEN];
    normalized_ppd = ob_next_prcd[ZPPD_RAW_IDX + TIMESTATE_LEN];
    occupancy_status = ob_next_prcd[ZPCT_RAW_IDX + TIMESTATE_LEN];
    if occupancy_status == 0.0:
        effect_normalized_ppd = 0.0;
    else:
        effect_normalized_ppd = normalized_ppd;
    # Penalize on larger than ppd_penalty_limit PPD
    if effect_normalized_ppd > ppd_penalty_limit:
        effect_normalized_ppd = 1.0;
    if mode == 'l2':
        ret = - LA.norm(np.array([effect_normalized_ppd, normalized_hvac_energy]));
    if mode == 'linear':
        ret = - (e_weight * normalized_hvac_energy + p_weight * effect_normalized_ppd);
    return ret;


def err_energy_reward_iw(ob_next_prcd, e_weight, p_weight, err_penalty_scl):
    """
    NOTE: to make this reward function work, the e_weight needs to be much smaller than p_weight,
    if e_weight == p_weight, the policy may be stuck at turning off heating and get a -0.5 reward 
    all the time. 

    Get the reward from heating demand and error between indoor setpoint and temp. 
    
    Args:
        ob_next_prcd:

        e_weight: float
            The weight to HVAC energy consumption.
        p_weight: float
            The weight to PPD. 

        ppd_penalty_limit:

    Return: float
        The combined reward. 
    """
    HVACE_RAW_IDX = 10;
    ZAT_RAW_IDX = 9;
    ZATSSP_RAW_IDX = 8;
    normalized_hvac_energy = ob_next_prcd[HVACE_RAW_IDX + TIMESTATE_LEN];
    normalized_zat = ob_next_prcd[ZAT_RAW_IDX + TIMESTATE_LEN];
    normalized_zatssp = ob_next_prcd[ZATSSP_RAW_IDX + TIMESTATE_LEN];
    normalized_err = min(max(normalized_zatssp - normalized_zat, 0.0) * err_penalty_scl, 1.0); 
    ret = - (e_weight * normalized_hvac_energy + p_weight * normalized_err);
    return ret;

def err_energy_reward_iw_v2(ob_next_prcd, e_weight, p_weight, err_penalty_scl):
    """
    NOTE: this reward does not work well, perhaps because the agent can learn too little
    by giving it a constant negative reward -1.0 when temperature violates. But not too 
    sure. Tried e_weight == 1.0 and p_weight == 0.0, not work well. 

    Get the reward from heating demand and error between indoor setpoint and temp. 
    
    Args:
        ob_next_prcd:

        e_weight: float
            The weight to HVAC energy consumption.
        p_weight: float
            The weight to PPD. 

        ppd_penalty_limit:

    Return: float
        The combined reward. 
    """
    HVACE_RAW_IDX = 10;
    ZAT_RAW_IDX = 9;
    ZATSSP_RAW_IDX = 8;
    normalized_hvac_energy = ob_next_prcd[HVACE_RAW_IDX + TIMESTATE_LEN];
    normalized_zat = ob_next_prcd[ZAT_RAW_IDX + TIMESTATE_LEN];
    normalized_zatssp = ob_next_prcd[ZATSSP_RAW_IDX + TIMESTATE_LEN];
    normalized_err = (normalized_zat - normalized_zatssp) * err_penalty_scl; # Negative means zone is too cold
    ret = 0;
    if normalized_err < -1.0:
        ret = - 1.0; # The minimum reward
    else:
        normalized_err = min(normalized_err, 0.0) # Filter out positive error
        normalized_err = abs(normalized_err)
        ret = - (e_weight * normalized_hvac_energy + p_weight * normalized_err);
    return ret;

def err_energy_reward_iw_v3(ob_next_prcd, e_weight, p_weight, err_penalty_scl):
    """
    NOTE: this reward function does not work well because the reward is too negative (-10.0)
    which will make learning very hard. 

    Get the reward from heating demand and error between indoor setpoint and temp. 
    
    Args:
        ob_next_prcd:

        e_weight: float
            The weight to HVAC energy consumption.
        p_weight: float
            The weight to PPD. 

        ppd_penalty_limit:

    Return: float
        The combined reward. 
    """
    HVACE_RAW_IDX = 10;
    ZAT_RAW_IDX = 9;
    ZATSSP_RAW_IDX = 8;
    normalized_hvac_energy = ob_next_prcd[HVACE_RAW_IDX + TIMESTATE_LEN];
    normalized_zat = ob_next_prcd[ZAT_RAW_IDX + TIMESTATE_LEN];
    normalized_zatssp = ob_next_prcd[ZATSSP_RAW_IDX + TIMESTATE_LEN];
    normalized_err = (normalized_zat - normalized_zatssp) * err_penalty_scl; # Negative means zone is too cold
    ret = 0;
    if normalized_err < -1.0:
        ret = - 10.0; # Give a very minimum reward
    else:
        normalized_err = min(abs(normalized_err), 1.0)
        ret = - (e_weight * normalized_hvac_energy + p_weight * normalized_err);
    return ret;

def err_energy_reward_iw_v4(ob_next_prcd, e_weight, p_weight, err_penalty_scl):
    """
    NOTE: by shrink the reward by a large factor, the effect of energy consumption may be too samll.

    Get the reward from heating demand and error between indoor setpoint and temp. 
    
    Args:
        ob_next_prcd:

        e_weight: float
            The weight to HVAC energy consumption.
        p_weight: float
            The weight to PPD. 

        ppd_penalty_limit:

    Return: float
        The combined reward. 
    """
    HVACE_RAW_IDX = 10;
    ZAT_RAW_IDX = 9;
    ZATSSP_RAW_IDX = 8;
    normalized_hvac_energy = ob_next_prcd[HVACE_RAW_IDX + TIMESTATE_LEN];
    normalized_zat = ob_next_prcd[ZAT_RAW_IDX + TIMESTATE_LEN];
    normalized_zatssp = ob_next_prcd[ZATSSP_RAW_IDX + TIMESTATE_LEN];
    normalized_err = (normalized_zat - normalized_zatssp) * err_penalty_scl; # Negative means zone is too cold

    if normalized_err > 0:
        normalized_err = 0.0; 
    else:
        normalized_err = - normalized_err;
        
    loss_raw = normalized_hvac_energy + normalized_err;
    # Shrink the reward to 0 and -1
    maxLossExpected = 0.5 * err_penalty_scl;
    loss_norm = min(loss_raw / maxLossExpected, 1.0);
    reward_norm = - loss_norm;
    return reward_norm;

def err_energy_reward_iw_v5(ob_next_prcd, e_floor, p_neg_floor, err_penalty_scl):
    """
    NOTE: think about a situation that IAT is ok, but OA is cold, so we turn SSP to 75C, but the 
    reward is still zero. The same as do not turn on heating and IAT is very low. This is not right. 
    We should give temp violation more negative reward. 
          think about another one. If IAT is very low, no matter how we increase SSP, reward is 
          still zero. Agent learn nothing. 

    Get the reward from heating demand and error between indoor setpoint and temp. 
    
    Args:
        ob_next_prcd:

        e_weight: float
            The weight to HVAC energy consumption.
        p_weight: float
            The weight to PPD. 

        ppd_penalty_limit:

    Return: float
        The combined reward. 
    """
    HVACE_RAW_IDX = 10;
    ZAT_RAW_IDX = 9;
    ZATSSP_RAW_IDX = 8;
    RWD_PW = 2.0;
    normalized_hvac_energy = ob_next_prcd[HVACE_RAW_IDX + TIMESTATE_LEN];
    normalized_zat = ob_next_prcd[ZAT_RAW_IDX + TIMESTATE_LEN];
    normalized_zatssp_low = ob_next_prcd[ZATSSP_RAW_IDX + TIMESTATE_LEN];
    normalized_zatssp_high = 0.8; # About 26 C
    # The temp violation error
    if normalized_zat < normalized_zatssp_low:
        normalized_err = (normalized_zatssp_low - normalized_zat)
    elif normalized_zat > normalized_zatssp_high:
        normalized_err = (normalized_zat - normalized_zatssp_high)
    else:
        normalized_err = 0;
    normalized_err_scl = normalized_err * err_penalty_scl; # This is positive
    # The energy efficiency reward
    eff_rwd = max((1.0 - normalized_hvac_energy), e_floor); # small value means least energy efficiency
    # Cmbed rwd
    cmbd_rwd_raw = eff_rwd ** RWD_PW - normalized_err_scl ** RWD_PW
    cmbd_rwd = max(cmbd_rwd_raw, p_neg_floor); # Rwd in range 0-1

    return cmbd_rwd;

def err_energy_reward_iw_v6(ob_next_prcd, e_floor, p_division, err_penalty_scl):
    """
    NOTE: try to give the agent a clear reward about what to do at the time that IAT is very 
    low. But is it really good?
    In this function, if IAT violation is too much, then reward is proportional to the heating
    demand, meaning should consume more energy to drive the IAT violation down. 

    Get the reward from heating demand and error between indoor setpoint and temp. 
    
    Args:
        ob_next_prcd:

        e_weight: float
            The weight to HVAC energy consumption.
        p_weight: float
            The weight to PPD. 

        ppd_penalty_limit:

    Return: float
        The combined reward. 
    """
    HVACE_RAW_IDX = 10;
    ZAT_RAW_IDX = 9;
    ZATSSP_RAW_IDX = 8;
    HW_SSP_RAW_IDX = 7;
    RWD_PW = 2.0;
    normalized_hw_spp = ob_next_prcd[HW_SSP_RAW_IDX + TIMESTATE_LEN];
    normalized_hvac_energy = ob_next_prcd[HVACE_RAW_IDX + TIMESTATE_LEN];
    normalized_zat = ob_next_prcd[ZAT_RAW_IDX + TIMESTATE_LEN];
    normalized_zatssp_low = ob_next_prcd[ZATSSP_RAW_IDX + TIMESTATE_LEN];
    normalized_zatssp_high = 0.8; # About 26 C
    # The temp violation error
    if normalized_zat < normalized_zatssp_low:
        normalized_err = (normalized_zatssp_low - normalized_zat)
    elif normalized_zat > normalized_zatssp_high:
        normalized_err = (normalized_zat - normalized_zatssp_high)
    else:
        normalized_err = 0;
    normalized_err_scl = normalized_err * err_penalty_scl; # This is positive
    # The energy efficiency reward
    eff_rwd = max((1.0 - normalized_hvac_energy), e_floor); # small value means least energy efficiency
    # Cmbed rwd
    cmbd_rwd_raw = eff_rwd ** RWD_PW - normalized_err_scl ** RWD_PW
    cmbd_rwd = max(cmbd_rwd_raw, 0.0); # Rwd in range 0-1
    # If normalized error is too large, triger a special reward to let agent know more clearly about
    # what to do
    if normalized_err_scl > 1.0:
        cmbd_rwd += normalized_hvac_energy / p_division;

    return cmbd_rwd;

def ppd_energy_reward_iw_timeRelated(ob_next_prcd, e_weight, p_weight, ppd_penalty_limit):
    """
    Get the reward from hvac energy and pmv. If occupancy status is 0 (not 
    occupied), then the PPD will be 0.0; else, PPD is the original normalized
    PPD. 
    
    Args:
        ob_next_prcd:

        e_weight: float
            The weight to HVAC energy consumption.
        p_weight: float
            The weight to PPD. 
        mode:

        ppd_penalty_limit:

    Return: float
        The combined reward. 
    """
    HVACE_RAW_IDX = 12;
    ZPPD_RAW_IDX = 7;
    ZPCT_RAW_IDX = 11;
    ZIAT_SSP_LG_RAW_IDX = 10;
    ZIAT_RAW_IDX = 9;
    HOUR_RAW_IDX = -1;
    WEEKDAY_RAW_IDX = -2;
    normalized_weekday = ob_next_prcd[WEEKDAY_RAW_IDX + TIMESTATE_LEN];
    normalized_hour = ob_next_prcd[HOUR_RAW_IDX + TIMESTATE_LEN];
    normalized_hvac_energy = ob_next_prcd[HVACE_RAW_IDX + TIMESTATE_LEN];
    normalized_ppd = ob_next_prcd[ZPPD_RAW_IDX + TIMESTATE_LEN];
    normalized_pct = ob_next_prcd[ZPCT_RAW_IDX + TIMESTATE_LEN];
    normalized_iatssp_lg = ob_next_prcd[ZIAT_SSP_LG_RAW_IDX + TIMESTATE_LEN];
    normalized_iat = ob_next_prcd[ZIAT_RAW_IDX + TIMESTATE_LEN];
    comfort_rwd = 0;
    # Determine the occupied period or not to determine the comfort reward
    if normalized_pct < 0.5: # Not in occupy mode
        if (normalized_iatssp_lg - normalized_iat) > 0: # IAT is colder than ssp by logics
            comfort_rwd = -1.0;
        else:
            comfort_rwd = 0.0;
    else: # In occupy mode
        if normalized_ppd > ppd_penalty_limit:
            comfort_rwd = -1.0;
        else:
            comfort_rwd = -normalized_ppd;
    # Energy reward
    energy_rwd = -normalized_hvac_energy;
    # The combined reward
    ret = e_weight * energy_rwd + p_weight * comfort_rwd;
    return ret;

def ppd_energy_reward_iw_timeRelated_v2(ob_next_prcd, e_weight, p_weight, ppd_penalty_limit):
    """
    Get the reward from hvac energy and pmv. If occupancy status is 0 (not 
    occupied), then the PPD will be 0.0; else, PPD is the original normalized
    PPD. 
    
    Args:
        ob_next_prcd:

        e_weight: float
            The weight to HVAC energy consumption.
        p_weight: float
            The weight to PPD. 
        mode:

        ppd_penalty_limit:

    Return: float
        The combined reward. 
    """
    HVACE_RAW_IDX = 12;
    ZPPD_RAW_IDX = 7;
    ZPCT_RAW_IDX = 11;
    ZIAT_SSP_LG_RAW_IDX = 10;
    ZIAT_RAW_IDX = 9;
    HOUR_RAW_IDX = -1;
    WEEKDAY_RAW_IDX = -2;
    normalized_weekday = ob_next_prcd[WEEKDAY_RAW_IDX + TIMESTATE_LEN];
    normalized_hour = ob_next_prcd[HOUR_RAW_IDX + TIMESTATE_LEN];
    normalized_hvac_energy = ob_next_prcd[HVACE_RAW_IDX + TIMESTATE_LEN];
    normalized_ppd = ob_next_prcd[ZPPD_RAW_IDX + TIMESTATE_LEN];
    normalized_pct = ob_next_prcd[ZPCT_RAW_IDX + TIMESTATE_LEN];
    normalized_iatssp_lg = ob_next_prcd[ZIAT_SSP_LG_RAW_IDX + TIMESTATE_LEN];
    normalized_iat = ob_next_prcd[ZIAT_RAW_IDX + TIMESTATE_LEN];
    comfort_rwd = 0;
    # Determine the occupied period or not to determine the comfort reward
    if normalized_pct < 0.5: # Not in occupy mode
        if (normalized_iatssp_lg - normalized_iat) > 0: # IAT is colder than ssp by logics
            return -1.0;
        else:
            comfort_rwd = 0.0;
    else: # In occupy mode
        if normalized_ppd > ppd_penalty_limit:
            return -1.0;
        else:
            comfort_rwd = -normalized_ppd;
    # Energy reward
    energy_rwd = -normalized_hvac_energy;
    # The combined reward
    ret = e_weight * energy_rwd + p_weight * comfort_rwd;
    return ret;