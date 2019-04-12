from a3c_v0_1.state_index import *
from numpy import linalg as LA

import numpy as np

TIMESTATE_LEN = 2;

def ppd_energy_reward_smlRefBld(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, mode, ppd_penalty_limit):
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


def err_energy_reward_iw(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, err_penalty_scl):
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

def err_energy_reward_iw_v2(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, err_penalty_scl):
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

def err_energy_reward_iw_v3(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, err_penalty_scl):
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

def err_energy_reward_iw_v4(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, err_penalty_scl):
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

def ppd_energy_reward_iw_timeRelated(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, ppd_penalty_limit):
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

def ppd_energy_reward_iw_timeRelated_v2(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, ppd_penalty_limit):
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

def ppd_energy_reward_iw_timeRelated_v3(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, ppd_penalty_limit):
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
            ret = -1.0;
            return ret + 1.0;
        else:
            comfort_rwd = 0.0;
    else: # In occupy mode
        if normalized_ppd > ppd_penalty_limit:
            ret = -1.0;
            return ret + 1.0;
        else:
            comfort_rwd = -normalized_ppd;
    # Energy reward
    energy_rwd = -normalized_hvac_energy;
    # The combined reward
    ret = e_weight * energy_rwd + p_weight * comfort_rwd;
    return ret + 1.0;

def ppd_energy_reward_iw_timeRelated_v4(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, ppd_penalty_limit):
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
            comfort_rwd_imd = - (normalized_iatssp_lg - normalized_iat) * 10.0
            if comfort_rwd_imd < -1.0:
                ret = -1.0;
                return ret;
            else:
                comfort_rwd = comfort_rwd_imd;
        else:
            comfort_rwd = 0.0;
    else: # In occupy mode
        if normalized_ppd > ppd_penalty_limit:
            ret = -1.0;
            return ret;
        else:
            comfort_rwd = -normalized_ppd;
    # Energy reward
    energy_rwd = -normalized_hvac_energy;
    # The combined reward
    ret = e_weight * energy_rwd + p_weight * comfort_rwd;
    return ret;

def ppd_energy_reward_iw_timeRelated_v5(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, ppd_penalty_limit):
    """
    Get the reward from hvac energy and comfort level. If occupancy status is 0 (not 
    occupied), then the comfort level will be determined from the difference between 
    the setpoint; else, PPD is the comfort level metric. 
    
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
            comfort_rwd_imd = - (normalized_iatssp_lg - normalized_iat) * 10.0
            if comfort_rwd_imd < -1.0:
                comfort_rwd = -1.0;
            else:
                comfort_rwd = comfort_rwd_imd;
        else:
            comfort_rwd = 0.0;
    else: # In occupy mode
        comfort_rwd_imd = normalized_ppd / ppd_penalty_limit;
        if comfort_rwd_imd > 1.0:
            comfort_rwd = -1.0;
        else:
            comfort_rwd = -comfort_rwd_imd;
    # Energy reward
    energy_rwd = -normalized_hvac_energy;
    # The combined reward
    ret = e_weight * energy_rwd + p_weight * comfort_rwd;
    return ret;

def ppd_energy_reward_iw_timeRelated_v6(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, ppd_penalty_limit):
    """
    Get the reward from hvac energy and comfort level. If occupancy status is 0 (not 
    occupied), then the comfort level will be determined from the difference between 
    the setpoint; else, PPD is the comfort level metric. 
    
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
            comfort_rwd_imd = - (normalized_iatssp_lg - normalized_iat) * 10.0
            if comfort_rwd_imd < 0.0:
                return -1.0;
            else:
                comfort_rwd = 0.0;
        else:
            comfort_rwd = 0.0;
    else: # In occupy mode
        comfort_rwd_imd = normalized_ppd / ppd_penalty_limit;
        if comfort_rwd_imd > 1.0:
            comfort_rwd = -1.0;
        else:
            comfort_rwd = -comfort_rwd_imd;
    # Energy reward
    energy_rwd = -normalized_hvac_energy;
    # The combined reward
    ret = e_weight * energy_rwd + p_weight * comfort_rwd;
    return ret;

def ppd_energy_reward_iw_timeRelated_v7(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, ppd_penalty_limit, stpt_violation_scl):
    """
    Get the reward from hvac energy and comfort level. If occupancy status is 0 (not 
    occupied), then the comfort level will be determined from the difference between 
    the setpoint; else, PPD is the comfort level metric. 
    
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
            comfort_rwd = - (normalized_iatssp_lg - normalized_iat) * stpt_violation_scl
        else:
            comfort_rwd = 0.0;
    else: # In occupy mode
        if normalized_ppd > 0.1:
            ppd_scale = 1.0 / (ppd_penalty_limit - 0.1);
            comfort_rwd = -((normalized_ppd - 0.1) * ppd_scale)**2.0;
        else:
            comfort_rwd = 0.0;
    # Energy reward
    energy_rwd = -normalized_hvac_energy;
    # The combined reward
    ret = max(min(e_weight * energy_rwd + p_weight * comfort_rwd, 0.0), -1.0);
    return ret;

def ppd_energy_reward_iw_timeRelated_v8(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, ppd_penalty_limit, stpt_violation_scl):
    """
    Get the reward from hvac energy and comfort level. If occupancy status is 0 (not 
    occupied), then the comfort level will be determined from the difference between 
    the setpoint; else, PPD is the comfort level metric. 
    
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
            comfort_rwd = - (normalized_iatssp_lg - normalized_iat) * stpt_violation_scl
        else:
            comfort_rwd = 0.0;
    else: # In occupy mode
        if normalized_ppd > 0.1:
            ppd_scale = 1.0 / (ppd_penalty_limit - 0.1);
            comfort_rwd = -((normalized_ppd - 0.1) * ppd_scale)**2.0;
        else:
            comfort_rwd = 0.0;
    # Energy reward
    energy_rwd = -normalized_hvac_energy;
    # The combined reward
    ret = max(min(e_weight * energy_rwd + p_weight * comfort_rwd, 0.0), -1.0);
    # Shift reward to 0 - 1
    ret += 1.0;
    return ret;

def ppd_energy_reward_iw_timeRelated_v9(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, ppd_penalty_limit, stpt_violation_scl):
    """
    Get the reward from hvac energy and comfort level. If occupancy status is 0 (not 
    occupied), then the comfort level will be determined from the difference between 
    the setpoint; else, PPD is the comfort level metric. 
    
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
    tempVio_rwd = 0;
    ppd_rwd = 0;
    # Temp violation penalty
    if (normalized_iatssp_lg - normalized_iat) > 0: # IAT is colder than ssp by logics
        tempVio_rwd = - (normalized_iatssp_lg - normalized_iat) * stpt_violation_scl
    else:
        tempVio_rwd = 0.0;
    # Determine the occupied period or not to determine the comfort reward
    if normalized_pct < 0.5: # Not in occupy mode
        comfort_rwd = tempVio_rwd;
    else: # In occupy mode, use the minimum one between the PPD penalty and temp violation penalty
        if normalized_ppd > 0.1:
            ppd_scale = 1.0 / (ppd_penalty_limit - 0.1);
            ppd_rwd = -((normalized_ppd - 0.1) * ppd_scale)**2.0;
        else:
            ppd_rwd = 0.0;
        comfort_rwd = min(ppd_rwd, tempVio_rwd);
    # Energy reward
    energy_rwd = -normalized_hvac_energy;
    # The combined reward
    ret = max(min(e_weight * energy_rwd + p_weight * comfort_rwd, 0.0), -1.0);
    # Shift reward to 0 - 1
    ret += 1.0;
    return ret;


def stptVio_energy_reward_cslDxCool_v1(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Get the reward from hvac energy and indoor air temperature setpoint violation level (the max one
    is used for the multi-zone case). 
    
    Args:
        ob_next_prcd:
            Processed observation.
        e_weight: float
            The weight to HVAC energy consumption.
        p_weight: float
            The weight to indoor air temperature setpoint violation.
        ppd_penalty_limit:

    Return: float
        The reward. 
    """
    ZONE_NUM = 22;
    IAT_FIRST_RAW_IDX = 4;
    IATSSP_FIRST_RAW_IDX = 26;
    ENERGY_RAW_IDX = 48;
    normalized_iats = np.array(ob_next_prcd[TIMESTATE_LEN + IAT_FIRST_RAW_IDX: TIMESTATE_LEN + IAT_FIRST_RAW_IDX + ZONE_NUM]);
    normalized_iatssp = np.array(ob_next_prcd[TIMESTATE_LEN + IATSSP_FIRST_RAW_IDX: TIMESTATE_LEN + IATSSP_FIRST_RAW_IDX + ZONE_NUM]);
    normalized_sspVio_max = max(normalized_iats - normalized_iatssp); # For cooling, the IAT should be less than the IATSSP
    normalized_energy = ob_next_prcd[TIMESTATE_LEN + ENERGY_RAW_IDX];
    
    energy_rwd = - normalized_energy;
    comfort_rwd = - max(normalized_sspVio_max, 0) * stpt_violation_scl; # Penalty for the positive setpoint violation
    ret = max(min(e_weight * energy_rwd + p_weight * comfort_rwd, 0.0), -1.0);
    # Shift reward to 0 - 1
    ret += 1.0;
    return ret;


def stpt_viol_energy_reward_part1_v1(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Get the reward from hvac energy and indoor air temperature setpoint violation level (the max one
    is used for the multi-zone case). 
    reward = [- p * (scl * (max_htgssp_viol + max_clgssp_viol)) - e * energy]_{limit:0,-1} + 1
    
    Args:
        ob_next_prcd:
            Processed observation.
        e_weight: float
            The weight to HVAC energy consumption.
        p_weight: float
            The weight to indoor air temperature setpoint violation.
        ppd_penalty_limit:

    Return: float
        The reward. 
    """
    ZONE_NUM = 22;
    IAT_FIRST_RAW_IDX = 4;
    CLGSSP_FIRST_RAW_IDX = 26;
    HTGSSP_FIRST_RAW_IDX = 48;
    ENERGY_RAW_IDX = 70;
    normalized_iats = np.array(ob_next_prcd[TIMESTATE_LEN + IAT_FIRST_RAW_IDX: TIMESTATE_LEN + IAT_FIRST_RAW_IDX + ZONE_NUM]);
    normalized_clgssp = np.array(ob_next_prcd[TIMESTATE_LEN + CLGSSP_FIRST_RAW_IDX: TIMESTATE_LEN + CLGSSP_FIRST_RAW_IDX + ZONE_NUM]);
    normalized_htgssp = np.array(ob_next_prcd[TIMESTATE_LEN + HTGSSP_FIRST_RAW_IDX: TIMESTATE_LEN + HTGSSP_FIRST_RAW_IDX + ZONE_NUM]);
    normalized_clgssp_viol_max = max(normalized_iats - normalized_clgssp); # For cooling, the IAT should be less than the CLGSSP
    normalized_htgssp_viol_max = max(normalized_htgssp - normalized_iats); # For heating, the IAT should be larger than the HTGSSP
    normalized_energy = ob_next_prcd[TIMESTATE_LEN + ENERGY_RAW_IDX];
    
    energy_rwd = - normalized_energy;
    comfort_rwd = - (max(normalized_clgssp_viol_max, 0) + max(normalized_htgssp_viol_max, 0)) * stpt_violation_scl; # Penalty for the positive setpoint violation
    ret = max(min(e_weight * energy_rwd + p_weight * comfort_rwd, 0.0), -1.0); # Limit to 0 ~ -1
    # Shift reward to 0 ~ 1
    ret += 1.0;
    return ret;

def stpt_viol_energy_reward_part2_v1(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Get the reward from hvac energy and indoor air temperature setpoint violation level (the max one
    is used for the multi-zone case). 
    reward = [- p * (scl * (max_htgssp_viol + max_clgssp_viol)) - e * energy]_{limit:0,-1} + 1
    
    Args:
        ob_next_prcd:
            Processed observation.
        e_weight: float
            The weight to HVAC energy consumption.
        p_weight: float
            The weight to indoor air temperature setpoint violation.
        ppd_penalty_limit:

    Return: float
        The reward. 
    """
    ZONE_NUM = 22;
    IAT_FIRST_RAW_IDX = 4;
    CLGSSP_FIRST_RAW_IDX = 26;
    HTGSSP_FIRST_RAW_IDX = 48;
    ENERGY_RAW_IDX = 70;
    normalized_iats = np.array(ob_next_prcd[TIMESTATE_LEN + IAT_FIRST_RAW_IDX: TIMESTATE_LEN + IAT_FIRST_RAW_IDX + ZONE_NUM]);
    normalized_clgssp = np.array(ob_next_prcd[TIMESTATE_LEN + CLGSSP_FIRST_RAW_IDX: TIMESTATE_LEN + CLGSSP_FIRST_RAW_IDX + ZONE_NUM]);
    normalized_htgssp = np.array(ob_next_prcd[TIMESTATE_LEN + HTGSSP_FIRST_RAW_IDX: TIMESTATE_LEN + HTGSSP_FIRST_RAW_IDX + ZONE_NUM]);
    normalized_clgssp_viol_max = max(normalized_iats - normalized_clgssp); # For cooling, the IAT should be less than the CLGSSP
    normalized_htgssp_viol_max = max(normalized_htgssp - normalized_iats); # For heating, the IAT should be larger than the HTGSSP
    normalized_energy = ob_next_prcd[TIMESTATE_LEN + ENERGY_RAW_IDX];
    
    energy_rwd = - normalized_energy;
    comfort_rwd = - (max(normalized_clgssp_viol_max, 0) + max(normalized_htgssp_viol_max, 0)) * stpt_violation_scl; # Penalty for the positive setpoint violation
    ret = max(min(e_weight * energy_rwd + p_weight * comfort_rwd, 0.0), -1.0); # Limit to 0 ~ -1
    # Shift reward to 0 ~ 1
    ret += 1.0;
    return ret;

def stpt_viol_energy_metric_part1_v1(ob_next_raw, this_ep_energy, this_ep_comfort):
    """
    
    """
    ZONE_NUM = 22;
    IAT_FIRST_RAW_IDX = 4;
    CLGSSP_FIRST_RAW_IDX = 26;
    HTGSSP_FIRST_RAW_IDX = 48;
    ENERGY_RAW_IDX = 70;
    iats = np.array(ob_next_raw[IAT_FIRST_RAW_IDX: IAT_FIRST_RAW_IDX + ZONE_NUM]);
    clgssp = np.array(ob_next_raw[CLGSSP_FIRST_RAW_IDX: CLGSSP_FIRST_RAW_IDX + ZONE_NUM]);
    htgssp = np.array(ob_next_raw[HTGSSP_FIRST_RAW_IDX: HTGSSP_FIRST_RAW_IDX + ZONE_NUM]);
    clgssp_viol_sum = sum(((iats - clgssp) > 0.5)); # For cooling, the IAT should be less than the CLGSSP (tolerance 0.5 C)
    htgssp_viol_sum = sum(((htgssp - iats) > 0.5)); # For heating, the IAT should be larger than the HTGSSP (tolerance 0.5 C)
    energy = ob_next_raw[ENERGY_RAW_IDX];
    
    this_ep_energy_toNow = this_ep_energy + energy; # Unit is Wh*timestep
    this_ep_comfort_toNow = this_ep_comfort + clgssp_viol_sum + htgssp_viol_sum; # Unit is hr*timestep
    
    return (this_ep_energy_toNow, this_ep_comfort_toNow);

def stpt_viol_energy_metric_part2_v1(ob_next_raw, this_ep_energy, this_ep_comfort):
    """
    
    """
    ZONE_NUM = 22;
    IAT_FIRST_RAW_IDX = 4;
    CLGSSP_FIRST_RAW_IDX = 26;
    HTGSSP_FIRST_RAW_IDX = 48;
    ENERGY_RAW_IDX = 70;
    iats = np.array(ob_next_raw[IAT_FIRST_RAW_IDX: IAT_FIRST_RAW_IDX + ZONE_NUM]);
    clgssp = np.array(ob_next_raw[CLGSSP_FIRST_RAW_IDX: CLGSSP_FIRST_RAW_IDX + ZONE_NUM]);
    htgssp = np.array(ob_next_raw[HTGSSP_FIRST_RAW_IDX: HTGSSP_FIRST_RAW_IDX + ZONE_NUM]);
    clgssp_viol_sum = sum(((iats - clgssp) > 0.5)); # For cooling, the IAT should be less than the CLGSSP (tolerance 0.5 C)
    htgssp_viol_sum = sum(((htgssp - iats) > 0.5)); # For heating, the IAT should be larger than the HTGSSP (tolerance 0.5 C)
    energy = ob_next_raw[ENERGY_RAW_IDX];
    
    this_ep_energy_toNow = this_ep_energy + energy; # Unit is Wh*timestep
    this_ep_comfort_toNow = this_ep_comfort + clgssp_viol_sum + htgssp_viol_sum; # Unit is hr*timestep
    
    return (this_ep_energy_toNow, this_ep_comfort_toNow);

def stptVio_energy_metric_cslDxCool_v1(ob_next_raw, this_ep_energy, this_ep_comfort):
    """
    
    """
    ZONE_NUM = 22;
    IAT_FIRST_RAW_IDX = 4;
    IATSSP_FIRST_RAW_IDX = 26;
    ENERGY_RAW_IDX = 48;
    iats = np.array(ob_next_raw[IAT_FIRST_RAW_IDX: IAT_FIRST_RAW_IDX + ZONE_NUM]);
    iatssp = np.array(ob_next_raw[IATSSP_FIRST_RAW_IDX: IATSSP_FIRST_RAW_IDX + ZONE_NUM]);
    sspVio_sum = sum(((iats - iatssp) > 0.5)/12); # For cooling, the IAT should be less than the IATSSP (tolerance 0.5 C)
    energy = ob_next_raw[ENERGY_RAW_IDX];
    
    this_ep_energy_toNow = this_ep_energy + energy/12/1000; # Unit is kWh
    this_ep_comfort_toNow = this_ep_comfort + sspVio_sum; # Unit is hr
    
    return (this_ep_energy_toNow, this_ep_comfort_toNow);

def stptVio_energy_reward_cslDxCool_v2(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Get the reward from hvac energy and indoor air temperature setpoint violation level (the max one
    is used for the multi-zone case). 
    
    Args:
        ob_next_prcd:
            Processed observation.
        e_weight: float
            The weight to HVAC energy consumption.
        p_weight: float
            The weight to indoor air temperature setpoint violation.
        ppd_penalty_limit:

    Return: float
        The reward. 
    """
    STPTVIO_IDX = 4;
    ENERGY_RAW_IDX = 5;
    normalized_sspVio_max = ob_next_prcd[TIMESTATE_LEN + STPTVIO_IDX]; 
    normalized_energy = ob_next_prcd[TIMESTATE_LEN + ENERGY_RAW_IDX];
    
    energy_rwd = - normalized_energy;
    comfort_rwd = - max(normalized_sspVio_max, 0) * stpt_violation_scl; # Penalty for the positive setpoint violation
    ret = max(min(e_weight * energy_rwd + p_weight * comfort_rwd, 0.0), -1.0);
    # Shift reward to 0 - 1
    ret += 1.0;
    return ret;

def stptVio_energy_metric_cslDxCool_v2(ob_next_raw, this_ep_energy, this_ep_comfort):
    """
    
    """
    STPTVIO_IDX = 4;
    ENERGY_RAW_IDX = 5;
    sspVioMax = ob_next_raw[STPTVIO_IDX];
    sspVio_hr = (sspVioMax > 0.5)/12; # For cooling, the IAT should be less than the IATSSP (tolerance 0.5 C)
    energy = ob_next_raw[ENERGY_RAW_IDX];
    
    this_ep_energy_toNow = this_ep_energy + energy/12/1000; # Unit is kWh
    this_ep_comfort_toNow = this_ep_comfort + sspVio_hr; # Unit is hr
    
    return (this_ep_energy_toNow, this_ep_comfort_toNow);

def rl_parametric_reward_part3_v1(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Reward = system_cop + shortcycle_penl + lowplr_penl + stptnmt_penl
    """
    CHILLER1_ONOFF_IDX = 2;
    CHILLER1_SHTCY_IDX = 3;
    CHILLER1_PRTLR_IDX = 4;

    CHILLER2_ONOFF_IDX = 5;
    CHILLER2_SHTCY_IDX = 6;
    CHILLER2_PRTLR_IDX = 7;

    CHILLER3_ONOFF_IDX = 8;
    CHILLER3_SHTCY_IDX = 9;
    CHILLER3_PRTLR_IDX = 10;

    CHW_TEMP_IDX = 11;
    CHW_TEMP_STPT_IDX = 12;

    CLG_DMD_IDX = 14;
    CLG_DLD_IDX = 15;
    HVAC_E_IDX = 16;

    clg_delivered_min = pcd_state_limits[0][CLG_DLD_IDX + TIMESTATE_LEN]
    hvac_energy_min = pcd_state_limits[0][HVAC_E_IDX + TIMESTATE_LEN]
    clg_delivered_max = pcd_state_limits[1][CLG_DLD_IDX + TIMESTATE_LEN]
    hvac_energy_max = pcd_state_limits[1][HVAC_E_IDX + TIMESTATE_LEN]
    # shtcyc_penl: penalize short cycle actions
    is_chillers_short_cycle_ls = [_is_chiller_short_cycle(ob_this_prcd, ob_next_prcd, 
                                                        CHILLER1_SHTCY_IDX, CHILLER1_ONOFF_IDX),
                                _is_chiller_short_cycle(ob_this_prcd, ob_next_prcd, 
                                                        CHILLER2_SHTCY_IDX, CHILLER2_ONOFF_IDX),
                                _is_chiller_short_cycle(ob_this_prcd, ob_next_prcd, 
                                                        CHILLER3_SHTCY_IDX, CHILLER3_ONOFF_IDX),
                                ];
    is_chillers_short_cycle = max(is_chillers_short_cycle_ls);
    shtcyc_penl = is_chillers_short_cycle * 1.0;
    # lowplr_penl: penalize too low part load ratio
    chiller1_plr = ob_next_prcd[CHILLER1_PRTLR_IDX + TIMESTATE_LEN];
    chiller2_plr = ob_next_prcd[CHILLER2_PRTLR_IDX + TIMESTATE_LEN];
    chiller3_plr = ob_next_prcd[CHILLER3_PRTLR_IDX + TIMESTATE_LEN];
    chiller1_on = ob_next_prcd[CHILLER1_ONOFF_IDX + TIMESTATE_LEN];
    chiller2_on = ob_next_prcd[CHILLER2_ONOFF_IDX + TIMESTATE_LEN];
    chiller3_on = ob_next_prcd[CHILLER3_ONOFF_IDX + TIMESTATE_LEN];
    is_chillers_plr_low_ls = [chiller1_plr == 0.0 and chiller1_on == 1,
                              chiller2_plr == 0.0 and chiller2_on == 1,
                              chiller3_plr == 0.0 and chiller3_on == 1]; 
                              # In eplus, the plr cannot be lower than the low
                              # limit of the plr, this is why == 0.0
    is_chillers_plr_low = max(is_chillers_plr_low_ls);
    lowplr_penl = is_chillers_plr_low * 1.0;
    # stptnm_penl: penalize the stpt not met
    chw_temp = ob_next_prcd[CHW_TEMP_IDX + TIMESTATE_LEN];
    chw_temp_stpt = ob_next_prcd[CHW_TEMP_STPT_IDX + TIMESTATE_LEN];
    stptnm_penl = max((chw_temp - chw_temp_stpt), 0) * stpt_violation_scl * p_weight;
    # erwd: energy reward
    clg_delivered = (ob_next_prcd[CLG_DLD_IDX + TIMESTATE_LEN] 
                    * (clg_delivered_max - clg_delivered_min) + clg_delivered_min);
    hvac_energy = (ob_next_prcd[HVAC_E_IDX + TIMESTATE_LEN] 
                    * (hvac_energy_max - hvac_energy_min) + hvac_energy_min);
    system_cop = max(clg_delivered/hvac_energy, 0);
    system_cop_scl = system_cop/e_weight;
    # final reward
    reward = max(system_cop_scl - stptnm_penl - lowplr_penl - shtcyc_penl, 0);
    return reward;


def rl_parametric_reward_part3_v2(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Reward = system_cop + shortcycle_penl + stptnmt_penl
    """
    CHILLER1_ONOFF_IDX = 2;
    CHILLER1_SHTCY_IDX = 3;
    CHILLER1_PRTLR_IDX = 4;

    CHILLER2_ONOFF_IDX = 5;
    CHILLER2_SHTCY_IDX = 6;
    CHILLER2_PRTLR_IDX = 7;

    CHILLER3_ONOFF_IDX = 8;
    CHILLER3_SHTCY_IDX = 9;
    CHILLER3_PRTLR_IDX = 10;

    CHW_TEMP_IDX = 11;
    CHW_TEMP_STPT_IDX = 12;

    CLG_DMD_IDX = 14;
    CLG_DLD_IDX = 15;
    HVAC_E_IDX = 16;

    clg_delivered_min = pcd_state_limits[0][CLG_DLD_IDX + TIMESTATE_LEN]
    hvac_energy_min = pcd_state_limits[0][HVAC_E_IDX + TIMESTATE_LEN]
    clg_delivered_max = pcd_state_limits[1][CLG_DLD_IDX + TIMESTATE_LEN]
    hvac_energy_max = pcd_state_limits[1][HVAC_E_IDX + TIMESTATE_LEN]
    # shtcyc_penl: penalize short cycle actions
    is_chillers_short_cycle_ls = [_is_chiller_short_cycle(ob_this_prcd, ob_next_prcd, 
                                                        CHILLER1_SHTCY_IDX, CHILLER1_ONOFF_IDX),
                                _is_chiller_short_cycle(ob_this_prcd, ob_next_prcd, 
                                                        CHILLER2_SHTCY_IDX, CHILLER2_ONOFF_IDX),
                                _is_chiller_short_cycle(ob_this_prcd, ob_next_prcd, 
                                                        CHILLER3_SHTCY_IDX, CHILLER3_ONOFF_IDX),
                                ];
    is_chillers_short_cycle = max(is_chillers_short_cycle_ls);
    shtcyc_penl = is_chillers_short_cycle * 1.0;
    # stptnm_penl: penalize the stpt not met
    chw_temp = ob_next_prcd[CHW_TEMP_IDX + TIMESTATE_LEN];
    chw_temp_stpt = ob_next_prcd[CHW_TEMP_STPT_IDX + TIMESTATE_LEN];
    stptnm_penl = max((chw_temp - chw_temp_stpt), 0) * stpt_violation_scl * p_weight;
    # erwd: energy reward
    clg_delivered = (ob_next_prcd[CLG_DLD_IDX + TIMESTATE_LEN] 
                    * (clg_delivered_max - clg_delivered_min) + clg_delivered_min);
    hvac_energy = (ob_next_prcd[HVAC_E_IDX + TIMESTATE_LEN] 
                    * (hvac_energy_max - hvac_energy_min) + hvac_energy_min);
    system_cop = max(clg_delivered/hvac_energy, 0);
    system_cop_scl = system_cop/e_weight;
    # final reward
    reward = max(system_cop_scl - stptnm_penl - shtcyc_penl, 0);
    return reward;

def rl_parametric_reward_part3_v3(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Reward = system_energy + shortcycle_penl + lowplr_penl + stptnmt_penl
    """
    CHILLER1_ONOFF_IDX = 2;
    CHILLER1_SHTCY_IDX = 3;
    CHILLER1_PRTLR_IDX = 4;

    CHILLER2_ONOFF_IDX = 5;
    CHILLER2_SHTCY_IDX = 6;
    CHILLER2_PRTLR_IDX = 7;

    CHILLER3_ONOFF_IDX = 8;
    CHILLER3_SHTCY_IDX = 9;
    CHILLER3_PRTLR_IDX = 10;

    CHW_TEMP_IDX = 11;
    CHW_TEMP_STPT_IDX = 12;

    CLG_DMD_IDX = 14;
    CLG_DLD_IDX = 15;
    HVAC_E_IDX = 16;

    clg_delivered_min = pcd_state_limits[0][CLG_DLD_IDX + TIMESTATE_LEN]
    hvac_energy_min = pcd_state_limits[0][HVAC_E_IDX + TIMESTATE_LEN]
    clg_delivered_max = pcd_state_limits[1][CLG_DLD_IDX + TIMESTATE_LEN]
    hvac_energy_max = pcd_state_limits[1][HVAC_E_IDX + TIMESTATE_LEN]
    # shtcyc_penl: penalize short cycle actions
    is_chillers_short_cycle_ls = [_is_chiller_short_cycle(ob_this_prcd, ob_next_prcd, 
                                                        CHILLER1_SHTCY_IDX, CHILLER1_ONOFF_IDX),
                                _is_chiller_short_cycle(ob_this_prcd, ob_next_prcd, 
                                                        CHILLER2_SHTCY_IDX, CHILLER2_ONOFF_IDX),
                                _is_chiller_short_cycle(ob_this_prcd, ob_next_prcd, 
                                                        CHILLER3_SHTCY_IDX, CHILLER3_ONOFF_IDX),
                                ];
    is_chillers_short_cycle = max(is_chillers_short_cycle_ls);
    shtcyc_penl = is_chillers_short_cycle * 1.0;
    # lowplr_penl: penalize too low part load ratio
    chiller1_plr = ob_next_prcd[CHILLER1_PRTLR_IDX + TIMESTATE_LEN];
    chiller2_plr = ob_next_prcd[CHILLER2_PRTLR_IDX + TIMESTATE_LEN];
    chiller3_plr = ob_next_prcd[CHILLER3_PRTLR_IDX + TIMESTATE_LEN];
    chiller1_on = ob_next_prcd[CHILLER1_ONOFF_IDX + TIMESTATE_LEN];
    chiller2_on = ob_next_prcd[CHILLER2_ONOFF_IDX + TIMESTATE_LEN];
    chiller3_on = ob_next_prcd[CHILLER3_ONOFF_IDX + TIMESTATE_LEN];
    is_chillers_plr_low_ls = [chiller1_plr == 0.0 and chiller1_on == 1,
                              chiller2_plr == 0.0 and chiller2_on == 1,
                              chiller3_plr == 0.0 and chiller3_on == 1]; 
                              # In eplus, the plr cannot be lower than the low
                              # limit of the plr, this is why == 0.0
    is_chillers_plr_low = max(is_chillers_plr_low_ls);
    lowplr_penl = is_chillers_plr_low * 1.0;
    # stptnm_penl: penalize the stpt not met
    chw_temp = ob_next_prcd[CHW_TEMP_IDX + TIMESTATE_LEN];
    chw_temp_stpt = ob_next_prcd[CHW_TEMP_STPT_IDX + TIMESTATE_LEN];
    stptnm_penl = max((chw_temp - chw_temp_stpt), 0) * stpt_violation_scl * p_weight;
    # erwd: energy reward
    hvac_energy = ob_next_prcd[HVAC_E_IDX + TIMESTATE_LEN];
    hvac_energy_rwd = 1 - hvac_energy * e_weight;
    # final reward
    reward = max(hvac_energy_rwd - stptnm_penl - lowplr_penl - shtcyc_penl, 0);
    return reward;

def rl_parametric_reward_part3_v4(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Reward = system_energy + shortcycle_penl + stptnmt_penl
    """
    CHILLER1_ONOFF_IDX = 2;
    CHILLER1_SHTCY_IDX = 3;
    CHILLER1_PRTLR_IDX = 4;

    CHILLER2_ONOFF_IDX = 5;
    CHILLER2_SHTCY_IDX = 6;
    CHILLER2_PRTLR_IDX = 7;

    CHILLER3_ONOFF_IDX = 8;
    CHILLER3_SHTCY_IDX = 9;
    CHILLER3_PRTLR_IDX = 10;

    CHW_TEMP_IDX = 11;
    CHW_TEMP_STPT_IDX = 12;

    CLG_DMD_IDX = 14;
    CLG_DLD_IDX = 15;
    HVAC_E_IDX = 16;

    clg_delivered_min = pcd_state_limits[0][CLG_DLD_IDX + TIMESTATE_LEN]
    hvac_energy_min = pcd_state_limits[0][HVAC_E_IDX + TIMESTATE_LEN]
    clg_delivered_max = pcd_state_limits[1][CLG_DLD_IDX + TIMESTATE_LEN]
    hvac_energy_max = pcd_state_limits[1][HVAC_E_IDX + TIMESTATE_LEN]
    # shtcyc_penl: penalize short cycle actions
    is_chillers_short_cycle_ls = [_is_chiller_short_cycle(ob_this_prcd, ob_next_prcd, 
                                                        CHILLER1_SHTCY_IDX, CHILLER1_ONOFF_IDX),
                                _is_chiller_short_cycle(ob_this_prcd, ob_next_prcd, 
                                                        CHILLER2_SHTCY_IDX, CHILLER2_ONOFF_IDX),
                                _is_chiller_short_cycle(ob_this_prcd, ob_next_prcd, 
                                                        CHILLER3_SHTCY_IDX, CHILLER3_ONOFF_IDX),
                                ];
    is_chillers_short_cycle = max(is_chillers_short_cycle_ls);
    shtcyc_penl = is_chillers_short_cycle * 1.0;
    # stptnm_penl: penalize the stpt not met
    chw_temp = ob_next_prcd[CHW_TEMP_IDX + TIMESTATE_LEN];
    chw_temp_stpt = ob_next_prcd[CHW_TEMP_STPT_IDX + TIMESTATE_LEN];
    stptnm_penl = max((chw_temp - chw_temp_stpt), 0) * stpt_violation_scl * p_weight;
    # erwd: energy reward
    hvac_energy = ob_next_prcd[HVAC_E_IDX + TIMESTATE_LEN];
    hvac_energy_rwd = 1 - hvac_energy;
    # final reward
    reward = max(hvac_energy_rwd - stptnm_penl - shtcyc_penl, 0);
    return reward;

def rl_parametric_metric_part3_v1(ob_next_raw, this_ep_energy, this_ep_comfort):
    """
    
    """ 
    CHW_TEMP_IDX = 11;
    CHW_TEMP_STPT_IDX = 12;
    HVAC_E_IDX = 16;

    energy = ob_next_raw[HVAC_E_IDX]; # W
    stpt_vio = 1.0*((ob_next_raw[CHW_TEMP_IDX] - ob_next_raw[CHW_TEMP_STPT_IDX]) > 0.02)
    
    this_ep_energy_toNow = this_ep_energy + energy; # Unit is Wh*timestep
    this_ep_comfort_toNow = this_ep_comfort + stpt_vio; # Unit is hr*timestep
    
    return (this_ep_energy_toNow, this_ep_comfort_toNow);

def rl_parametric_reward_part4_v1(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Reward = system_energy + pmv_penal
    """
    PMV_IDX = 6;
    OCP_IDX = 7;
    BGR_IDX = 9;

    pmv_min = pcd_state_limits[0][PMV_IDX + TIMESTATE_LEN]
    pmv_max = pcd_state_limits[1][PMV_IDX + TIMESTATE_LEN]
    # PMV_penl: penalize PMV smaller than -0.5
    pmv_prcd = ob_next_prcd[PMV_IDX + TIMESTATE_LEN];
    pmv_raw = pmv_min + pmv_prcd*(pmv_max - pmv_min);
    ocp = ob_next_prcd[OCP_IDX + TIMESTATE_LEN];
    pmv_penl = 0.0;
    pmv_thres = -0.5;
    if ocp == 0:
        pmv_penl = 0.0;
    else:
        if pmv_raw >= pmv_thres:
            pmv_penl = 0.0;
        else:
            pmv_penl = (pmv_thres - pmv_raw) * stpt_violation_scl;
    pmv_penl = p_weight * pmv_penl;
    # energy reward
    hvac_energy = ob_next_prcd[BGR_IDX + TIMESTATE_LEN];
    egy_penl = e_weight*hvac_energy;
    # final reward
    reward = max(1 - egy_penl - pmv_penl, 0);
    return reward;

def rl_parametric_reward_part4_v2(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Reward = energy_heur + pmv_penal (consider pmv gradient)
    """
    PMV_IDX = 6;
    OCP_IDX = 7;
    BGR_IDX = 10;
    SPT_IDX = 4;
    IAT_IDX = 5;

    pmv_min = pcd_state_limits[0][PMV_IDX + TIMESTATE_LEN]
    pmv_max = pcd_state_limits[1][PMV_IDX + TIMESTATE_LEN]
    spt_min = pcd_state_limits[0][SPT_IDX + TIMESTATE_LEN]
    spt_max = pcd_state_limits[1][SPT_IDX + TIMESTATE_LEN]
    iat_min = pcd_state_limits[0][IAT_IDX + TIMESTATE_LEN]
    iat_max = pcd_state_limits[1][IAT_IDX + TIMESTATE_LEN]
    
    pmv_prcd = ob_next_prcd[PMV_IDX + TIMESTATE_LEN];
    pmv_prcd_last = ob_this_prcd[PMV_IDX + TIMESTATE_LEN];
    pmv_raw = pmv_min + pmv_prcd*(pmv_max - pmv_min);
    pmv_raw_last = pmv_min + pmv_prcd_last*(pmv_max - pmv_min);
    pmv_thres = -0.5;

    iat_prcd = ob_next_prcd[IAT_IDX + TIMESTATE_LEN];
    iat_prcd_last = ob_this_prcd[IAT_IDX + TIMESTATE_LEN];
    iat_raw = iat_min + iat_prcd*(iat_max - iat_min);
    iat_raw_last = iat_min + iat_prcd_last*(iat_max - iat_min);
    
    ocp = ob_next_prcd[OCP_IDX + TIMESTATE_LEN];
    # energy penalty
    hvac_energy_prcd = ob_next_prcd[BGR_IDX + TIMESTATE_LEN]; # [0, 1]
    egy_heur = 0;
    egy_penl_heur = min(max(e_weight*hvac_energy_prcd - egy_heur, 0), 1); # larger than 0
    # comfort penalty
    if ocp == 0:
        cmf_penl = min(max((19 - iat_raw) * stpt_violation_scl, 0),1); # [0, 1]
        cmf_heur = 0;
        cmf_penl_heur = min(max(cmf_penl - cmf_heur, 0), 1);
    else:
        cmf_penl = min(max((pmv_thres - pmv_raw) * p_weight, 0), 1); # [0, 1]
        pmv_heur = 0;
        cmf_penl_heur = min(max(cmf_penl - pmv_heur, 0), 1); # [0, 1]
    # optimize comfort only if comfort cannot be met
    penal_total = (1 - cmf_penl) * egy_penl_heur + cmf_penl * cmf_penl_heur;
    reward = 1 - penal_total;
    return reward;

def rl_parametric_reward_part4_v3(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Reward = energy_heur + pmv_penal (consider pmv gradient)
    """
    PMV_IDX = 6;
    OCP_IDX = 7;
    BGR_IDX = 10;
    SPT_IDX = 4;
    IAT_IDX = 5;

    pmv_min = pcd_state_limits[0][PMV_IDX + TIMESTATE_LEN]
    pmv_max = pcd_state_limits[1][PMV_IDX + TIMESTATE_LEN]
    spt_min = pcd_state_limits[0][SPT_IDX + TIMESTATE_LEN]
    spt_max = pcd_state_limits[1][SPT_IDX + TIMESTATE_LEN]
    iat_min = pcd_state_limits[0][IAT_IDX + TIMESTATE_LEN]
    iat_max = pcd_state_limits[1][IAT_IDX + TIMESTATE_LEN]
    
    pmv_prcd = ob_next_prcd[PMV_IDX + TIMESTATE_LEN];
    pmv_prcd_last = ob_this_prcd[PMV_IDX + TIMESTATE_LEN];
    pmv_raw = pmv_min + pmv_prcd*(pmv_max - pmv_min);
    pmv_raw_last = pmv_min + pmv_prcd_last*(pmv_max - pmv_min);
    pmv_thres = -0.5;

    iat_prcd = ob_next_prcd[IAT_IDX + TIMESTATE_LEN];
    iat_prcd_last = ob_this_prcd[IAT_IDX + TIMESTATE_LEN];
    iat_raw = iat_min + iat_prcd*(iat_max - iat_min);
    iat_raw_last = iat_min + iat_prcd_last*(iat_max - iat_min);
    
    ocp = ob_next_prcd[OCP_IDX + TIMESTATE_LEN];
    # energy penalty
    hvac_energy_prcd = ob_next_prcd[BGR_IDX + TIMESTATE_LEN]; # [0, 1]
    egy_heur = 0;
    egy_penl_heur = min(max(e_weight*hvac_energy_prcd - egy_heur, 0), 1); # larger than 0
    # comfort penalty
    if ocp == 0:
        cmf_penl = 0; # [0, 1]
        cmf_heur = 0;
        cmf_penl_heur = min(max(cmf_penl - cmf_heur, 0), 1);
    else:
        cmf_penl = min(max((pmv_thres - pmv_raw) * p_weight, 0), 1); # [0, 1]
        pmv_heur = 0;
        cmf_penl_heur = min(max(cmf_penl - pmv_heur, 0), 1); # [0, 1]
    # optimize comfort only if comfort cannot be met
    penal_total = (1 - cmf_penl) * egy_penl_heur + cmf_penl * cmf_penl_heur;
    reward = 1 - penal_total;
    return reward;

def rl_parametric_reward_part4_heuri_v1(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Reward = system_energy + pmv_penal
    """
    PMV_IDX = 6;
    OCP_IDX = 7;
    BGR_IDX = 9;

    pmv_min = pcd_state_limits[0][PMV_IDX + TIMESTATE_LEN]
    pmv_max = pcd_state_limits[1][PMV_IDX + TIMESTATE_LEN]
    act_stpt = action_this_prcd[0];
    # PMV_penl: penalize PMV smaller than -0.5
    pmv_prcd = ob_next_prcd[PMV_IDX + TIMESTATE_LEN];
    pmv_raw = pmv_min + pmv_prcd*(pmv_max - pmv_min);
    ocp = ob_next_prcd[OCP_IDX + TIMESTATE_LEN];
    pmv_penl = 0.0;
    pmv_thres = -0.5;
    if ocp == 0:
        pmv_penl = 0.0;
    else:
        if pmv_raw >= pmv_thres:
            pmv_penl = 0.0;
        else:
            pmv_penl = (pmv_thres - pmv_raw) * stpt_violation_scl;
    pmv_penl = p_weight * pmv_penl;
    # energy reward
    hvac_energy = ob_next_prcd[BGR_IDX + TIMESTATE_LEN];
    egy_penl = e_weight*hvac_energy;
    # energy heuristic
    egy_heur = (act_stpt - 19)/(26 - 19);
    # final reward
    reward = max(1 - egy_penl * 0.5 - egy_heur * 0.5 - pmv_penl, 0);
    return reward;

def rl_parametric_reward_part4_heuri_v2(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Reward = energy_heur + pmv_penal
    """
    PMV_IDX = 6;
    OCP_IDX = 7;
    BGR_IDX = 9;

    pmv_min = pcd_state_limits[0][PMV_IDX + TIMESTATE_LEN]
    pmv_max = pcd_state_limits[1][PMV_IDX + TIMESTATE_LEN]
    act_stpt = action_this_prcd[0];
    # PMV_penl: penalize PMV smaller than -0.5
    pmv_prcd = ob_next_prcd[PMV_IDX + TIMESTATE_LEN];
    pmv_raw = pmv_min + pmv_prcd*(pmv_max - pmv_min);
    ocp = ob_next_prcd[OCP_IDX + TIMESTATE_LEN];
    pmv_penl = 0.0;
    pmv_thres = -0.5;
    if ocp == 0:
        pmv_penl = 0.0;
    else:
        if pmv_raw >= pmv_thres:
            pmv_penl = 0.0;
        else:
            pmv_penl = (pmv_thres - pmv_raw) * stpt_violation_scl;
    pmv_penl = p_weight * pmv_penl;
    # energy heuristic
    egy_heur = (act_stpt - 19)/(26 - 19);
    # final reward
    reward = max(1 - egy_heur - pmv_penl, 0);
    return reward;

def rl_parametric_reward_part4_heuri_v3(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Reward = energy_heur + pmv_penal
    """
    PMV_IDX = 6;
    OCP_IDX = 7;
    BGR_IDX = 9;
    SPT_IDX = 4;
    IAT_IDX = 5;

    pmv_min = pcd_state_limits[0][PMV_IDX + TIMESTATE_LEN]
    pmv_max = pcd_state_limits[1][PMV_IDX + TIMESTATE_LEN]
    spt_min = pcd_state_limits[0][SPT_IDX + TIMESTATE_LEN]
    spt_max = pcd_state_limits[1][SPT_IDX + TIMESTATE_LEN]
    iat_min = pcd_state_limits[0][IAT_IDX + TIMESTATE_LEN]
    iat_max = pcd_state_limits[1][IAT_IDX + TIMESTATE_LEN]

    act_stpt = action_this_prcd[0];
    
    stpt_last_prcd = ob_this_prcd[SPT_IDX + TIMESTATE_LEN];
    stpt_last_raw = spt_min + stpt_last_prcd*(spt_max - spt_min);
    act_stpt_delta = act_stpt - stpt_last_raw;
    
    iat_prcd_last = ob_this_prcd[IAT_IDX + TIMESTATE_LEN];
    iat_raw_last = iat_min + iat_prcd_last*(iat_max - iat_min);
    iat_err_raw_last = stpt_last_raw - iat_raw_last;

    iat_prcd = ob_next_prcd[IAT_IDX + TIMESTATE_LEN];
    iat_raw = iat_min + iat_prcd*(iat_max - iat_min);
    iat_err_raw = act_stpt - iat_raw;
    
    pmv_prcd = ob_next_prcd[PMV_IDX + TIMESTATE_LEN];
    pmv_raw = pmv_min + pmv_prcd*(pmv_max - pmv_min);
    pmv_thres = -0.5;
    
    ocp = ob_next_prcd[OCP_IDX + TIMESTATE_LEN];
    hvac_energy_prcd = ob_next_prcd[BGR_IDX + TIMESTATE_LEN];
    egy_heur = 1 - (act_stpt - 19)/(26 - 19); # act_stpt==19 -> 1, act_stpt==26 -> 0
    egy_penl_heur = max(e_weight*hvac_energy_prcd - egy_heur, 0);

    reward = 0;
    # if not occupied, just optimize energy
    if ocp == 0:
        reward = 1 - egy_penl_heur;
    else:
        # if occupied and pmv ok, optimze energy
        if pmv_raw >= pmv_thres:
            reward = 1 - egy_penl_heur;
        else:
        # if occupied and pmv not ok, reward good actions
        # good actions include: increase stpt actions and
        # keep stpt action if the stpt cannot be met
            pmv_penl = min((pmv_thres - pmv_raw) * stpt_violation_scl, 1);
            # Determine the pmv heuristic
            pmv_heur = 0;
            if act_stpt_delta < 0: # The action is to decrease the stpt
                pmv_heur = 0; # bad action
            else:  # The action is to increase or keep the stpt
                if iat_err_raw_last > 0.5: # If the last stpt cannot be met
                    pmv_heur = 1; # good action
                else: # If the last stpt can be met
                    if act_stpt_delta == 0: # If the action is to keep the stpt
                        pmv_heur = 0; # bad action
                    else:
                        pmv_heur = 1; # good action
            pmv_penl_heur = pmv_penl - pmv_heur * p_weight;
            reward = 1 - pmv_penl_heur;
    reward = min(max(reward, 0), 1);
    return reward;

def rl_parametric_reward_part4_heuri_v4(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Reward = energy_heur + pmv_penal (consider pmv gradient)
    """
    PMV_IDX = 6;
    OCP_IDX = 7;
    BGR_IDX = 9;
    SPT_IDX = 4;

    pmv_min = pcd_state_limits[0][PMV_IDX + TIMESTATE_LEN]
    pmv_max = pcd_state_limits[1][PMV_IDX + TIMESTATE_LEN]
    spt_min = pcd_state_limits[0][SPT_IDX + TIMESTATE_LEN]
    spt_max = pcd_state_limits[1][SPT_IDX + TIMESTATE_LEN]

    act_stpt = action_this_prcd[0];
    
    pmv_prcd = ob_next_prcd[PMV_IDX + TIMESTATE_LEN];
    pmv_prcd_last = ob_this_prcd[PMV_IDX + TIMESTATE_LEN];
    pmv_raw = pmv_min + pmv_prcd*(pmv_max - pmv_min);
    pmv_raw_last = pmv_min + pmv_prcd_last*(pmv_max - pmv_min);
    pmv_thres = -0.5;
    
    ocp = ob_next_prcd[OCP_IDX + TIMESTATE_LEN];
    hvac_energy_prcd = ob_next_prcd[BGR_IDX + TIMESTATE_LEN];
    egy_heur = 1 - (act_stpt - 19)/(26 - 19); # act_stpt==19 -> 1, act_stpt==26 -> 0
    egy_penl_heur = max(e_weight*hvac_energy_prcd - egy_heur, 0);

    reward = 0;
    # if not occupied, just optimize energy
    if ocp == 0:
        reward = 1 - egy_penl_heur;
    else:
        # if occupied and pmv ok, optimze energy
        if pmv_raw >= pmv_thres:
            reward = 1 - egy_penl_heur;
        else:
        # if occupied and pmv not ok, reward good actions
        # good actions include: pmv is increasing
            pmv_penl = min((pmv_thres - pmv_raw) * stpt_violation_scl, 1);
            # Determine the pmv heuristic
            pmv_heur = max(pmv_raw - pmv_raw_last, 0);
            pmv_penl_heur = max(pmv_penl - pmv_heur * p_weight, 0);
            reward = 1 - pmv_penl_heur;
    reward = min(max(reward, 0), 1);
    return reward;

def rl_parametric_reward_part4_heuri_v5(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Reward = energy_heur + pmv_penal
    """
    PMV_IDX = 6;
    OCP_IDX = 7;
    BGR_IDX = 9;
    SPT_IDX = 4;
    IAT_IDX = 5;

    pmv_min = pcd_state_limits[0][PMV_IDX + TIMESTATE_LEN]
    pmv_max = pcd_state_limits[1][PMV_IDX + TIMESTATE_LEN]
    spt_min = pcd_state_limits[0][SPT_IDX + TIMESTATE_LEN]
    spt_max = pcd_state_limits[1][SPT_IDX + TIMESTATE_LEN]
    iat_min = pcd_state_limits[0][IAT_IDX + TIMESTATE_LEN]
    iat_max = pcd_state_limits[1][IAT_IDX + TIMESTATE_LEN]

    act_stpt = action_this_prcd[0];
    
    stpt_last_prcd = ob_this_prcd[SPT_IDX + TIMESTATE_LEN];
    stpt_last_raw = spt_min + stpt_last_prcd*(spt_max - spt_min);
    act_stpt_delta = act_stpt - stpt_last_raw;
    
    iat_prcd_last = ob_this_prcd[IAT_IDX + TIMESTATE_LEN];
    iat_raw_last = iat_min + iat_prcd_last*(iat_max - iat_min);
    iat_err_raw_last = stpt_last_raw - iat_raw_last;

    iat_prcd = ob_next_prcd[IAT_IDX + TIMESTATE_LEN];
    iat_raw = iat_min + iat_prcd*(iat_max - iat_min);
    iat_err_raw = act_stpt - iat_raw;
    
    pmv_prcd = ob_next_prcd[PMV_IDX + TIMESTATE_LEN];
    pmv_raw = pmv_min + pmv_prcd*(pmv_max - pmv_min);
    pmv_thres = -0.5;
    
    ocp = ob_next_prcd[OCP_IDX + TIMESTATE_LEN];
    hvac_energy_prcd = ob_next_prcd[BGR_IDX + TIMESTATE_LEN];
    egy_heur = 0;
    egy_penl_heur = max(e_weight*hvac_energy_prcd - egy_heur, 0);

    reward = 0;
    # if not occupied, just optimize energy
    if ocp == 0:
        reward = 1 - egy_penl_heur;
    else:
        # if occupied and pmv ok, optimze energy
        if pmv_raw >= pmv_thres:
            reward = 1 - egy_penl_heur;
        else:
        # if occupied and pmv not ok, reward good actions
        # good actions include: increase stpt actions and
        # keep stpt action if the stpt cannot be met
            pmv_penl = min((pmv_thres - pmv_raw) * stpt_violation_scl, 1);
            # Determine the pmv heuristic
            pmv_heur = 0;
            if act_stpt_delta < 0: # The action is to decrease the stpt
                pmv_heur = 0; # bad action
            else:  # The action is to increase or keep the stpt
                if iat_err_raw_last > 0.5: # If the last stpt cannot be met
                    pmv_heur = 1; # good action
                else: # If the last stpt can be met
                    if act_stpt_delta == 0: # If the action is to keep the stpt
                        pmv_heur = 0; # bad action
                    else:
                        pmv_heur = 1; # good action
            pmv_penl_heur = pmv_penl - pmv_heur * p_weight;
            reward = 1 - pmv_penl_heur;
    reward = min(max(reward, 0), 1);
    return reward;

def rl_parametric_reward_part4_heuri_v6(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Reward = energy_heur + pmv_penal (consider pmv gradient)
    """
    PMV_IDX = 6;
    OCP_IDX = 7;
    BGR_IDX = 9;
    SPT_IDX = 4;

    pmv_min = pcd_state_limits[0][PMV_IDX + TIMESTATE_LEN]
    pmv_max = pcd_state_limits[1][PMV_IDX + TIMESTATE_LEN]
    spt_min = pcd_state_limits[0][SPT_IDX + TIMESTATE_LEN]
    spt_max = pcd_state_limits[1][SPT_IDX + TIMESTATE_LEN]

    act_stpt = action_this_prcd[0];
    
    pmv_prcd = ob_next_prcd[PMV_IDX + TIMESTATE_LEN];
    pmv_prcd_last = ob_this_prcd[PMV_IDX + TIMESTATE_LEN];
    pmv_raw = pmv_min + pmv_prcd*(pmv_max - pmv_min);
    pmv_raw_last = pmv_min + pmv_prcd_last*(pmv_max - pmv_min);
    pmv_thres = -0.5;
    
    ocp = ob_next_prcd[OCP_IDX + TIMESTATE_LEN];
    hvac_energy_prcd = ob_next_prcd[BGR_IDX + TIMESTATE_LEN];
    egy_heur = 0;
    egy_penl_heur = max(e_weight*hvac_energy_prcd - egy_heur, 0);

    reward = 0;
    # if not occupied, just optimize energy
    if ocp == 0:
        reward = 1 - egy_penl_heur;
    else:
        # if occupied and pmv ok, optimze energy
        if pmv_raw >= pmv_thres:
            reward = 1 - egy_penl_heur;
        else:
        # if occupied and pmv not ok, reward good actions
        # good actions include: pmv is increasing
            pmv_penl = min((pmv_thres - pmv_raw) * stpt_violation_scl, 1);
            # Determine the pmv heuristic
            pmv_heur = max(pmv_raw - pmv_raw_last, 0);
            pmv_penl_heur = max(pmv_penl - pmv_heur * p_weight, 0);
            reward = 1 - pmv_penl_heur;
    reward = min(max(reward, 0), 1);
    return reward;

def rl_parametric_reward_part4_heuri_v7(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Reward = energy_heur + pmv_penal (consider pmv gradient)
    """
    PMV_IDX = 6;
    OCP_IDX = 7;
    BGR_IDX = 10;
    SPT_IDX = 4;
    IAT_IDX = 5;

    pmv_min = pcd_state_limits[0][PMV_IDX + TIMESTATE_LEN]
    pmv_max = pcd_state_limits[1][PMV_IDX + TIMESTATE_LEN]
    spt_min = pcd_state_limits[0][SPT_IDX + TIMESTATE_LEN]
    spt_max = pcd_state_limits[1][SPT_IDX + TIMESTATE_LEN]
    iat_min = pcd_state_limits[0][IAT_IDX + TIMESTATE_LEN]
    iat_max = pcd_state_limits[1][IAT_IDX + TIMESTATE_LEN]
    
    pmv_prcd = ob_next_prcd[PMV_IDX + TIMESTATE_LEN];
    pmv_prcd_last = ob_this_prcd[PMV_IDX + TIMESTATE_LEN];
    pmv_raw = pmv_min + pmv_prcd*(pmv_max - pmv_min);
    pmv_raw_last = pmv_min + pmv_prcd_last*(pmv_max - pmv_min);
    pmv_thres = -0.5;

    iat_prcd = ob_next_prcd[IAT_IDX + TIMESTATE_LEN];
    iat_prcd_last = ob_this_prcd[IAT_IDX + TIMESTATE_LEN];
    iat_raw = iat_min + iat_prcd*(iat_max - iat_min);
    iat_raw_last = iat_min + iat_prcd_last*(iat_max - iat_min);
    
    ocp = ob_next_prcd[OCP_IDX + TIMESTATE_LEN];
    # energy penalty
    hvac_energy_prcd = ob_next_prcd[BGR_IDX + TIMESTATE_LEN]; # [0, 1]
    egy_heur = 0;
    egy_penl_heur = min(max(e_weight*hvac_energy_prcd - egy_heur, 0), 1); # larger than 0
    # comfort penalty
    if ocp == 0:
        cmf_penl = min(max((19 - iat_raw) * 2, 0),1); # [0, 1]
        cmf_heur = max(iat_raw - iat_raw_last, 0);
        cmf_penl_heur = min(max(cmf_penl - cmf_heur * stpt_violation_scl, 0), 1);
    else:
        cmf_penl = min(max((pmv_thres - pmv_raw) * 2, 0), 1); # [0, 1]
        # if occupied and pmv not ok, reward good actions
        # good actions include: pmv is increasing
        pmv_heur = max(pmv_raw - pmv_raw_last, 0);
        cmf_penl_heur = min(max(cmf_penl - pmv_heur * p_weight, 0), 1); # [0, 1]
    # optimize comfort only if comfort cannot be met
    penal_total = 0.5 * egy_penl_heur + 0.5 * cmf_penl_heur;
    reward = 1 - penal_total;
    return reward;

def rl_parametric_reward_part4_prior_v1(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Deterministic schedule reward
    """
    WEEKDAY_IDX = 0;
    HOUR_IDX = 1;

    is_weekday = ob_next_prcd[WEEKDAY_IDX]
    now_hour = ob_next_prcd[HOUR_IDX] * 23.0;
    act_stpt = action_this_prcd[0];
    reward = 0;

    if is_weekday == 1:
        if now_hour <= 6.5:
            if act_stpt <= 20.0:
                reward = 1;
        elif now_hour > 6.5 and now_hour <= 19.5:
            if act_stpt >= 22.0:
                reward = 1;
        else:
            if act_stpt <= 20.0:
                reward = 1;
    else:
        if act_stpt <= 20.0:
            reward = 1;

    return reward;

def rl_parametric_reward_part4_prior_v2(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Reward = system_energy + pmv_penal
    """
    PMV_IDX = 6;
    OCP_IDX = 7;
    BGR_IDX = 9;

    pmv_min = pcd_state_limits[0][PMV_IDX + TIMESTATE_LEN]
    pmv_max = pcd_state_limits[1][PMV_IDX + TIMESTATE_LEN]
    # PMV_penl: penalize PMV smaller than -0.5
    pmv_prcd = ob_next_prcd[PMV_IDX + TIMESTATE_LEN];
    pmv_raw = pmv_min + pmv_prcd*(pmv_max - pmv_min);
    ocp = ob_next_prcd[OCP_IDX + TIMESTATE_LEN];
    pmv_penl = 0.0;
    pmv_thres = -0.5;
    if ocp == 0:
        pmv_penl = 0.0;
    else:
        if pmv_raw >= pmv_thres:
            pmv_penl = 0.0;
        else:
            pmv_penl = (pmv_thres - pmv_raw) * stpt_violation_scl;
    pmv_penl = p_weight * pmv_penl;
    # energy reward
    hvac_energy = ob_next_prcd[BGR_IDX + TIMESTATE_LEN];
    egy_penl = e_weight*hvac_energy;
    # final reward
    if ocp == 0:
        reward = max(1 - egy_penl, 0);
    else:
        reward = max(1 - pmv_penl, 0);
    return reward;

def rl_parametric_reward_part4_prior_v3(ob_this_prcd, action_this_prcd, ob_next_prcd, pcd_state_limits, e_weight, p_weight, stpt_violation_scl):
    """
    Reward = system_energy
    """
    PMV_IDX = 6;
    OCP_IDX = 7;
    BGR_IDX = 9;

    pmv_min = pcd_state_limits[0][PMV_IDX + TIMESTATE_LEN]
    pmv_max = pcd_state_limits[1][PMV_IDX + TIMESTATE_LEN]
    # energy reward
    hvac_energy = ob_next_prcd[BGR_IDX + TIMESTATE_LEN];
    egy_penl = e_weight*hvac_energy;
    # final reward
    reward = max(1 - egy_penl, 0);
    return reward;

def rl_parametric_metric_part4_v1(ob_next_raw, this_ep_energy, this_ep_comfort):
    """
    """ 
    PMV_IDX = 6;
    OCP_IDX = 7;
    BGR_IDX = 9;

    energy = ob_next_raw[BGR_IDX]; # W
    pmv = ob_next_raw[PMV_IDX]; # %
    ocp = ob_next_raw[OCP_IDX]; # 1 or 0

    this_ep_energy_toNow = this_ep_energy + energy; # Unit is kWh*timestep
    if ocp == 0:
        this_ep_comfort_toNow = this_ep_comfort + 0;
    else:
        this_ep_comfort_toNow = this_ep_comfort + pmv;
    
    return (this_ep_energy_toNow, this_ep_comfort_toNow);

def rl_parametric_metric_part4_v2(ob_next_raw, this_ep_energy, this_ep_comfort):
    """
    """ 
    PMV_IDX = 6;
    OCP_IDX = 7;
    BGR_IDX = 10;

    energy = ob_next_raw[BGR_IDX]; # W
    pmv = ob_next_raw[PMV_IDX]; # %
    ocp = ob_next_raw[OCP_IDX]; # 1 or 0

    this_ep_energy_toNow = this_ep_energy + energy; # Unit is kWh*timestep
    if ocp == 0:
        this_ep_comfort_toNow = this_ep_comfort + 0;
    else:
        this_ep_comfort_toNow = this_ep_comfort + pmv;
    
    return (this_ep_energy_toNow, this_ep_comfort_toNow);

def _is_chiller_short_cycle(ob_this_prcd, ob_next_prcd, CHILLERX_SHTCY_IDX, CHILLERX_ONOFF_IDX):
    chillerX_cycle_this = ob_this_prcd[CHILLERX_SHTCY_IDX + TIMESTATE_LEN];
    chillerX_onoff_this = ob_this_prcd[CHILLERX_ONOFF_IDX + TIMESTATE_LEN];
    chillerX_onoff_next = ob_next_prcd[CHILLERX_ONOFF_IDX + TIMESTATE_LEN];
    chillerX_is_state_change = abs(chillerX_onoff_next - chillerX_onoff_this);

    if chillerX_cycle_this < 1.0 and chillerX_is_state_change == 1.0:
        return True;
    else:
        return False;

reward_func_dict = {'1': err_energy_reward_iw,
                    '2': err_energy_reward_iw_v2,
                    '3': err_energy_reward_iw_v3,
                    '4': err_energy_reward_iw_v4,
                    '5': err_energy_reward_iw_v5,
                    '6': err_energy_reward_iw_v6,
                    '7': ppd_energy_reward_iw_timeRelated,
                    '8': ppd_energy_reward_iw_timeRelated_v2,
                    '9': ppd_energy_reward_iw_timeRelated_v3,
                    '10': ppd_energy_reward_iw_timeRelated_v4,
                    '11': ppd_energy_reward_iw_timeRelated_v5,
                    '12': ppd_energy_reward_iw_timeRelated_v6,
                    '13': ppd_energy_reward_iw_timeRelated_v7,
                    '14': ppd_energy_reward_iw_timeRelated_v8,
                    '15': ppd_energy_reward_iw_timeRelated_v9,
                    'cslDxCool_1': stptVio_energy_reward_cslDxCool_v1,
                    'cslDxCool_2': stptVio_energy_reward_cslDxCool_v2,
                    'part1_v1': stpt_viol_energy_reward_part1_v1,
                    'part2_v1': stpt_viol_energy_reward_part2_v1,
                    'part3_v1': rl_parametric_reward_part3_v1,
                    'part3_v2': rl_parametric_reward_part3_v2,
                    'part3_v3': rl_parametric_reward_part3_v3,
                    'part3_v4': rl_parametric_reward_part3_v4,
                    'part4_v1': rl_parametric_reward_part4_v1,
                    'part4_prior_v1': rl_parametric_reward_part4_prior_v1,
                    'part4_prior_v2': rl_parametric_reward_part4_prior_v2,
                    'part4_prior_v3': rl_parametric_reward_part4_prior_v3,
                    'part4_heuri_v1': rl_parametric_reward_part4_heuri_v1,
                    'part4_heuri_v2': rl_parametric_reward_part4_heuri_v2,
                    'part4_heuri_v3': rl_parametric_reward_part4_heuri_v3,
                    'part4_heuri_v4': rl_parametric_reward_part4_heuri_v4,
                    'part4_heuri_v5': rl_parametric_reward_part4_heuri_v5,
                    'part4_heuri_v6': rl_parametric_reward_part4_heuri_v6,
                    'part4_heuri_v7': rl_parametric_reward_part4_heuri_v7,
                    'part4_v2': rl_parametric_reward_part4_v2,
                    'part4_v3': rl_parametric_reward_part4_v3}

metric_func_dict = {
                    'cslDxCool_1': stptVio_energy_metric_cslDxCool_v1,
                    'cslDxCool_2': stptVio_energy_metric_cslDxCool_v2,
                    'part1_v1': stpt_viol_energy_metric_part1_v1,
                    'part2_v1': stpt_viol_energy_metric_part2_v1,
                    'part3_v1': rl_parametric_metric_part3_v1,
                    'part4_v1': rl_parametric_metric_part4_v1,
                    'part4_v2': rl_parametric_metric_part4_v2}

