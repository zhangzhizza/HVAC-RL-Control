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
    normalized_err = min(abs(normalized_zat - normalized_zatssp) * err_penalty_scl, 1.0); 
    ret = - (e_weight * normalized_hvac_energy + p_weight * normalized_err);
    return ret;

def err_energy_reward_iw(ob_next_prcd, e_weight, p_weight, err_penalty_scl):
    """
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
    normalized_err = min(abs(normalized_zat - normalized_zatssp) * err_penalty_scl, 1.0); 
    ret = - (e_weight * normalized_hvac_energy + p_weight * normalized_err);
    return ret;

def err_energy_reward_iw_v2(ob_next_prcd, e_weight, p_weight, err_penalty_scl):
    """
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
        ret = - (e_weight * normalized_hvac_energy + p_weight * normalized_err);
    return ret;