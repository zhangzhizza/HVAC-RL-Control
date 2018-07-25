def raw_state_process_smlRefBld(raw_state):
	"""
	This processor change the occupant count to occupancy status.

	Args:
		raw_state: python list, 1-D.
	"""
	ZPCT_RAW_IDX = 11;
	if raw_state[ZPCT_RAW_IDX] > 0:
		raw_state[ZPCT_RAW_IDX] = 1;
	else:
		raw_state[ZPCT_RAW_IDX] = 0;
	return raw_state;

def raw_state_process_iw(raw_state):
	"""
	This processor change the occupant count to occupancy status.

	Args:
		raw_state: python list, 1-D.
	"""
	return raw_state;


def raw_state_process_cslDx_1(raw_state):
	"""
	This processor does nothing.

	Args:
		raw_state: python list, 1-D.
	"""
	return raw_state;


def raw_state_process_cslDx_2(raw_state):
	"""
	This processor repaces the raw temps and cool stpts into the max stpt violation.

	Args:
		raw_state: python list, 1-D.

	Ret:
		processed raw state, python list, 1-D:
		[OAT, RH, DifSol, DirSol, coolStptMaxViol, Energy]
	"""
	ZONE_NUM = 22;
    IAT_FIRST_RAW_IDX = 4;
    IATSSP_FIRST_RAW_IDX = 26;
    iats = np.array(raw_state[IAT_FIRST_RAW_IDX: IAT_FIRST_RAW_IDX + ZONE_NUM]);
    iatssp = np.array(raw_state[IATSSP_FIRST_RAW_IDX: IATSSP_FIRST_RAW_IDX + ZONE_NUM]);
    sspVio_max = max(iats - iatssp); # For cooling, the IAT should be less than the IATSSP
    
    ret = [raw_state[0], raw_state[1], raw_state[2], raw_state[3], sspVio_max, raw_state[-1]];
    
    return ret;


def raw_stateLimit_process_cslDx_1(raw_stateLimit):
	"""
	This processor is used with raw_state_process_cslDx_2.
	"""
	return raw_stateLimit;


def raw_stateLimit_process_cslDx_2(raw_stateLimit):
	"""
	This processor is used with raw_state_process_cslDx_2.
	"""
	ret = [raw_stateLimit[0], raw_stateLimit[1], raw_stateLimit[2], raw_stateLimit[3], 
			raw_stateLimit[4], raw_stateLimit[-1]];
	return ret;


raw_state_process_map = {'cslDx_1': [raw_state_process_cslDx_1, raw_stateLimit_process_cslDx_1],
						 'cslDx_2': [raw_state_process_cslDx_2, raw_stateLimit_process_cslDx_2]}