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
