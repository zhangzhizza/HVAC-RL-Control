from util.time import get_time_from_seconds

class BuildingOptStatus(object):
	"""
	An abstract class to determine the building operation status.

	"""
	def __init__(self, env_st_year, env_st_mon, env_st_day, env_st_weekday):
		"""
		env_st_year: int
			The environment simulation start year
		env_st_mon: int
			The environment simulation start month, 1 - 12
		env_st_day: int
			The environment simulation start day of the month, start from 1
		env_st_weekday: int
			The environment simulation start weekday, 0 is monday
		"""
		self._env_st_year = env_st_year;
		self._env_st_mon = env_st_mon;
		self._env_st_day = env_st_day;
		self._env_st_weekday = env_st_weekday;

	def get_is_opt(self, sim_time, state_obs):
		"""
		Return whether the building is in operation.

		sim_time: int
			The seconds from the simulation start time.
		state_obs: python list
			The raw environment state observation.

		Return: bool
		"""
		raise NotImplementedError;


class BuildingWeekdayPatOpt(BuildingOptStatus):
	"""
	A class determine the building operation status depending on the regular 
	weekday patterns, i.e. Monday to Friday 8:00 to 18:00
	"""
	def __init__(self, env_st_year, env_st_mon, env_st_day, env_st_weekday):
		super(BuildingWeekdayPatOpt, self).__init__(env_st_year, env_st_mon, 
													env_st_day, env_st_weekday);
		self._bldStTime = 8;
		self._bldEdTime = 18;

	def get_is_opt(self, sim_time, state_obs):
		nowWeekday, nowHour = get_time_from_seconds(sim_time, self._env_st_year, 
                                                    self._env_st_mon, 
                                                    self._env_st_day, 
                                                    self._env_st_weekday);
		if (nowWeekday < 5 and nowHour >= self._bldStTime and nowHour <= self._bldEdTime): 
			return True;
		else:
			return False;

		
