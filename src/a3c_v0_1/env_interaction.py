"""
This file contains the wrapper classes for environment interactions. 
"""

class IWEnvInteract(object):

	def __init__(self, env, ob_state_process_func):
		self._env = env;
		self._ob_state_process_func = ob_state_process_func;

	def reset(self):
		
		print ('enter reset!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
		return self._interact(mode = 'reset', actions = None);

	def step(self, actions):

		return self._interact(mode = 'step', actions = actions);

	def _interact(self, mode, actions = None):
		ret = [];
		# Reset the env
		forecast = None;
		env_get = None; 
		time = None;
		ob_raw = None;
		is_terminal = None;
		if mode == 'reset':
			print ('before reseted!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
			env_get = self._env.reset();
			print ('reseted!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
		elif mode == 'step':
			env_get = self._env.step(actions);
		if len(env_get) == 4:
			time, ob_raw, forecast, is_terminal = env_get;
		elif len(env_get) == 3:
			time, ob_raw, is_terminal = env_get;
        # Process and normalize the raw observation
		ob_raw = self._ob_state_process_func(ob_raw);
		if forecast is not None:
        	# Add forecast info to ob_this_raw so they can be normalized
			ob_raw.extend(forecast);
		ret.append(time);
		ret.append(ob_raw);
		ret.append(is_terminal);
		return ret;

