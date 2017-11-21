"""
Eplus gym environment weather forecast artificial randomness std.
"""
forecastRandStd_1 = [1.0, # OA Forecast
	                 10.0, # RH Forecast
	                 50.0,  # DirS Forecast
	                 50.0] # DifS Forecast 

min_max_limits_dict = {'IW-tmy3Weather-v9601': iw_v9601_limits,
					   'IW-realWeather-v9601': iw_v9601_limits};
