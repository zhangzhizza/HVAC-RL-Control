"""
Eplus gym environment states range.
"""
iw_v9601_limits = [(-13.0, 26.0), # OA
                   ( 0.0, 100.0), # RH
	               ( 0.0, 11.0),  # WS
	               ( 0.0, 360.0), # WD
	               ( 0.0, 378.0), # DifS
	               ( 0.0, 1000),  # DirS 
	               ( -30.0, 30.0),  # OAESSPs
	               ( 0.0, 100.0), # PPD
	               ( 18.0, 25.0), # IATSSP
	               ( 18.0, 25.0), # IAT
	               ( 18.0, 25.0), # IAT Logics
	               ( 0.0,  1.0), # Occupy flag
	               ( 0.0, 85.0), # HTDMD
	               (-13.0, 26.0), # OA Forecast
	               ( 0.0, 100.0), # RH Forecast
	               ( 0.0, 1000),  # DirS Forecast
	               ( 0.0, 378.0), # DifS Forecast
	               (-13.0, 26.0), # OA Forecast
	               ( 0.0, 100.0), # RH Forecast
	               ( 0.0, 1000),  # DirS Forecast
	               ( 0.0, 378.0), # DifS Forecast
	               (-13.0, 26.0), # OA Forecast
	               ( 0.0, 100.0), # RH Forecast
	               ( 0.0, 1000),  # DirS Forecast
	               ( 0.0, 378.0), # DifS Forecast
	               (-13.0, 26.0), # OA Forecast
	               ( 0.0, 100.0), # RH Forecast
	               ( 0.0, 1000),  # DirS Forecast
	               ( 0.0, 378.0), # DifS Forecast
	               (-13.0, 26.0), # OA Forecast
	               ( 0.0, 100.0), # RH Forecast
	               ( 0.0, 1000),  # DirS Forecast
	               ( 0.0, 378.0), # DifS Forecast
	               (-13.0, 26.0), # OA Forecast
	               ( 0.0, 100.0), # RH Forecast
	               ( 0.0, 1000),  # DirS Forecast
	               ( 0.0, 378.0), # DifS Forecast
	               (-13.0, 26.0), # OA Forecast
	               ( 0.0, 100.0), # RH Forecast
	               ( 0.0, 1000),  # DirS Forecast
	               ( 0.0, 378.0), # DifS Forecast
	               (-13.0, 26.0), # OA Forecast
	               ( 0.0, 100.0), # RH Forecast
	               ( 0.0, 1000),  # DirS Forecast
	               ( 0.0, 378.0), # DifS Forecast
	               (-13.0, 26.0), # OA Forecast
	               ( 0.0, 100.0), # RH Forecast
	               ( 0.0, 1000),  # DirS Forecast
	               ( 0.0, 378.0), # DifS Forecast
	               (-13.0, 26.0), # OA Forecast
	               ( 0.0, 100.0), # RH Forecast
	               ( 0.0, 1000),  # DirS Forecast
	               ( 0.0, 378.0), # DifS Forecast
	               (-13.0, 26.0), # OA Forecast
	               ( 0.0, 100.0), # RH Forecast
	               ( 0.0, 1000),  # DirS Forecast
	               ( 0.0, 378.0), # DifS Forecast
	               (-13.0, 26.0), # OA Forecast
	               ( 0.0, 100.0), # RH Forecast
	               ( 0.0, 1000),  # DirS Forecast
	               ( 0.0, 378.0)] # DifS Forecast 

iw_v9602_limits = [(-13.0, 26.0), # OA 0
                   ( 0.0, 100.0), # RH 1
	               ( 0.0, 11.0),  # WS 2
	               ( 0.0, 360.0), # WD 3
	               ( 0.0, 378.0), # DifS 4
	               ( 0.0, 1000),  # DirS 5
	               ( -30.0, 30.0),  # OAESSPs 6
	               ( 0.0, 100.0), # PPD 7
	               ( 18.0, 25.0), # IATSSP 8
	               ( 18.0, 25.0), # IAT 9
	               ( 18.0, 25.0), # IAT Logics 10
	               ( 0.0,  1.0), # Occupy flag 11
	               ( 0.0, 85.0)] # HTDMD 12

iw_v9706_limits = [(-13.0, 26.0), # OA
                   ( 0.0, 100.0), # RH
	               ( 0.0, 11.0),  # WS
	               ( 0.0, 360.0), # WD
	               ( 0.0, 378.0), # DifS
	               ( 0.0, 1000),  # DirS 
	               ( -30.0, 30.0),  # OAESSPs
	               ( 0.0, 100.0), # PPD
	               ( 20.0, 65.0), # HWSSP
	               ( 18.0, 25.0), # IAT
	               ( 18.0, 25.0), # IAT Logics
	               ( 0.0,  1.0), # Occupy flag
	               ( 0.0, 85.0)] # HTDMD 

iw_fore_v9706_limits = [(-13.0, 26.0), # OA
                   ( 0.0, 100.0), # RH
	               ( 0.0, 11.0),  # WS
	               ( 0.0, 360.0), # WD
	               ( 0.0, 378.0), # DifS
	               ( 0.0, 1000),  # DirS 
	               ( -30.0, 30.0),  # OAESSPs
	               ( 0.0, 100.0), # PPD
	               ( 20.0, 65.0), # HWSSP
	               ( 18.0, 25.0), # IAT
	               ( 18.0, 25.0), # IAT Logics
	               ( 0.0,  1.0), # Occupy flag
	               ( 0.0, 85.0), # HTDMD
	               (-13.0, 26.0), # OA Forecast
	               ( 0.0, 100.0), # RH Forecast
	               ( 0.0, 1000),  # DirS Forecast
	               ( 0.0, 378.0), # DifS Forecast
	               (-13.0, 26.0), # OA Forecast
	               ( 0.0, 100.0), # RH Forecast
	               ( 0.0, 1000),  # DirS Forecast
	               ( 0.0, 378.0), # DifS Forecast
	               (-13.0, 26.0), # OA Forecast
	               ( 0.0, 100.0), # RH Forecast
	               ( 0.0, 1000),  # DirS Forecast
	               ( 0.0, 378.0), # DifS Forecast
	               (-13.0, 26.0), # OA Forecast
	               ( 0.0, 100.0), # RH Forecast
	               ( 0.0, 1000),  # DirS Forecast
	               ( 0.0, 378.0), # DifS Forecast
	               (-13.0, 26.0), # OA Forecast
	               ( 0.0, 100.0), # RH Forecast
	               ( 0.0, 1000),  # DirS Forecast
	               ( 0.0, 378.0), # DifS Forecast
	               (-13.0, 26.0), # OA Forecast
	               ( 0.0, 100.0), # RH Forecast
	               ( 0.0, 1000),  # DirS Forecast
	               ( 0.0, 378.0),] # DifS Forecast

fiveZ_tmy3Weather_v1 = [];

min_max_limits_dict = {'IW-tmy3Weather-v9601': iw_v9601_limits,
					   'IW-realWeather-v9601': iw_v9601_limits,
					   'IW-tmy3Weather-v9602': iw_v9602_limits,
					   'IW-realWeather-v9602': iw_v9602_limits,
					   'IW-tmy3Weather-v9603': iw_v9601_limits,
					   'IW-realWeather-v9603': iw_v9601_limits,
					   'IW-tmy3Weather-v9604': iw_v9602_limits,
					   'IW-realWeather-v9604': iw_v9602_limits,
					   'IW-tmy3Weather-v9605': iw_v9601_limits,
					   'IW-realWeather-v9605': iw_v9601_limits,
					   'IW-tmy3Weather-v9606': iw_v9602_limits,
					   'IW-realWeather-v9606': iw_v9602_limits,
					   'IW-tmy3Weather-v9706': iw_v9706_limits,
					   'IW-realWeather-v9706': iw_v9706_limits,
					   'IW-tmy3Weather-v9701': iw_v9706_limits,
					   'IW-realWeather-v9701': iw_v9706_limits,
					   'IW-tmy3Weather-fore-v9706': iw_fore_v9706_limits,
					   'IW-realWeather-fore-v9706': iw_fore_v9706_limits,
					   '5z-tmy3Weather-v1': fiveZ_tmy3Weather_v1};
