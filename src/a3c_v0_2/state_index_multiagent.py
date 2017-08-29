"""
State feature index for multiagent agent case.
"""
# Building level state 
OADT_RAW_IDX = 0; # Outdoor air temperature
OARH_RAW_IDX = 1; # Outdoor air RH
WS_RAW_IDX = 2; # Wind speed
WD_RAW_IDX = 3; # Wind direction
DFS_RAW_IDX = 4; # Diffuse solar radiation
DIS_RAW_IDX = 5; # Direct solar radiation
HVACE_RAW_IDX = 6; # HVAC electric power demand
# Zone level state
HTSP_RAW_IDX = 0; # Indoor heating setpoint
CLSP_RAW_IDX = 1; # Indoor cooling setpoint
ZADT_RAW_IDX = 2; # Zone air temperature
ZARH_RAW_IDX = 3; # Zone air RH
ZPPD_RAW_IDX = 4; # Zone PPD

ZN_OB_NUM = 5 # Five elements are related to zone observation
BD_OB_NUM = 7 # Seven elements are related to building level observation