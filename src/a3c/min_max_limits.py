"""The minimum and maximum limits for each state feature, in the order of
        0. Day of the week (0-6, 0 is Monday)
        1. Hour of the day (0-23)
        2. Site Outdoor Air Drybulb Temperature (C)
        3. Site Outdoor Air Relative Humidity (%)
        4. Site Wind Speed (m/s)
        5. Site Wind Direction (degree from north)
        6. Site Diffuse Solar Radiation Rate per Area (W/m2)
        7. Site Direct Solar Radiation Rate per Area (W/m2)
        8. Zone Air Temperature (C)
        9. Zone Air Relative Humidity (%)
        10. Zone Thermostat Heating Setpoint Temperature (C)
        11. Zone Thermostat Cooling Setpoint Temperature (C)
        12. Zone Thermal Comfort Fanger Model PMV
        13. Zone People Occupancy status (0 or 1)
        14. Facility Total HVAC Electric Demand Power (W)
"""
min_max_limits = [[0.0, 0.0,  -16.7, 0.0,   0.0,  0.0,   0.0,   0.0,   15.0, 0.0,   15.0, 15.0, -3.0, 0.0, 0.0   ],
                  [6.0, 23.0, 26.0,  100.0, 23.1, 360.0, 389.0, 905.0, 30.0, 100.0, 30.0, 30.0, 3.0,  1.0, 6000.0]]

stpt_limits = (15.0, 30.0);