# eplus-env

This environment wraps the EnergyPlus-v-8-6 into the OpenAI gym environment interface. 
### Installation
EnergyPlus is platform dependent. Current version only works in Linux OS. If Mac, download
the EnergyPlus-v-8-6 from https://energyplus.net/downloads, extract it, and replace the 
eplus_env/envs/EnergyPlus-8-6-0 with it. 

The environment depends on BCVTB-1.6.0 (https://simulationresearch.lbl.gov/bcvtb). There 
is no need to re-install it since this repository already had it. But BCVTB-1.6.0 is compiled
with Java-1.8. Make sure you have Java-1.8 on your OS. 

```sh
$ cd eplus-env
$ pip install -e .
```
### Usage
#### Overview
The environment wraps a one-story 5-zone office EnergyPlus model. It is assumed the building is located in
Pittsburgh, PA, USA. The HAVC system is centralized VAV with terminal reheat. The building has one air handling
unit serving all five zones, and the primary heating/cooling source is DX-coil. Simulation time step is 900 seconds
, and running period for one episode is Jan 1st 00:00:00 to Mar 31st 24:00:00. For more info about the 
building model, please refer to the top part of eplus_env/envs/5ZoneAutoDXVAV.idf file. 
#### Action input and environment output
There are two environments, Eplus-v0 and Eplus-forecast-v0. Both environments take a 1-D python list of float with
length 2 as action input, representing the heating setpoint and cooling setpoint (in order) in degree Celsius of 
the south zone. The ouput from reset or step function is different for the two environments. Eplus-v0 outputs a 
1-D python tuple of [curSimTime, sensorOut, isTerminal] and Eplus-forecast-v0 outputs a 1-D python tuple of
[curSimTime, sensorOut, weatherForecast, isTerminal]. 
⋅⋅* curSimTime: current simulation time counting in seconds from Jan 1st 00:00:00.
..* sensorOut: 1-D python list of float [Site Outdoor Air Drybulb Temperature (C), 
Site Outdoor Air Relative Humidity (%), Site Wind Speed (m/s), Site Wind Direction (degree from north), 
Site Diffuse Solar Radiation Rate per Area (W/m2), Site Direct Solar Radiation Rate per Area (W/m2), 
Zone Air Temperature (C), Zone Air Relative Humidity (%), Zone Thermostat Heating Setpoint Temperature (C), 
Zone Thermostat Cooling Setpoint Temperature (C), Zone Thermal Comfort Fanger Model PMV, Zone People Occupant Count, 
Facility Total HVAC Electric Demand Power (W)]. **Note**: Zone People Occupant Count is here just for an indication
whether the zone is occupied or not; don not use it directly please actual number of people in a room in reality 
is very hard to detect (but occupancy or not is easy). 
..* weatherForecast: 2-D python list with shape (36, 28) where row x is the weather forecast information for 
the x time steps ahead of the curSimTime, where columns are the weather variables and the order is 
Dry Bulb Temperature {C},Dew Point Temperature
{C},Relative Humidity {%},Atmospheric Pressure {Pa},Extraterrestrial Horizontal Radiation
{Wh/m2},Extraterrestrial Direct Normal Radiation {Wh/m2},Horizontal Infrared
Radiation Intensity from Sky {Wh/m2},Global Horizontal Radiation {Wh/m2},Direct
Normal Radiation {Wh/m2},Diffuse Horizontal Radiation {Wh/m2},Global Horizontal
Illuminance {lux},Direct Normal Illuminance {lux},Diffuse Horizontal Illuminance
{lux},Zenith Luminance {Cd/m2},Wind Direction {deg},Wind Speed {m/s},Total Sky
Cover {.1},Opaque Sky Cover {.1},Visibility {km},Ceiling Height {m},Present Weather
Observation,Present Weather Codes,Precipitable Water {mm},Aerosol Optical Depth
{.001},Snow Depth {cm},Days Since Last Snow,Albedo {.01},Liquid Precipitation Depth
{mm},Liquid Precipitation Quantity {hr}.
..* isTerminal: whether the current episode finishs or not. When the simulation time reaches the end of the 
EnergyPlus run period (Mar 31st 24:00:00), the episode ends. 
#### Running output
EnergyPlus logs its own output. The output will be stored under the directory $pwd/Eplus-env-runX/Eplus-env-sub_runX/output.
The "sub_run" directory is the directory for each episode that the environment runs. 
#### Example

```python
import gym;
import eplus_env;

env = gym.make('Eplus-v0');
curSimTime, ob, isTerminal = env.reset(); # Reset the env (creat the EnergyPlus subprocess)
while not isTerminal:
    action = someFuncToGetAction(curSimTime, ob); # Should return a python list of float with len 2
    curSimTime, ob, isTerminal = env.step(action);
curSimTime, ob, isTerminal = env.reset(); # Reset the env (creat the EnergyPlus subprocess)
while not isTerminal:
    action = someFuncToGetAction(curSimTime, ob); # Should return a python list of float with len 2
    curSimTime, ob, isTerminal = env.step(action);
                  
env.end_env(); # Safe end the environment after use. 
