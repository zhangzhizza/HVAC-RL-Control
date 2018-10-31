import subprocess, os

dev_ins = '701104' # Device instance of the controller
obj_type = '2' # 2 is the code for analog value. Can be found in a file bacenum.h of the stack software
obj_ins = '31' # Object instance for the IAT setpoint
prop = '85' # 85 is the code for the present value (in bacenum.h)
prio = '16' # Priority of the value, 8 is usually used for manual overwriting
idx = '-1' # not application
tag = '4' # 4 is the real value (in bacenum.h)
value = '75' # The actual value needs to be written (IW data use IP unit)

fd = os.path.dirname(os.path.realpath(__file__));

print (subprocess.run([fd + '/bacnet-stack-0.8.5/bin/bacwp',\
				dev_ins, obj_type, obj_ins, prop, prio, idx, tag, value], 
				stdout=subprocess.PIPE).stdout.decode());