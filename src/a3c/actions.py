"""Actions for the environment. A 2-D python list, where each row is one
action choice. For each row, the index 0 is the action number and index 1
is a tuple showing the delta change for the current heating and cooling
indoor air temperature setpoint. """

actions_delta_stpt = [( 0.5, -0.5),
           ( 0.5,  0.0),
           ( 0.5,  0.5),
           ( 0.0, -0.5),
           ( 0.0,  0.0),
           ( 0.0,  0.5),
           (-0.5, -0.5),
           (-0.5,  0.0),
           (-0.5,  0.5),
           ( 1.0, -1.0),
           ( 1.0,  0.0),
           ( 1.0,  1.0),
           ( 0.0, -1.0),
           ( 0.0,  1.0),
           (-1.0, -1.0),
           (-1.0,  0.0),
           (-1.0,  1.0)]

actions_htcl_cmd = [( 1.0,  1.0),
                    (-1.0, -1.0),
                    (-1.0,  1.0),
                    ( 0.0,  0.0)];