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

actions_htcl_cmd_exp1 = [( 1.0,  1.0),
                    (-1.0, -1.0),
                    (-1.0,  1.0),
                    ( 2.0,  2.0),
                    (-2.0, -2.0),
                    (-2.0,  2.0),
                    ( 3.0,  3.0),
                    (-3.0, -3.0),
                    (-3.0,  3.0),
                    ( 0.0,  0.0)];

actions_htcl_cmd_exp2 = [( 1.0,  1.0),
                    (-1.0, -1.0),
                    (-1.0,  1.0),
                    ( 2.0,  2.0),
                    (-2.0, -2.0),
                    (-2.0,  2.0),
                    ( 5.0,  5.0),
                    (-5.0, -5.0),
                    (-5.0,  5.0),
                    ( 0.0,  0.0)];

actions_htcl_cmd_exp3 = [( 1.0,  1.0),
                    (-1.0, -1.0),
                    (-1.0,  1.0),
                    ( 0.5,  0.5),
                    (-0.5, -0.5),
                    (-0.5,  0.5),
                    ( 0.0,  0.0)];
action_map = {'default': actions_htcl_cmd, 'exp1': actions_htcl_cmd_exp1, 'exp2':actions_htcl_cmd_exp2, 'exp3': actions_htcl_cmd_exp3}
