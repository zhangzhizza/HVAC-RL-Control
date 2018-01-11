#!python
"""
This is the entry script for the A3C HVAC control algorithm. 

Run EnergyPlus Environment with Asynchronous Advantage Actor Critic (A3C).

Algorithm taken from https://arxiv.org/abs/1602.01783
'Asynchronous Methods for Deep Reinforcement Learning'

Author: Zhiang Zhang
Last update: Aug 28th, 2017

"""
from main_args import *
from a3c_v0_1.reward_funcs import reward_func_dict
from a3c_v0_1.action_funcs import act_func_dict
from a3c_v0_1.raw_state_processors import raw_state_process_iw

def main():
    # Common args
    parser = get_args();
    # Specific args
    parser.add_argument('--err_penalty_scl', default=0.15, type=float,
                        help='Scale PPD penalty, default is 0.15.')
    parser.add_argument('--violation_penalty_scl', default=10.0, type=float,
                        help='Scale temperature setpoint violation error, default is 10.0.')
    parser.add_argument('--train_act_func', default='1', type=str,
                        help='The action function corresponding to the action space, default is 1, '
                             'corresponding to the actions space iw_1')
    parser.add_argument('--eval_act_func', default='1', type=str,
                        help='The action function corresponding to the action space, default is 1, '
                             'corresponding to the actions space iw_1')
    parser.add_argument('--reward_func', default='1', type=str)
    args = parser.parse_args();
    # Prepare case specific args
    reward_func = reward_func_dict[args.reward_func]
    rewardArgs = [args.err_penalty_scl, args.violation_penalty_scl];
    train_action_func = act_func_dict[args.train_act_func][0];
    train_action_limits = act_func_dict[args.train_act_func][1];
    eval_action_func = act_func_dict[args.eval_act_func][0];
    eval_action_limits = act_func_dict[args.eval_act_func][1];
    raw_state_process_func = raw_state_process_iw;
    effective_main(args, reward_func, rewardArgs, train_action_func, eval_act_func, train_action_limits, eval_action_limits, raw_state_process_func);
        

if __name__ == '__main__':
    main()

"""
if args.act_func == '1':
      action_func = mull_stpt_iw;
      action_limits = act_limits_iw_1;
    elif args.act_func == '2':
      action_func = mull_stpt_oaeTrans_iw;
      action_limits = act_limits_iw_2;
    elif args.act_func == '3':
      action_func = mull_stpt_noExpTurnOffMullOP;
      action_limits = act_limits_iw_2
    elif args.act_func == '4':
      action_func = mull_stpt_directSelect;
      action_limits = act_limits_iw_2
    elif args.act_func == '5':
      action_func = iw_iat_stpt_noExpHeatingOp;
      action_limits = act_limits_iw_3;
    elif args.act_func == '6':
      action_func = iw_iat_stpt_noExpHeatingOp;
      action_limits = act_limits_iw_4;
"""