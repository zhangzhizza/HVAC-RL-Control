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
from a3c_v0_1.reward_funcs import err_energy_reward_iw, err_energy_reward_iw_v2, \
                                  err_energy_reward_iw_v3, err_energy_reward_iw_v4, \
                                  err_energy_reward_iw_v5, err_energy_reward_iw_v6, \
                                  ppd_energy_reward_iw_timeRelated, ppd_energy_reward_iw_timeRelated_v2, \
                                  ppd_energy_reward_iw_timeRelated_v3, ppd_energy_reward_iw_timeRelated_v4
from a3c_v0_1.action_funcs import mull_stpt_iw, mull_stpt_oaeTrans_iw, mull_stpt_noExpTurnOffMullOP, \
                                  mull_stpt_directSelect, iw_iat_stpt_noExpHeatingOp
from a3c_v0_1.raw_state_processors import raw_state_process_iw
from a3c_v0_1.action_limits import act_limits_iw_1, act_limits_iw_2, act_limits_iw_3, act_limits_iw_4

def main():
    # Common args
    parser = get_args();
    # Specific args
    parser.add_argument('--err_penalty_scl', default=13.0, type=float,
                        help='Scale the IAT temperature difference error, default is 13.0')
    parser.add_argument('--act_func', default='1', type=str,
                        help='The action function corresponding to the action space, default is 1, '
                             'corresponding to the actions space iw_1')
    parser.add_argument('--reward_func', default='1', type=str)
    args = parser.parse_args();
    # Prepare case specific args
    if args.reward_func == '1':
      reward_func = err_energy_reward_iw;
    elif args.reward_func == '2':
      reward_func = err_energy_reward_iw_v2;
    elif args.reward_func == '3':
      reward_func = err_energy_reward_iw_v3;
    elif args.reward_func == '4':
      reward_func = err_energy_reward_iw_v4;
    elif args.reward_func == '5':
      reward_func = err_energy_reward_iw_v5;
    elif args.reward_func == '6':
      reward_func = err_energy_reward_iw_v6;
    elif args.reward_func == '7':
      reward_func = ppd_energy_reward_iw_timeRelated;
    elif args.reward_func == '8':
      reward_func = ppd_energy_reward_iw_timeRelated_v2;
    elif args.reward_func == '9':
      reward_func = ppd_energy_reward_iw_timeRelated_v3;
    elif args.reward_func == '10':
      reward_func = ppd_energy_reward_iw_timeRelated_v4;
    
      


    rewardArgs = [args.err_penalty_scl];

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
    raw_state_process_func = raw_state_process_iw;
    effective_main(args, reward_func, rewardArgs, action_func, action_limits, raw_state_process_func);
        

if __name__ == '__main__':
    main()
 
