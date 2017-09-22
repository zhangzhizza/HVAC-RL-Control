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
from a3c_v0_1.reward_funcs import ppd_energy_reward_smlRefBld     
from a3c_v0_1.action_funcs import iat_stpt_smlRefBld
from a3c_v0_1.action_limits import act_limits_smlRefBld   

def main():
    # Common args
    parser = get_args();
    # Specific args
    parser.add_argument('--reward_mode', default='linear', type=str);
    parser.add_argument('--ppd_penalty_limit', default=0.15, type=float,
                        help='Larger than ppd_penalty_limit PPD will be changed '
                             'to the max PPD. Should be 0~1, default is 0.15.')
    args = parser.parse_args();
    # Prepare case specific args
    reward_func = ppd_energy_reward_smlRefBld;
    rewardArgs = [args.reward_mode, args.ppd_penalty_limit];
    action_func = iat_stpt_smlRefBld;
    action_limits = act_limits_smlRefBld;
    effective_main(args, reward_func, rewardArgs, action_func, action_limits);
        

if __name__ == '__main__':
    main()
 
