#!python
"""
This is the entry script for the A3C HVAC control algorithm. 

Run EnergyPlus Environment with Asynchronous Advantage Actor Critic (A3C).

Algorithm taken from https://arxiv.org/abs/1602.01783
'Asynchronous Methods for Deep Reinforcement Learning'

Author: Zhiang Zhang
Last update: Aug 28th, 2017

"""

      

def main():
    # Common args
    parser = get_args();
    # Specific args
    parser.add_argument('--reward_mode', default='linear', type=str);
    parser.add_argument('--ppd_penalty_limit', default=0.15, type=float,
                        help='Larger than ppd_penalty_limit PPD will be changed '
                             'to the max PPD. Should be 0~1, default is 0.15.')
    args = parser.parse_args();
    
        

if __name__ == '__main__':
    main()
 
