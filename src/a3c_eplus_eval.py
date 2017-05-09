#!python
"""
Evaluate the RL policy network.

Author: Zhiang Zhang
Last update: Apr 29th, 2017

"""
import argparse
import os
import random
import gym
import logging
import multiprocessing

import numpy as np
import tensorflow as tf

import eplus_env

from util.logger import Logger
from a3c.a3c import A3CAgent
        
NAME = 'RL_POLICY_EVAL_MAIN'
LOG_LEVEL = 'INFO'
LOG_FORMATTER = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";

def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main(): 
    main_logger = Logger().getLogger(NAME, LOG_LEVEL, LOG_FORMATTER);
    parser = argparse.ArgumentParser(description='Run A3C on EnergyPlus')
    parser.add_argument(
        '-o', '--output', default='a3c-eval', help='Directory to save data to')
    parser.add_argument('--window_len', default=4, type=int);
    parser.add_argument('--e_weight', default=0.4, type=float,
                        help='Reward weight on HVAC energy consumption.');
    parser.add_argument('--p_weight', default=0.6, type=float,
                        help='Reward wegith on PPD.');
    parser.add_argument('--reward_mode', default='linear', type=str);
    parser.add_argument('--action_space', type=str);
    parser.add_argument('--eval_epi_num', default=5, type=int);
    parser.add_argument('--model_dir', default='None');
    parser.add_argument('--test_env', default='Eplus-eval-v0', type=str);
    parser.add_argument('--test_mode', default='single', type=str);
    parser.add_argument('--agent_num', default=5, type=int);
    
    args = parser.parse_args();
    args.output = get_output_folder(args.output, args.test_env)
    tf.gfile.MakeDirs(args.output + '/model_data')
    main_logger.info(args)
    
    # State size
    state_dim = 15 + 2 # 15 for the raw state dim, 2 is the additonal time info
    # Create the agent
    a3c_agent = A3CAgent(state_dim = state_dim, window_len = args.window_len,
                         vloss_frac = 0.5,
                         ploss_frac = 1.0, 
                         hregu_frac = 0.001,
                         num_threads = 1, 
                         learning_rate = 0.0001, 
                         rmsprop_decay = 0.99,
                         rmsprop_momet = 0.0,
                         rmsprop_epsil = 1e-10,
                         clip_norm = 5.0, log_dir = args.output,
                         init_epsilon = 0.5, end_epsilon = 0.05, 
                         decay_steps = 100000, action_space_name = args.action_space);
    main_logger.info ('Start compiling...')
    (g, sess, coordinator, global_network, workers, global_summary_writer, 
     global_saver) = a3c_agent.compile(True, args.model_dir, 'global');
    main_logger.info ('Start the testing...')
    a3c_agent.test(sess, global_network, args.test_env, args.eval_epi_num, args.e_weight, 
                       args.p_weight, args.reward_mode, args.test_mode, args.agent_num);
        

if __name__ == '__main__':
    main()
 
