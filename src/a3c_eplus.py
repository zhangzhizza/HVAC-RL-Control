#!python
"""
Run EnergyPlus Environment with Asynchronous Advantage Actor Critic (A3C).

Algorithm taken from https://arxiv.org/abs/1602.01783
'Asynchronous Methods for Deep Reinforcement Learning'

Author: Zhiang Zhang
Last update: Apr 4th, 2017

"""
import argparse
import os
import random
import gym
import logging

import numpy as np
import tensorflow as tf

import eplus_env

from util.logger import Logger
        
NAME = 'A3C_Agent'
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
    logger = Logger();
    main_logger = logger.getLogger(NAME, LOG_LEVEL, LOG_FORMATTER);
    main_logger.debug ('A3C agent starts!');
    parser = argparse.ArgumentParser(description='Run A3C on EnergyPlus')
    parser.add_argument('--env', default='Eplus-v0', help='EnergyPlus env name')
    parser.add_argument(
        '-o', '--output', default='a3c-res', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--max_interactions', default=50000000, type=int);
    parser.add_argument('--window_len', default=16, type=int);
    parser.add_argument('--gamma', default=0.99);
    parser.add_argument('--save_freq', default=2500, type=int);
    parser.add_argument('--train_interval', default=4, type=int, 
                        help = 'Steps of environment interaction before \
                                performing training.');
    parser.add_argument('--eval_freq', default=5000, type=int);
    parser.add_argument('--eval_epi_num', default=20, type=int);
    parser.add_argument('--learning_rate', default=0.0001)
    parser.add_argument('--is_warm_start', default=False, type=bool);
    parser.add_argument('--model_dir', default='None');

    args = parser.parse_args()

    args.output = get_output_folder(args.output, args.env)
    tf.gfile.MakeDirs(args.output + '/model_data')
    
    main_logger.info(args)
    
    # Create the env
    env = gym.make(args.env);
    env_eval = gym.make(args.env);

    action_size = 9; # the element of permutation set with (-0.5, 0. 0.5)
    
    # Create the agent
    preprocessor = Preprocessor();

    dnqnAgent = DNQNAgent(preprocessor, replayMem, args.gamma
                        , args.target_update_freq, args.burn_in_size
                        , args.train_freq, args.eval_freq, args.eval_epi_num
                        , args.batch_size, state_size, action_size
                        , args.learning_rate, args.start_epsilon
                        , args.end_epsilon, args.e_decay_num_steps
                        , args.output, args.save_freq);
    logging.info ('Start compiling...')

    dnqnAgent.compile(tf.train.AdamOptimizer, mean_huber_loss,
        args.is_warm_start, args.model_dir);
    
    #run the training
    logging.info ('Start the learning...')
    dnqnAgent.fit(env, env_eval, args.max_interactions, max_episode_length=None)
        
        

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO);
    main()
 
