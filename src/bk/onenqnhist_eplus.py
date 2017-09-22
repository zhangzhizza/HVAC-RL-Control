#!/usr/bin/env python
"""Run Energy Plus Environment with DNQN."""
import argparse
import os
import random
import gym
import logging
import time

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import Model
from keras.optimizers import Adam

import rl 
from rl.onenqnhist import OneNQNAgent
from rl.core import ReplayMemory, Preprocessor
from rl.preprocessors import HistoryPreprocessor
from rl.objectives import mean_huber_loss
from gym import wrappers
import eplus_env

        

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
    logging.info ('ONENQN started!!!!!!!!!!!!!!!!!!!!!');
    parser = argparse.ArgumentParser(description='Run ONENQN on EnergyPlus')
    parser.add_argument('--env', default='Eplus-v0', help='EnergyPlus env name')
    parser.add_argument(
        '-o', '--output', default='onenqnhist-res', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--max_interactions', default=50000000, type=int);
    parser.add_argument('--mem_size', default=8640, type=int);
    parser.add_argument('--window_len', default=4, type=int);
    parser.add_argument('--gamma', default=0.99);
    parser.add_argument('--target_update_freq', default=5000, type=int);
    parser.add_argument('--save_freq', default=2500, type=int);
    parser.add_argument('--train_freq', default=4, type=int);
    parser.add_argument('--eval_freq', default=2500, type=int);
    parser.add_argument('--eval_epi_num', default=20, type=int);
    parser.add_argument('--batch_size', default=32, type=int);
    parser.add_argument('--train_set_size', default=8640, type=int);
    parser.add_argument('--learning_rate', default=0.0001);
    parser.add_argument('--start_epsilon', default=0.5, type=float);
    parser.add_argument('--end_epsilon', default=0.05);
    parser.add_argument('--e_decay_num_steps', default=1000000, type=int);
    parser.add_argument('--burn_in_size', default=50000, type=int);
    parser.add_argument('--is_warm_start', default=False, type=bool);
    parser.add_argument('--model_dir', default='None');

    args = parser.parse_args()
    #args.input_shape = tuple(args.input_shape)

    args.output = get_output_folder(args.output, args.env)
    tf.gfile.MakeDirs(args.output + '/model_data')
    logging.info(args)

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.
    
    #create the env
    env = gym.make(args.env);
    time.sleep(60)
    env_eval = gym.make(args.env);
   # env_eval = wrappers.Monitor(gym.make(args.env), args.output + '/eval_res', video_callable = lambda episode_id: episode_id%20 == 0);

    # 
    action_size = 8; # the element of permutation set with (-0.5, 0. 0.5)
    state_size = 16;
    
    #create the agent
    replayMem = ReplayMemory(args.mem_size);
    preprocessor = Preprocessor();
    histPreprocessor = HistoryPreprocessor(args.window_len);

    onenqnAgent = OneNQNAgent(histPreprocessor, preprocessor, replayMem, args.gamma
                        , args.target_update_freq, args.burn_in_size
                        , args.train_freq, args.eval_freq, args.eval_epi_num
                        , args.batch_size, args.window_len
                        , state_size, action_size
                        , args.learning_rate, args.start_epsilon
                        , args.end_epsilon, args.e_decay_num_steps
                        , args.output, args.save_freq);
    logging.info ('Start compiling...')

    onenqnAgent.compile(tf.train.AdamOptimizer, mean_huber_loss,
        args.is_warm_start, args.model_dir);
    
    #run the training
    logging.info ('Start the learning...')
    onenqnAgent.fit(env, env_eval, args.max_interactions, max_episode_length=None)
        
        

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO);
    main()
