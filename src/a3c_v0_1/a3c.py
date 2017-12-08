"""Main A3C agent.
Some codes are inspired by 
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
"""
import os
import threading
import time
import gym
import eplus_env
import numpy as np
import tensorflow as tf
from multiprocessing import Value, Lock
from keras import backend as K

from util.logger import Logger
from a3c_v0_1.objectives import a3c_loss
from a3c_v0_1.a3c_network import A3C_Network
from a3c_v0_1.actions import action_map
from a3c_v0_1.preprocessors import HistoryPreprocessor, process_raw_state_cmbd
from a3c_v0_1.utils import init_variables, get_hard_target_model_updates, get_uninitialized_variables
from a3c_v0_1.state_index import *
from a3c_v0_1.a3c_eval import A3CEval_multiagent, A3CEval
from a3c_v0_1.env_interaction import IWEnvInteract

ACTION_MAP = action_map;
LOG_LEVEL = 'DEBUG';
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";
TIME_DIM = 2;

class A3CThread:
    """
    The thread worker of the A3C algorithm. 
    """
    
    def __init__(self, graph, scope_name, global_name, effec_state_dim, forecast_dim, 
                action_size, vloss_frac, ploss_frac, hregu_frac, hregu_decay_bounds, shared_optimizer,
                 clip_norm, global_train_step, window_len, init_epsilon,
                 end_epsilon, decay_steps, global_counter):
        """
        Constructor.
        
        Args:
            graph: tf.Graph
                The tensorflow computation graph.
            scope_name: String
                The scope name of this thread.
            global_name: String
                The global network scope name. 
            state_dim: int 
                The state dimension. It should be the dimension of the raw 
                observation of the environment plus 2 (time info). 
            action_size: int 
                Number of action choices. 
            vloss_frac: float
                Used for constructing the loss operation. The fraction to the 
                value loss. 
            ploss_frac: float
                Used for constructing the loss operation. The fraction to the 
                policy loss.
            hregu_frac: float
                Used for constructing the loss operation. The fraction to the 
                entropy regulation in the loss function.
            shared_optimizer: tf.train.Optimizer
                The tensorflow train optimizer.
            clip_norm: float
                Used for gradient clipping.
            global_train_step: tf.Variable
                A shared tensorflow variable to record the global training steps.
            window_len: int 
                The window length to include the history state into the state
                representation. 
            init_epsilon: float
                The initial epsilon value for exploration.
            end_epsilon: float
                The final epsilon value for exploration.
            decay_steps: float
                The epsilon decay steps.
        
        """
        
        ###########################################
        ### Create the policy and value network ###
        ###########################################
        network_state_dim = effec_state_dim;
        self._a3c_network = A3C_Network(graph, scope_name, network_state_dim, 
                                        action_size);
        self._policy_pred = self._a3c_network.policy_pred;
        self._value_pred = self._a3c_network.value_pred;
        self._state_placeholder = self._a3c_network.state_placeholder;
        self._keep_prob = self._a3c_network.keep_prob;
        self._shared_layer = self._a3c_network.shared_layer;
        
        with graph.as_default(), tf.name_scope(scope_name):
        
        #################################
        ### Create the loss operation ###
        #################################
            # Generate placeholders state and "true" q values
            self._q_true_placeholder = tf.placeholder(tf.float32,
                                                      shape=(None, 1),
                                                      name='q_true_pl');
            # Generate the tensor for one hot policy probablity
            self._action_idx_placeholder = tf.placeholder(tf.uint8,
                                                          shape=(None),
                                                          name='action_idx_pl');
            pi_one_hot = tf.reduce_sum((tf.one_hot(self._action_idx_placeholder,
                                                   action_size) * 
                                        self._policy_pred),
                                        1, True); 
            self._pi_one_hot = pi_one_hot;
            # Add to the Graph the Ops for loss calculation.
            self._this_thread_global_counter = tf.Variable(0, trainable = False, dtype = tf.int32, name = 'global_counter_this_thread');
            self._global_step_pl = tf.placeholder(tf.int32, name = 'global_step_pl');
            self._assg_global_step = tf.assign(self._this_thread_global_counter, self._global_step_pl, name = 'global_step_assign');
            if len(hregu_frac) == 1:
                hregu_frac_to_loss = tf.constant(hregu_frac[0], name = 'H_regu_cst');
            else:
                hregu_frac_to_loss = tf.train.piecewise_constant(self._this_thread_global_counter, hregu_decay_bounds, hregu_frac, name='H_regu_decay')
            loss = a3c_loss(self._q_true_placeholder, self._value_pred, 
                                  self._policy_pred, pi_one_hot, vloss_frac, 
                                  ploss_frac, hregu_frac_to_loss);
            self._loss = loss;
        
        #####################################
        ### Create the training operation ###
        #####################################
            merged_summary_list = [];
            # Add a scalar summary for the snapshot loss.
            self._loss_summary = tf.summary.scalar('loss', loss)
            merged_summary_list.append(self._loss_summary);
            # Compute the gradients
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                           scope_name);
            grads_and_vars = shared_optimizer.compute_gradients(loss, local_vars);
            grads = [item[0] for item in grads_and_vars]; # Need only the gradients
            grads, grad_norms = tf.clip_by_global_norm(grads, clip_norm) 
                                                          # Grad clipping
            # Add a histogram for the snapshot gradient and variable values
            for var_i in range(len(grads_and_vars)):
                merged_summary_list.append(tf.summary.histogram(
                            grads_and_vars[var_i][1].name + '/grad', grads[var_i]));
            for var in local_vars:
                merged_summary_list.append(tf.summary.histogram(var.name, var));
            # Apply local gradients to global network
            self._grads = grads;
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                            global_name);
            self._train_op = shared_optimizer.apply_gradients(
                                                    zip(grads,global_vars),
                                                    global_step=global_train_step);
            # Merge three summaries into one
            self._merged_summary = tf.summary.merge(merged_summary_list);
        ######################################
        ### Create local network update op ###
        ######################################
        self._local_net_update = get_hard_target_model_updates(graph,
                                                               scope_name, 
                                                               global_name);
        
        #####################################################
        self._forecast_dim = forecast_dim;
        self._network_state_dim = network_state_dim;
        self._window_len = window_len;
        self._histProcessor = HistoryPreprocessor(window_len, forecast_dim);
        self._scope_name = scope_name;
        self._graph = graph;
        self._grad_norms = grad_norms;
        self._action_size = action_size;
        self._epsilon_decay_delta = (init_epsilon - end_epsilon)/decay_steps;
        self._e_greedy = init_epsilon;
        self._hregu_frac_to_loss = hregu_frac_to_loss;
        self._global_counter = global_counter;
        
            
    def train(self, sess, t_max, env_name, coordinator, global_lock, 
              gamma, e_weight, p_weight, save_freq, log_dir, global_saver, 
              global_summary_writer, T_max, global_agent_eval_list, eval_freq, 
              global_res_list, action_space_name, dropout_prob, reward_func, 
              rewardArgs, action_func, action_limits, raw_state_process_func):
        """
        The function that the thread worker works to train the networks.
        
        Args:
            sess: tf.Session
                The shared session of tensorflow.
            t_max: int 
                The interaction number with the environment before performing
                one training. 
            env_name: String 
                The environment name.
            coordinator: tf.train.Coordinator
                The shared coordinator for multithreading training.
            global_counter: python multiprocessing.Value
                The shared counter. 
            global_lock: python threading.Lock
                The shared thread lock.
            gamma: float
                The discount rate.
            e_weight: float
                Used for constructing reward. The weight to the HVAC energy.
            p_weight: float
                Used for constructing reward. The weight to the PPD. 
            save_freq: int 
                The frequency to save the global network regarding the global
                counter.
            log_dir: String
                The directory to save the global network.
            global_saver: tf.Saver
                The global saver object to save the network.
            local_logger: Logger object.
                The local Logger object for logging. 
            global_summary_writer: tf.summary.FileWriter
                The global FileWriter object to save the summary output.
            T_max: int 
                The global maximum number of interactions with the environment.
            global_agent_eval_list: list of A3CEval
                A shared A3C evaluation object list. 
            eval_freq: int 
                The evaluation frequency regarding the global training step.
            ppd_penalty_limit: float
                Larger than ppd_penalty_limit PPD will be changed to 1.0.
                
        """
        action_space = ACTION_MAP[action_space_name];
        self._local_logger = Logger().getLogger('A3C_AGENT_WORKER-%s'
                                    %(threading.current_thread().getName()),
                                              LOG_LEVEL, LOG_FMT, log_dir + '/main.log');
        self._local_logger.info('Local worker starts!')
        # Assign value to global_counter this thread
        sess.run(self._assg_global_step, 
                 feed_dict = {self._global_step_pl: int(self._global_counter.value)});
        # Init some variables
        t = 0;
        t_st = 0;
        # Create the thread specific environment
        env = gym.make(env_name);
        # Prepare env-related information
        env_st_yr = env.start_year;
        env_st_mn = env.start_mon;
        env_st_dy = env.start_day;
        env_st_wd = env.start_weekday;
        env_state_limits = env.min_max_limits;
        env_state_limits.insert(0, (0, 23)); # Add hour limit
        env_state_limits.insert(0, (0, 6)); # Add weekday limit
        pcd_state_limits = np.transpose(env_state_limits);
        env_interact_wrapper = IWEnvInteract(env, raw_state_process_func);
        # Reset the env
        time_this, ob_this_raw, is_terminal = env_interact_wrapper.reset();
        ob_this_prcd = process_raw_state_cmbd(ob_this_raw, [time_this], env_st_yr, 
                                              env_st_mn, env_st_dy, env_st_wd, 
                                              pcd_state_limits); # 1-D list
        # Get the history stacked state
        ob_this_hist_prcd = self._histProcessor.\
                            process_state_for_network(ob_this_prcd) # 2-D array
    
        
        while not coordinator.should_stop():
            sess.run(self._local_net_update);
            # print debug
            #print ('global net ......................................................')
            #global_collection = self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
            #                               'global');
            #print (sess.run(global_collection[0]));
            #print ('worker 0 net ......................................................')
            #worker_collection = self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
            #                               'worker_0');
            #print (sess.run(worker_collection[0]));

            # Reset the counter
            t_st = t;
            # Interact with env
            trajectory_list = []; # A list of (s_t, a_t, r_t) tuples
            while (not is_terminal) and (t - t_st != t_max):
                #self._local_logger.debug('The processed stacked state at %0.04f '
                #                         'is %s.'%(time_this, str(ob_this_hist_prcd)));
                # Get the action
                #################FOR DEBUG#######################
                dbg_rdm = np.random.uniform();
                noForecastDim = 13;
                forecastSingleEntryDim = 4;
                dbg_thres = 0.0005;
                is_show_dbg = True if dbg_rdm < dbg_thres else False;
                #################################################

                action_raw_idx = self._select_sto_action(ob_this_hist_prcd, sess,
                                                         self._e_greedy, is_show_dbg, 
                                                         dropout_prob = dropout_prob); ####DEBUG FOR DROPOUT
                action_raw_tup = action_space[action_raw_idx];
                
                action_stpt_prcd, action_effec = action_func(action_raw_tup, action_limits, ob_this_raw);
                action_stpt_prcd = list(action_stpt_prcd);
                # Take the action
                time_next, ob_next_raw, is_terminal = env_interact_wrapper.step(action_stpt_prcd);
                ob_next_prcd = process_raw_state_cmbd(ob_next_raw, [time_next], 
                                              env_st_yr, env_st_mn, env_st_dy,
                                              env_st_wd, pcd_state_limits); # 1-D list
                # Get the reward
                reward_next = reward_func(ob_next_prcd, e_weight, p_weight, *rewardArgs);
                
                #################FOR DEBUG#######################
                if is_show_dbg:
                    current_hregu = sess.run(hregu_frac_to_loss);
                    self._local_logger.debug('TRAINING DEBUG INFO ======>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
                                         'Current H regulation is %0.04f, \n'
                                         'Environment debug: raw action idx is %d, \n'
                                         'current raw observation is %s, \n'
                                         'current ob forecast is %s, \n'
                                         'actual action is %s, \n'
                                         'sim time this is %0.04f, \n' 
                                         'sim time next is %0.04f, \n'
                                         'raw observation next is %s, \n'
                                         'processed observation next is %s, \n'
                                         'reward next is %0.04f. \n'
                                         '============================================='
                                         %(current_hregu, action_raw_idx, ob_this_raw[0: noForecastDim],
                                           np.insert(np.array(ob_this_raw[noForecastDim:]).astype('str'),
                                            range(0, (len(ob_this_raw) - noForecastDim), forecastSingleEntryDim), 'Next Hour'),
                                           str(action_stpt_prcd), time_this, time_next, 
                                           ob_next_raw[0: noForecastDim], ob_next_prcd, reward_next));
                #################################################

                # Get the history stacked state
                ob_next_hist_prcd = self._histProcessor.\
                            process_state_for_network(ob_next_prcd) # 2-D array
                # Remember the trajectory 
                trajectory_list.append((ob_this_hist_prcd, action_raw_idx, 
                                        reward_next)) 
                
                # Update local counter and global counter, do eval
                t += 1;
                self._update_e_greedy(); # Update the epsilon value
                with global_lock:
                    # Do the evaluation
                    if self._global_counter.value % eval_freq == 0: 
                        self._local_logger.info('Evaluating...');
                        global_res_list.append([self._global_counter.value]);
                        for global_agent_eval in global_agent_eval_list:
                            eval_res = global_agent_eval.evaluate(self._local_logger,
                                                    action_space_name, reward_func, rewardArgs, 
                                                    action_func, action_limits, raw_state_process_func);
                            global_res_list[-1].extend([eval_res]);
                        np.savetxt(log_dir + '/eval_res_hist.csv', 
                                   np.array(global_res_list), delimiter = ',');
                        self._local_logger.info ('Global step: %d, '
                                           'evaluation results %s'
                                           %(self._global_counter.value, str(global_res_list[-1])));
                    # Global counter increment
                    self._global_counter.value += 1;
                # Save the global network variable
                if self._global_counter.value % save_freq == 0: 
                    checkpoint_file = os.path.join(log_dir, 'model_data/model.ckpt');
                    global_saver.save(sess, checkpoint_file, 
                               global_step=int(self._global_counter.value));
               

                # ...
                if not is_terminal:
                    ob_this_hist_prcd = ob_next_hist_prcd;
                    ob_this_raw = ob_next_raw;
                    ob_this_prcd = ob_next_prcd;
                    time_this = time_next;
                else:
                    # Reset the env
                    time_this, ob_this_raw, is_terminal_cp = env_interact_wrapper.reset();
                    ob_this_prcd = process_raw_state_cmbd(ob_this_raw, [time_this], 
                                              env_st_yr, env_st_mn, env_st_dy,
                                              env_st_wd, pcd_state_limits); # 1-D list
                    # Get the history stacked state
                    self._histProcessor.reset();
                    ob_this_hist_prcd = self._histProcessor.\
                            process_state_for_network(ob_this_prcd) # 2-D array
            # Prepare for the training step
            R = 0 if is_terminal else sess.run(
                            self._value_pred,
                            feed_dict = {self._state_placeholder:ob_this_hist_prcd,
                                         self._keep_prob: 1.0 - dropout_prob})####DEBUG FOR DROPOUT####
            traj_len = len(trajectory_list);
            act_idx_list = np.zeros(traj_len, dtype = np.uint8);
            q_true_list = np.zeros((traj_len, 1));
            state_list = np.zeros((traj_len, self._network_state_dim));
            for i in range(traj_len):
                traj_i_from_last = trajectory_list[traj_len - i - 1]; #(s_t, a_t, r_t);
                R = gamma * R + traj_i_from_last[2];
                act_idx_list[i] = traj_i_from_last[1];
                q_true_list[i, :] = R;
                state_list[i, :] = traj_i_from_last[0];
            # Perform training
            training_feed_dict = {self._q_true_placeholder: q_true_list,
                                  self._state_placeholder: state_list,
                                  self._action_idx_placeholder: act_idx_list,
                                  self._keep_prob: 1.0 - dropout_prob};
            
            _, loss_res, value_pred = sess.run([self._train_op, self._loss, 
                                                self._value_pred], 
                                   feed_dict = training_feed_dict);
            #################FOR DEBUG#######################
            if is_show_dbg:
                self._local_logger.debug('Value prediction is %s, R is %s.'
                                     %(str(value_pred), str(q_true_list)));
            #################################################
            # Display and record the loss for this thread
            printStatusFreq = 100;
            if (t/t_max) % printStatusFreq == 0:
                self._local_logger.info ('Local step %d, global step %d: loss ' 
                                       '%0.04f'%(t, self._global_counter.value, loss_res));
                # Update the events file.
                #summary_str = sess.run(self._loss_summary, 
                #                             feed_dict=training_feed_dict)
            saveSummaryFreq = 1000;
            if (t/t_max) % saveSummaryFreq == 0: 
                summary_str_all = sess.run(self._merged_summary, feed_dict = training_feed_dict);
                global_summary_writer.add_summary(summary_str_all, t);
                global_summary_writer.flush();
            # ...
            if is_terminal:
                is_terminal = is_terminal_cp;
            # Check whether training should stop
            if self._global_counter.value > T_max:
                coordinator.request_stop()
        # Safely close the environment
        env.end_env();
            
            
    def _update_e_greedy(self):
        self._e_greedy -= self._epsilon_decay_delta;
        
    def _select_sto_action(self, state, sess, e_greedy, is_show_dbg, dropout_prob):
        """
        Given a state, run stochastic policy network to give an action.
        
        Args:
            state: np.ndarray, 1*m where m is the state feature dimension.
                Processed normalized state.
            sess: tf.Session.
                The tf session.
            e_greedy: float
                The exploration probability.
            dropout_prob: float
                The dropout probability, currently not in use. 
        
        Return: int 
            The action index.
        """
        # Random
        uni_rdm_greedy = np.random.uniform();
        if uni_rdm_greedy < e_greedy:
            return np.random.choice(self._action_size);
        # On policy
        softmax_a, shared_layer = sess.run([self._policy_pred, self._shared_layer],
                             feed_dict={self._state_placeholder:state,
                                        self._keep_prob: 1.0 - dropout_prob}) ####DEBUG FOR DROPOUT
        softmax_a = softmax_a.flatten();
        if is_show_dbg:
            self._local_logger.debug('Policy network output: %s, sum to %0.04f'
                                 %(str(softmax_a), sum(softmax_a)));
        uni_rdm = np.random.uniform(); # Avoid select an action with too small probability
        imd_x = uni_rdm;
        for i in range(softmax_a.shape[-1]):
            imd_x -= softmax_a[i];
            if imd_x <= 0.0:
                selected_act = i;
                return selected_act;
    

class A3CAgent:
    """
    The A3C Agent class. 

    Args:
        forecast_dim: int
            The total forecast dimension.
        state_dim: int
            The state dimension.
        window_len: int
            The state stack window length.
        vloss_frac: float
            The value loss fraction.
        ploss_frac: float
            The policy loss fraction.
        hregu_frac: float
            The enthalpy regulation fraction.
        num_threads: int
            The number of threads to be used.
        learning_rate: float
            The learning rate.
        rmsprop_decay: float
            The decay rate of the RMSProp optimizer.
        rmsprop_momet: float
            The momentum value of the RMSProp optimizer.
        rmsprop_epsil: float
            The epsilon value of the RMSProp optimizer.
        clip_norm: float
            The gradient clip value.
        log_dir: str
            The log file storage directory.
        init_epsilon: float
            The initial epsilon value for exploration.
        end_epsilon: float
            The final epsilon value for exploration.
        decay_steps: float
            The epsilon decay steps.
        action_space_name: str
            The action space name.
        dropout_prob: float
            The dropout probility. Current not in use. 

    """
    def __init__(self,
                 forecast_dim,
                 state_dim,
                 window_len,
                 vloss_frac, 
                 ploss_frac, 
                 hregu_frac,
                 hregu_decay_bounds,
                 num_threads,
                 learning_rate,
                 rmsprop_decay,
                 rmsprop_momet,
                 rmsprop_epsil,
                 clip_norm,
                 log_dir,
                 init_epsilon,
                 end_epsilon,
                 decay_steps,
                 action_space_name,
                 dropout_prob,
                 global_logger):
        self._forecast_dim = forecast_dim;
        state_dim += TIME_DIM; # Add time info dimension
        self._state_dim = state_dim;
        self._window_len = window_len;
        self._effec_state_dim = state_dim * window_len + forecast_dim;
        self._action_size = len(ACTION_MAP[action_space_name]);
        self._vloss_frac = vloss_frac;
        self._ploss_frac = ploss_frac;
        self._hregu_frac = hregu_frac;
        self._hregu_decay_bounds = hregu_decay_bounds;
        self._num_threads = num_threads;
        self._learning_rate = learning_rate;
        self._rmsprop_decay = rmsprop_decay;
        self._rmsprop_momet = rmsprop_momet;
        self._rmsprop_epsil = rmsprop_epsil;
        self._clip_norm = clip_norm;
        self._log_dir = log_dir
        self._init_epsilon = init_epsilon;
        self._end_epsilon = end_epsilon;
        self._decay_steps = decay_steps;
        self._action_space_name = action_space_name;
        self._dropout_prob = dropout_prob;
        self._global_logger = global_logger;
        
    def compile(self, is_warm_start, model_dir, save_scope = 'global'):
        """
        This method sets up the required TF graph and operations.
        
        Args:
            is_warm_start: bool
                Whether to read trained neural network from the file, and train based on that. 
            model_dir: str
                If is_warm_start is true, this arg is the model directory.
            save_scope: str
                The model save scope, choice of global (save global network only) and all (save all networks).
        
        Return: tuple
            (tf.graph, tf.session, tf.train.Coordinator(), tf.tensor of the global_network, 
            tf.tensor of the workers' network, tf.summary.FileWriter, tf.train.Saver)
        
        """
        g = tf.Graph();
        # Create the global network
        global_network = A3C_Network(g, 'global', self._effec_state_dim,
                                     self._action_size);
        with g.as_default():
            # Create a shared optimizer
            with tf.name_scope('optimizer'):
                shared_optimizer = tf.train.RMSPropOptimizer(
                                                     self._learning_rate, 
                                                     self._rmsprop_decay,
                                                     self._rmsprop_momet,
                                                     self._rmsprop_epsil)
            # Create a coordinator for multithreading
            coordinator = tf.train.Coordinator();
            # Create a global train step variable to record global steps
            global_train_step = tf.Variable(0, name='global_train_step', 
                                            trainable=False);
        # Create the thread workers list
        global_counter = Value('d', 0.0);
        workers = [A3CThread(g, 'worker_%d'%(i), 'global', self._effec_state_dim, self._forecast_dim,
                             self._action_size, self._vloss_frac, self._ploss_frac,
                             self._hregu_frac, self._hregu_decay_bounds, shared_optimizer, self._clip_norm,
                             global_train_step, self._window_len, self._init_epsilon,
                             self._end_epsilon, self._decay_steps, global_counter)
                  for i in range(self._num_threads)];
        # Init global network variables or warm start
        with g.as_default():
            # Create a session for running Ops on the Graph.
            sess = tf.Session()
            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.summary.FileWriter(self._log_dir, sess.graph)
            # Create a saver for writing training checkpoints
            if save_scope == 'global':
                save_var_list = g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             scope=save_scope)
            if save_scope == 'all':
                save_var_list = None;
            saver = tf.train.Saver(var_list = save_var_list);
            # Init ops
            init_global_all_op = tf.global_variables_initializer();
            if not is_warm_start:
                sess.run(init_global_all_op);
            else:
                saver.restore(sess, model_dir);
        # Graph construction finished. No addiontal elements can be added to 
        # the graph. This is for thread safety.  
        g.finalize(); 
        return (g, sess, coordinator, global_network, workers, summary_writer, 
                saver);

    def test(self, sess, global_network, env_test_name, num_episodes, e_weight, p_weight, 
                reward_mode, test_mode, agent_num, ppd_penalty_limit, log_dir):

        """
        This method is used to test the trained agent for HVAC control.
        
        Args:
            sess: tf.Session
                The tf.Session object.
            global_network: tf.Tensor
                The global network tensor.
            env_test_name: str
                The test environment name.
            num_episodes: int
                The test episode number.
            e_weight: float
                The penalty weight on energy.
            p_weight: float
                The penalty weight on comfort.
            reward_mode: str
                The reward mode.
            test_mode: str
                The test mode, choice of Single or Multiple.
            agent_num: int
                If the test_mode is Multiple, then this arg determines how many zones 
                will be controlled. 
            ppd_penalty_limit: float
                After the PPD exceeds this limit, it will be set to 1.0.
            log_dir: str
                The log directory.
        
        Return: None
        
        """
        env_test = gym.make(env_test_name);
        if test_mode == 'single':
        	a3c_eval = A3CEval(sess, global_network, env_test, num_episodes, 
                                self._window_len, e_weight, p_weight);
        	eval_logger = Logger().getLogger('A3C_Test_Single-%s'%(threading.current_thread().getName()),
                                                 LOG_LEVEL, LOG_FMT, log_dir + '/main.log');
        if test_mode == 'multiple':
        	a3c_eval = A3CEval_multiagent(sess, global_network, env_test, num_episodes, self._window_len, 
                                            e_weight, p_weight, agent_num)
        	eval_logger = Logger().getLogger('A3C_Test_Multiple-%s'%(threading.current_thread().getName()),
                                                 LOG_LEVEL, LOG_FMT, log_dir + '/main.log');
        
        eval_logger.info("Testing...")
        eval_res = a3c_eval.evaluate(eval_logger, reward_mode, self._action_space_name, 
                                        ppd_penalty_limit, raw_state_process_func);
        eval_logger.info("Testing finished.")

    def fit(self, sess, coordinator, global_network, workers, global_summary_writer, global_saver,
            env_name_list, t_max, gamma, e_weight, p_weight, save_freq, T_max, eval_epi_num, eval_freq,
            reward_func, rewardArgs, action_func, action_limits, raw_state_process_func):
        """
        This method is used to train the neural network. 
        
        Args:
            sess: tf.Session
                The tf.Session object.
            coordinator: tf.train.Coordinator
                The multithreading coordinator object.
            global_network: tf.Tensor
                The global network tensor.
            workers: list
                The workers' network tensor list.
            global_summary_writer: tf.summary.FileWriter
                The FileWriter object.
            global_saver: tf.train.saver
                The Saver object. 
            env_name_list: list
                The list of the environment names. 
            t_max: int
                The interaction number with the environment before performing
                one training.
            gamma: float
                The discount rate.
            e_weight: float
                The penalty weight on energy.
            p_weight: float
                The penalty weight on comfort.
            save_freq: int
                The frequency to save the training.
            T_max: int
                The global maximum number of interactions with the environment.
            eval_epi_num: int
                The evaluation episode number. 
            eval_freq: int
                The evaluation frequency.
            reward_mode: str
                The reward mode.
            ppd_penalty_limit: float
                If PPD value exceeds this limit, then it will be set to 1.0. 
        
        Return: None
        
        """
        threads = [];
        global_lock = Lock();
        # Create the env for training evaluation
        global_agent_eval_list = [];
        self._global_logger.info('Prepare the evaluation environments %s ...', env_name_list);
        for env_name in env_name_list:
            env_eval = gym.make(env_name);
            global_agent_eval = A3CEval(sess, global_network, env_eval, eval_epi_num, 
                                    self._window_len, self._forecast_dim, e_weight, p_weight);
            global_agent_eval_list.append(global_agent_eval)

        global_res_list = [];
        thread_counter = 0;
        for worker in workers:
            self._global_logger.info('Prepare the local workers ...');
            worker_train = lambda: worker.train(sess, t_max, 
                                                env_name_list[0], coordinator, 
                                                global_lock, gamma, e_weight, p_weight, save_freq,
                                                self._log_dir, global_saver, global_summary_writer,
                                                T_max, global_agent_eval_list, eval_freq, global_res_list,
                                                self._action_space_name, self._dropout_prob, reward_func, 
                                                rewardArgs, action_func, action_limits, raw_state_process_func);

            thread = threading.Thread(target = (worker_train));
            thread.start();
            time.sleep(1); # Wait for a while for the env to setup
            threads.append(thread);
            thread_counter += 1;
            
        coordinator.join(threads);
