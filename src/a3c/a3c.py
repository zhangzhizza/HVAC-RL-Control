"""Main A3C agent.
Some codes are inspired by 
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
"""
import os
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Dense, Flatten, Input,
                          Permute)
from keras.models import Model

from a3c.objectives import a3c_loss
from a3c.a3c_network import A3C_Network
from a3c.utils import get_hard_target_model_updates
from a3c.actions import actions
from a3c.min_max_limits import min_max_limits, stpt_limits
from a3c.preprocessors import HistoryPreprocessor, process_raw_state_cmbd, get_legal_action, get_reward


ACTIONS = actions;
HT_STPT_IDX_RAW = 8;
CL_STPT_IDX_RAW = 9;
HVAC_EGY_IDX_RAW = 12;
PMV_IDX_RAW = 10;

class A3CThread:
    
    def __init__(self, graph, scope_name, global_name, state_dim, action_size,
                 vloss_frac, ploss_frac, hregu_frac, shared_optimizer,
                 clip_norm, global_train_step, env, window_len):
        
        ###########################################
        ### Create the policy and value network ###
        ###########################################
        network_state_dim = state_dim * window_len;
        self._a3c_network = A3C_Network(graph, scope_name, network_state_dim, 
                                        action_size);
        self._policy_pred = self._a3c_network.policy_pred;
        self._value_pred = self._a3c_network.value_pred;
        self._state_placeholder = self._a3c_network.state_placeholder;
        
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
            # Add to the Graph the Ops for loss calculation.
            loss = a3c_loss(self._q_true_placeholder, self._value_pred, 
                                  self._policy_pred, pi_one_hot, vloss_frac, 
                                  ploss_frac, hregu_frac);
        
        #####################################
        ### Create the training operation ###
        #####################################
            # Add a scalar summary for the snapshot loss.
            tf.summary.scalar('loss', loss)
            # Compute the gradients
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                           scope);
            grads_and_vars = shared_optimizer.compute_gradients(loss, local_vars);
            grads = [item[0] for item in grads_and_vars]; # Need only the gradients
            grads, grad_norms = tf.clip_by_global_norm(grads, clip_norm) 
                                                          # Grad clipping

            # Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                            global_name);
            self._train_op = shared_optimizer.apply_gradients(
                                                    zip(grads,global_vars),
                                                    global_step=global_train_step);
            
        ######################################
        ### Create local network update op ###
        ######################################
        self._local_net_update = get_hard_target_model_updates(graph,
                                                               scope_name, 
                                                               global_name);
        
        #####################################################
        self._env = env;
        self._env_st_yr = env.start_year;
        self._env_st_mn = env.start_mon;
        self._env_st_dy = env.start_day;
        self._env_st_wd = env.start_weekday;
        self._network_state_dim = network_state_dim;
        self._window_len = window_len;
        self._histProcessor = HistoryPreprocessor(window_len);
            
    def train(self, sess, t_max, coordinator, global_counter, global_lock, gamma):
        t = 0;
        t_st = 0;
        # Reset the env
        time_this, ob_this_raw, is_terminal = self._env.reset();
        # Process and normalize the raw observation
        ob_this_prcd = process_raw_state_cmbd(ob_this_raw, time_this, 
                                              self._env_st_yr, self._env_st_mn, 
                                              self._env_st_dy, self._env_st_wd, 
                                              min_max_limits); # 1-D list
        # Get the history stacked state
        ob_this_hist_prcd = self._histProcessor.\
                            process_state_for_network(ob_this_prcd) # 2-D array
        
        while not coordinator.should_stop():
            # Synchronize local network parameters with 
            # the global network parameters
            sess.run(self._local_net_update);
            # Reset the counter
            t_st = t;
            # Interact with env
            trajectory_list = []; # A list of (s_t, a_t, r_t) tuples
            while (not is_terminal) and (t - t_st != self._t_max):
                # Get the action
                action_raw_idx = self._select_sto_action(ob_this_hist_prcd, sess);
                action_raw_tup = actions[action_raw_idx];
                cur_htStpt = ob_this_raw[HT_STPT_IDX_RAW];
                cur_clStpt = ob_this_raw[CL_STPT_IDX_RAW];
                action_stpt_prcd, action_effec = get_legal_action(
                                                        cur_htStpt, cur_clStpt, 
                                                    action_raw_tup, stpt_limits);
                action_stpt_prcd = list(action_stpt_prcd);
                # Take the action
                time_next, ob_next_raw, is_terminal = \
                                                self._env.step(action_stpt_prcd);
                # Process and normalize the raw observation
                ob_next_prcd = process_raw_state_cmbd(ob_next_raw, time_next, 
                                              self._env_st_yr, self._env_st_mn, 
                                              self._env_st_dy, self._env_st_wd, 
                                              min_max_limits); # 1-D list
                # Get the reward
                normalized_hvac_energy = ob_next_prcd[HVAC_EGY_IDX_RAW + 2];
                raw_pmv = ob_next_raw[PMV_IDX_RAW];
                reward_next = get_reward(normalized_hvac_energy, raw_pmv);
                # Get the history stacked state
                ob_next_hist_prcd = self._histProcessor.\
                            process_state_for_network(ob_next_prcd) # 2-D array
                # Remember the trajectory 
                trajectory_list.append((ob_this_hist_prcd, action_raw_idx, 
                                        reward_next)) # Should I use the raw action or effective action ????????????
                # Update lock counter and global counter
                t += 1;
                with global_lock:
                    global_counter.value += 1;
                # ...
                if not is_terminal:
                    ob_this_hist_prcd = ob_next_hist_prcd;
                    ob_this_raw = ob_next_raw;
                else:
                    # Reset the env
                    time_this, ob_this_raw, is_terminal_cp = self._env.reset();
                    # Process and normalize the raw observation
                    ob_this_prcd = process_raw_state_cmbd(ob_this_raw, time_this, 
                                              self._env_st_yr, self._env_st_mn, 
                                              self._env_st_dy, self._env_st_wd, 
                                              min_max_limits); # 1-D list
                    # Get the history stacked state
                    self._histProcessor.reset();
                    ob_this_hist_prcd = self._histProcessor.\
                            process_state_for_network(ob_this_prcd) # 2-D array
            # Prepare for the training step
            R = 0 if is_terminal else sess.run(
                            self._value_pred,
                            feed_dict = {self._state_placeholder:ob_this_hist_prcd})
            traj_len = len(trajectory_list);
            act_idx_list = np.zeros(traj_len);
            q_true_list = np.zeros((traj_len, 1));
            state_list = np.zeros((traj_len, self._network_state_dim));
            for i in range(traj_len):
                traj_i_from_last = trajectory_list[traj_len - i - 1]; #(s_t, a_t, r_t);
                R = gamma * R + traj_i_from_last[2];
                act_idx_list[i] = traj_i_from_last[1];
                q_true_list[i, :] = R;
                state_list[i, :] = traj_i_from_last[0];
            # Perform training
            sess.run(_train_op, 
                     feed_dict = {self._q_true_placeholder: q_true_list,
                                  self._state_placeholder: state_list,
                                  self._action_idx_placeholder: act_idx_list});
            # ...
            is_terminal = is_terminal_cp;
            # Check whether training should stop
            if global_counter.value > t_max:
                coordinator.request_stop()
                
    
    def _select_sto_action(self, state, sess):
        """
        Given a state, run stochastic policy network to give an action.
        
        Args:
            state: np.ndarray, 1*m where m is the state feature dimension.
                Processed normalized state.
            sess: tf.Session.
                The tf session.
        
        Return: int 
            The action index.
        """
        softmax_a = sess.run(self._policy_pred, 
                             feed_dict={self._state_placeholder:state}).flatten();
        uni_rdm = np.random.uniform();
        imd_x = uni_rdm;
        for i in range(softmax_a.shape[-1]):
            imd_x -= softmax_a[i];
            if imd_x <= 0.0:
                return i;
    

class A3CAgent:
    """Class implementing A3C.


    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: rl.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: rl.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    action_size: int
      Number of actions.
    learning_rate: float, 0~1
    start_e: float, 0~1
      Start epsilon for linear decay e-greedy policy.
    end_e: float, 0~1
      End epsilon for linear decay e-greedy policy, or the epsilon
      for the e-greedy policy.
    num_steps: int
      The number of decay steps for the linear decay e-greedy policy.
    log_dir: string
      log file save directory
    """
    def __init__(self,
                 state_size,
                 window_len,
                 action_size,
                 vloss_frac, 
                 ploss_frac, 
                 hregu_frac,
                 
                 
                 preprocessor,
                 memory,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 eval_freq,
                 eval_epi_num,
                 batch_size,
                 
                 learning_rate,
                 start_e,
                 end_e,
                 num_steps,
                 log_dir,
                 save_freq):
        
        self._state_size = state_size;
        self._window_len = window_len;
        self._action_size = action_size;
        self._vloss_frac = vloss_frac;
        self._ploss_frac = ploss_frac;
        self._hregu_frac = hregu_frac;
        
        self._learning_rate = learning_rate;
        self._log_dir = log_dir;
        self._policy = Policy()
        self._uniformRandomPolicy = UniformRandomPolicy(action_size);
        self._linearDecayGreedyEpsilonPolicy = \
            LinearDecayGreedyEpsilonPolicy(start_e, end_e, num_steps);
        self._greedyPolicy = GreedyPolicy();
        self._num_burn_in = num_burn_in;
        self._preprocessor = preprocessor;
        self._memory = memory;
        self._train_freq = train_freq;
        self._batch_size = batch_size;
        self._gamma = gamma;
        self._target_update_freq = target_update_freq;
        self._save_freq = save_freq;
        self._eval_freq = eval_freq;
        self._eval_epi_num = eval_epi_num;
        self._mean_array = np.array([])
        self._std_array = np.array([])
        
    def compile(self, optimizer, loss_func, is_warm_start, model_dir):
        """
        This method sets up the required TF graph and operations.
        
        Args:
        
        Return:
        
        
        """
        g = tf.Graph();
        # Create the global network
        global_network = A3C_Network(g, 'global', state_dim, action_size)
        with g.as_default():
            # Generate placeholders state and "true" q values
            state_placeholder = tf.placeholder(tf.float32
                                             , shape=(None, 
                                                      self._state_size * 
                                                      self._window_len)
                                             , name='state_pl');
            q_true_placeholder = tf.placeholder(tf.float32
                                             , shape=(None, 1)
                                             , name='q_true_pl');
            # Build a Graph that computes predictions from the nn model.
            policy_pred, v_pred = create_model(state_placeholder
                                             , self._action_size
                                             , model_name='model');
            # Generate the tensor for one hot policy probablity
            action_idx_placeholder = tf.placeholder(tf.uint8
                                             , shape=(None)
                                             , name='action_idx_pl');
            pi_one_hot = tf.reduce_sum((tf.one_hot(action_idx_placeholder
                                                   , self._action_size) 
                                        * policy_pred), 1, True); 
            # Add to the Graph the Ops for loss calculation.
            loss = a3c_loss(q_true_placeholder, v_pred, policy_pred, pi_one_hot,
                            self._vloss_frac, self._ploss_frac, self._hregu_frac);

            # Add to the Graph the Ops that calculate and apply gradients.
            train_op = create_training_op(loss
                                          , optimizer
                                          , self._learning_rate);

            # Add to the Graph the Ops that calculate and apply gradients.
            train_op_1 = create_training_op(loss_1
                                          , optimizer
                                          , self._learning_rate);

            # Build the summary Tensor based on the TF collection of Summaries.
            summary = tf.summary.merge_all()
            
            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver();
            
            # Create a session for running Ops on the Graph.
            sess = tf.Session()
            
            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.summary.FileWriter(self._log_dir, sess.graph)
            
            # Init all trainable variables
            init_op = init_weights_uniform(g
                                           , 'q_network_0');
            init_op_1 = init_weights_uniform(g
                                           , 'q_network_1');
      
            #init cnn networks variables or warm start
            if not is_warm_start:
                sess.run(init_op);
                sess.run(init_op_1);
                #init remaining variables (probably those assocated with adam)
                sess.run(tf.variables_initializer(get_uninitialized_variables(sess)));
            else:
                saver.restore(sess, model_dir);
        self._state_placeholder = state_placeholder   
        self._q_placeholder = q_placeholder;
        self._q_pred_0 = q_pred_0;
        self._q_pred_1 = q_pred_1;
        self._sess = sess;
        self._train_op = train_op;
        self._train_op_1 = train_op_1;
        self._loss = loss;
        self._loss_1 = loss_1;
        self._saver = saver;
        self._summary = summary;
        self._summary_writer = summary_writer;
        self._g = g;
            


    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        return self._sess.run(self._q_pred_0,
                        feed_dict={self._state_placeholder:state});

    
    def calc_q_values_1(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        return self._sess.run(self._q_pred_1,
                        feed_dict={self._state_placeholder:state});

    
    def calc_mean_std(self):
        """
        Calculate mean and standanr deviation of all features

        Return
        ----------
        mean: numpy array
          The array of mean of each feature
        standard deviataion: numpy array
          The mean of standard deviataion of each feature

        """

        # get ob_next sets from memory
        memory_len = len(self._memory)
        all_obs_next = []
        col_len = len(self._memory[memory_len - 1].obs_nex)
      
        for i in range(memory_len):
            all_obs_next.append(self._memory[i].obs_nex)
       
        # cacualte average and standard diviation for each features   
        return (np.mean(np.array(all_obs_next).reshape(memory_len, 
                       col_len).transpose(), axis=1), 
                np.std(np.array(all_obs_next).reshape(memory_len, 
                       col_len).transpose(), axis=1))

          

    def fit(self, env, env_eval, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is Eplus environment. 
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """


   
        train_counter = 0;


        eval_res_hist = np.zeros((1,3));
         
        time_this, ob_this, is_terminal = env.reset();

        setpoint_this = ob_this[8:10]
        
                                                 
        this_ep_length = 0;
        flag_print_1 = True;
        flag_print_2 = True;
        action_counter = 0;

        for step in range(num_iterations):
            # select  network 1 and network 2 based on coin flip
            # coin 0, treated network 0 as  online network, coin 1, reverse
            coin = np.random.binomial(1, 0.5)
            #Check which stage is the agent at. If at the collecting stage,
            #then the actions will be random action.
            if step < self._num_burn_in:
                if flag_print_1:
                    logging.info ("Collecting samples to fill the replay memory...");
                    flag_print_1 = False;

                action_mem = self._uniformRandomPolicy.select_action();

                action = self._policy.process_action(setpoint_this, action_mem)

            else:  
   
                obs_this_net = self._preprocessor.process_observation_for_network(
                  ob_this, self._mean_array,  self._std_array)

                state_this_net = np.append(obs_this_net[0:10], obs_this_net[-1]).reshape(1,11)
                
                if flag_print_2:
                    logging.info ("Start training process...");
                    flag_print_2 = False;
                
                q_values = self.calc_q_values(state_this_net) + self.calc_q_values_1(state_this_net) 
                
                action_mem = self._linearDecayGreedyEpsilonPolicy.select_action(q_values, True),
      
                # covert command to setpoint action 
                action = self._policy.process_action(setpoint_this, action_mem[0])

            action_counter = action_counter + 1 if action_counter < 4 else 1
            time_next, ob_next, is_terminal = env.step(action)
            setpoint_next = ob_next[8:10]

      
            #check if exceed the max_episode_length
            if max_episode_length != None and \
                this_ep_length >= max_episode_length:
                is_terminal = True;
            self._memory.append(Sample(ob_this, action_mem, ob_next
                                       , is_terminal))
            self._mean_array, self._std_array = self.calc_mean_std()
            #Check which stage is the agent at. If at the training stage,
            #then do the training
            if step >= self._num_burn_in:
                #Check the train frequency
                if action_counter % self._train_freq == 0 \
                    and action_counter > 0:
                    action_counter = 0;
          #           #Eval the model
          #           if train_counter % self._eval_freq == 0:
          #               eval_res = self.evaluate(env_eval, self._eval_epi_num, 
          #                                    show_detail = True);
          #               eval_res_hist = np.append(eval_res_hist
          # , np.array([step
          # , eval_res[0], eval_res[1]]).reshape(1, 3)
          # , axis = 0);
          #               np.savetxt(self._log_dir + '/eval_res_hist.csv'
          #       , eval_res_hist, delimiter = ',');
          #               logging.info ('Global Step: %d, '%(step), 'evaluation average \
          #                      reward is %0.04f, average episode length is %d.'\
          #                          %eval_res);
                        
                    train_counter += 1;
                    #Sample from the replay memory
                    samples = self._preprocessor.process_batch(
                        self._memory.sample(self._batch_size), 
                        self._mean_array, self._std_array);
                    #Construct target values, one for each of the sample 
                    #in the minibatch
                    samples_x = None;
                    targets = None;
                    for sample in samples:
                        sample_s = np.append(sample.obs[0:10], sample.obs[-1]).reshape(1,11)
                        sample_s_nex = np.append(sample.obs_nex[0:10], 
                          sample.obs_nex[-1]).reshape(1,11)
                        sample_r = self._preprocessor.process_reward(
                          np.append(sample.obs_nex[0:11], sample.obs_nex[-1]))

                        if(coin == 0):
                            target = self.calc_q_values(sample_s); 
                            q_s_p = self.calc_q_values(sample_s_nex);
                            a_max = np.argmax(q_s_p); 
                        else:
                            target = self.calc_q_values_1(sample_s); 
                            q_s_p = self.calc_q_values_1(sample_s_nex);
                            a_max = np.argmax(q_s_p); 

                        if sample.is_terminal:
                            target[0, sample.a] = sample_r;
                        else:
                            if(coin == 0):
                                target[0, sample.a] = (sample_r
                                                + self._gamma 
                                                * self.calc_q_values_1(
                                                    sample_s_nex)[0, a_max]);
                            else:
                                target[0, sample.a] = (sample_r
                                                + self._gamma 
                                                * self.calc_q_values(
                                                    sample_s_nex)[0, a_max]);

                        if targets is None:
                            targets = target;
                        else:
                            targets = np.append(targets, target, axis = 0);
                        if samples_x is None:
                            samples_x = sample_s;
                        else:
                            samples_x = np.append(samples_x, sample_s, axis = 0);
                    #Run the training
                    feed_dict = {self._state_placeholder:samples_x
                                ,self._q_placeholder:targets}

                    if(coin == 0):
                        sess_res = self._sess.run([self._train_op, self._loss]
                                              , feed_dict = feed_dict);
                    else:
                        sess_res = self._sess.run([self._train_op_1, self._loss_1]
                                              , feed_dict = feed_dict);

                    #Save the parameters
                    if train_counter % self._save_freq == 0 or step + 1 == num_iterations:
                        checkpoint_file = os.path.join(self._log_dir
                                                       , 'model_data/model.ckpt');
                        self._saver.save(self._sess
                                         , checkpoint_file, global_step=step);
                    
                    if train_counter % 100 == 0:
                        logging.info ("Global Step %d: loss %0.04f"%(step, sess_res[1]));
                        # Update the events file.
                        summary_str = self._sess.run(self._summary, feed_dict=feed_dict)
                        self._summary_writer.add_summary(summary_str, train_counter);
                        self._summary_writer.flush()
            
            
            #check whether to start a new episode
            if is_terminal:
                time_this, ob_this, is_terminal = env.reset();
                setpoint_this = ob_this[8:10]
      

                #state_this_net = self._preprocessor.process_state_for_network(state_this, 
                 #      mean_state_array, std_state_array);
      
                this_ep_length = 0;
                action_counter = 0;
            else:
                ob_this = ob_next
                setpoint_this = setpoint_next
                this_ep_length += 1;
           
                
            

    def evaluate(self, env, num_episodes, max_episode_length=None
                 , show_detail = False):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        episode_counter = 1;
        average_reward = 0;
        average_episode_length = 0;
        time_this, ob_this, is_terminal = env.reset();
        
        state_this, setpoint_this, reward = (self._preprocessor
          .process_observation(time_this, ob_this))

        # get mean_array and std_array of state
        mean_state_array = np.delete(self._mean_array, [10,11]) 
        std_state_array = np.delete(self._std_array, [10,11])

        # get mean_array and std_array for reward
        mean_reward_array = np.array([self._mean_array[10], self._mean_array[-1]]) 
        std_reward_array = np.array([self._std_array[10], self._std_array[-1]])

        state_this_net = self._preprocessor.process_state_for_network(state_this, 
          mean_state_array, std_state_array);


        
        this_ep_reward = 0;
        this_ep_length = 0;
        while episode_counter <= num_episodes:
            coin = np.random.binomial(1, 0.5)
            q_values = self.calc_q_values(state_this_net) 
            q_values_1 = self.calc_q_values_1(state_this_net)
            if(coin == 0):
              command = [self._linearDecayGreedyEpsilonPolicy.select_action(q_values, True),
                  self._linearDecayGreedyEpsilonPolicy.select_action(q_values_1, True)]
            else:
              command = [self._linearDecayGreedyEpsilonPolicy.select_action(q_values_1, True),
                  self._linearDecayGreedyEpsilonPolicy.select_action(q_values, True)]

            # covert command to setpoint action 
            action = self._policy.process_action(setpoint_this, command)

            time_next, ob_next, is_terminal = env.step(action)

            state_next,setpoint_next, reward = self._preprocessor.process_observation(time_next, ob_next)

            state_next_net = self._preprocessor.process_state_for_network(state_next, 
                   mean_state_array, std_state_array);
      
            reward_processed = self._preprocessor.process_reward(reward, mean_reward_array, std_reward_array)


            this_ep_reward += reward_processed;

    
            #Check if exceed the max_episode_length
            if max_episode_length is not None and \
                this_ep_length >= max_episode_length:
                is_terminal = True;
            #Check whether to start a new episode
            if is_terminal:
                time_this, ob_this, is_terminal = env.reset();
                state_this, setpoint_this, reward = (self._preprocessor
                  .process_observation(time_this, ob_this))

                state_this_net = self._preprocessor.process_state_for_network(state_this, 
                    mean_state_array, std_state_array);
   
                average_reward = (average_reward * (episode_counter - 1) 
                                  + this_ep_reward) / episode_counter;
                average_episode_length = (average_episode_length 
                                          * (episode_counter - 1) 
                                          + this_ep_length) /  episode_counter;
                
                episode_counter += 1;
                if show_detail:
                    logging.info ('Episode ends. Cumulative reward is %0.04f '
                        'episode length is %d, average reward by now is %0.04f,'
                        ' average episode length by now is %d.' %(this_ep_reward,
                                                                  this_ep_length,
                                                                  average_reward,
                                                          average_episode_length));
                this_ep_length = 0;
                this_ep_reward = 0;
                
            else:
                state_this_net = state_next_net;
                this_ep_length += 1;
        return (average_reward, average_episode_length);
