import numpy as np
import tensorflow as tf
from a3c_v0_1.actions import action_map
from a3c_v0_1.preprocessors import HistoryPreprocessor, process_raw_state_cmbd
from a3c_v0_1.state_index import *
from a3c_v0_1.env_interaction import IWEnvInteract

ACTION_MAP = action_map;

class A3CEval_multiagent:
    def __init__(self, sess, global_network, env, num_episodes, window_len, 
                 e_weight, p_weight, agent_num):
        """
        This is the class for evaluation under the multi-zone control mode. 

        Args:
            sess: tf.Session
                The tf.Session object.
            global_network: tf.Tensor
                The global network object.
            env: str
                The test environment name.
            num_episodes: int
                The number of episode
            window_len: int
                The length of the state stacking window.
            e_weight: float
                The penalty weight on energy.
            p_weight: float
                The penalty weight on comfort.
            agent_num: int
                The number of zones to be controlled. 

        """
        self._sess = sess;
        self._global_network = global_network;
        self._env = env;
        self._num_episodes = num_episodes;
        self._histProcessor_list = [HistoryPreprocessor(window_len) for _ in range(agent_num)];
        # Prepare env-related information
        self._env_st_yr = env.start_year;
        self._env_st_mn = env.start_mon;
        self._env_st_dy = env.start_day;
        self._env_st_wd = env.start_weekday;
        env_state_limits = env.min_max_limits;
        env_state_limits.insert(0, (0, 23)); # Add hour limit
        env_state_limits.insert(0, (0, 6)); # Add weekday limit
        self._pcd_state_limits = np.transpose(env_state_limits);
        self._e_weight = e_weight;
        self._p_weight = p_weight;
        self._agent_num = agent_num;
        

    def evaluate(self, local_logger, reward_mode, action_space_name, 
                 ppd_penalty_limit, STPT_LIMITS, raw_state_process_func):
        """
        This method do the evaluation for the trained agent. 

        Args:
            local_logger: util.Logger
                The logger object for local logging. 
            reward_mode: str
                The reward mode.
            action_space_name: str
                The name for the action space.
            ppd_penalty_limit: float
                If the PPD exceed this limit, then it will be set to 1.0.
        Return: tuple
            The average reward for each controlled zone. 
        """
        action_space = ACTION_MAP[action_space_name];
        episode_counter = 1;
        average_reward = np.zeros(self._agent_num);
        # Reset the env
        time_this, ob_this_raw_all, is_terminal = self._env.reset();
        #ob_this_raw_all[-1] = 0; #print
        # Extract state for each agent
        ob_this_raw_list = [self._get_agent_state(ob_this_raw_all, agent_id = agent_i) for agent_i in range(self._agent_num)];
        # Get the history stacked state for each agent
        ob_this_hist_prcd_list = [];
        for agent_i in range(self._agent_num):
            # Process and normalize the raw observation
            ob_this_raw_agent_i = ob_this_raw_list[agent_i];
            ob_this_raw_agent_i = raw_state_process_func(ob_this_raw_agent_i);
            ob_this_prcd_agent_i = process_raw_state_cmbd(ob_this_raw_agent_i, [time_this], 
                                                        self._env_st_yr, self._env_st_mn, 
                                                        self._env_st_dy, self._env_st_wd, 
                                                        self._pcd_state_limits); # 1-D list
            histProcessor_i = self._histProcessor_list[agent_i]; 
            histProcessor_i.reset();
            ob_this_hist_prcd_agent_i = histProcessor_i.process_state_for_network(ob_this_prcd_agent_i) # 2-D array
            ob_this_hist_prcd_list.append(ob_this_hist_prcd_agent_i);
        # Do the eval
        this_ep_reward = np.zeros(self._agent_num);
        while episode_counter <= self._num_episodes:
            # Get the action
            action_list = [];
            for agent_i in range(self._agent_num):
                action_raw_idx_i = self._select_sto_action(ob_this_hist_prcd_list[agent_i]);
                action_raw_tup_i = action_space[action_raw_idx_i];
                cur_htStpt_i = ob_this_raw_list[agent_i][HTSP_RAW_IDX];
                cur_clStpt_i = ob_this_raw_list[agent_i][CLSP_RAW_IDX];
                action_stpt_prcd_i, action_effec_i = get_legal_action(cur_htStpt_i, cur_clStpt_i, action_raw_tup_i, STPT_LIMITS);
                action_stpt_prcd_i = list(action_stpt_prcd_i);
                action_list.extend(action_stpt_prcd_i);
            # Perform the action
            time_next, ob_next_raw_all, is_terminal = self._env.step(action_list);
            #ob_next_raw_all[-1] = 0;# print
            # Extract the state for each agent
            ob_next_raw_list = [self._get_agent_state(ob_next_raw_all, agent_id = agent_i) for agent_i in range(self._agent_num)];
            # Process the state and normalize it
            ob_next_raw_list = [raw_state_process_func(ob_next_raw_agent_i) for ob_next_raw_agent_i in ob_next_raw_list];
            ob_next_prcd_list = [process_raw_state_cmbd(ob_next_raw_agent_i, [time_next], self._env_st_yr, self._env_st_mn, 
                                                        self._env_st_dy, self._env_st_wd, self._pcd_state_limits) \
                                for ob_next_raw_agent_i in ob_next_raw_list];
            # Get the reward
            reward_next_list = [];
            for agent_i in range(self._agent_num):
                ob_next_prcd_i = ob_next_prcd_list[agent_i];
                normalized_hvac_energy_i = ob_next_prcd_i[HVACE_RAW_IDX + 2];
                normalized_ppd_i = ob_next_prcd_i[ZPPD_RAW_IDX + 2];
                occupancy_status = ob_next_prcd_i[ZPCT_RAW_IDX + 2];
                reward_next_i = get_reward(normalized_hvac_energy_i, normalized_ppd_i, self._e_weight, self._p_weight, occupancy_status,
                                           reward_mode, ppd_penalty_limit);
                reward_next_list.append(reward_next_i);
            this_ep_reward += reward_next_list;
            # Get the history stacked state
            ob_next_hist_prcd_list = [self._histProcessor_list[agent_i].process_state_for_network(ob_next_prcd_list[agent_i])\
                                      for agent_i in range(self._agent_num)] # 2-D array
            # Check whether to start a new episode
            if is_terminal:
                time_this, ob_this_raw_all, is_terminal = self._env.reset();
                # Extract state for each agent
                ob_this_raw_list = [self._get_agent_state(ob_this_raw_all, agent_id = agent_i) for agent_i in range(self._agent_num)];
                # Get the history stacked state for each agent
                ob_this_hist_prcd_list = [];
                for agent_i in range(self._agent_num):
                    # Process and normalize the raw observation
                    ob_this_raw_agent_i = raw_state_process_func(ob_this_raw_list[agent_i]);
                    ob_this_prcd_agent_i = process_raw_state_cmbd(ob_this_raw_agent_i, [time_this], 
                                                        self._env_st_yr, self._env_st_mn, 
                                                        self._env_st_dy, self._env_st_wd, 
                                                        self._pcd_state_limits); # 1-D list
                    histProcessor_i = self._histProcessor_list[agent_i]; 
                    histProcessor_i.reset();
                    ob_this_hist_prcd_agent_i = histProcessor_i.process_state_for_network(ob_this_prcd_agent_i) # 2-D array
                    ob_this_hist_prcd_list.append(ob_this_hist_prcd_agent_i);
                # Update the average reward
                average_reward = (average_reward * (episode_counter - 1) + this_ep_reward) / episode_counter;
                local_logger.info('Evaluation: average reward by now is ' + str(average_reward));
                episode_counter += 1;
                this_ep_reward = np.zeros(self._agent_num);
                 
            else:
                time_this = time_next;
                ob_this_hist_prcd_list = ob_next_hist_prcd_list;
                ob_this_raw_list = ob_next_raw_list;
                
        return (average_reward);
    
    def _select_sto_action(self, state):
        """
        Given a state, run stochastic policy network to give an action.
        
        Args:
            state: np.ndarray, 1*m where m is the state feature dimension.
                Processed normalized state.
        
        Return: int 
            The action index.
        """
        
        softmax_a = self._sess.run(self._global_network.policy_pred, 
                        feed_dict={self._global_network.state_placeholder:state,
                                   self._global_network.keep_prob: 1.0})\
                        .flatten();
        ### DEBUG
        dbg_rdm = np.random.uniform();
        if dbg_rdm < 0.01:
            print ('softmax', softmax_a)
        uni_rdm = np.random.uniform();
        imd_x = uni_rdm;
        for i in range(softmax_a.shape[-1]):
            imd_x -= softmax_a[i];
            if imd_x <= 0.0:
                return i;

    def _get_agent_state(self, ob_this_raw_all, agent_id):
        """
        This method returns the state corresponding to each controlled zone. 

        Args:
            ob_this_raw_all: list
                All state observations for this time step.
            agent_id: int
                The # of the agent.
        Return: list
            The state observation for the agent.  
        """
        ret = ob_this_raw_all[:DIS_RAW_IDX + 1]; # Copy the weather observations
        ret.extend(ob_this_raw_all[DIS_RAW_IDX + ZN_OB_NUM * agent_id + 1: DIS_RAW_IDX + ZN_OB_NUM * agent_id + 1 + ZN_OB_NUM]);
        ret.append(ob_this_raw_all[-1]);
        return ret;

class A3CEval:
    def __init__(self, sess, global_network, env, num_episodes, window_len, 
                 forecast_len, e_weight, p_weight):
        """
        This is the class for evaluation under the single-zone control mode. 

        Args:
            sess: tf.Session
                The tf.Session object.
            global_network: tf.Tensor
                The global network object.
            env: str
                The test environment name.
            num_episodes: int
                The number of episode
            window_len: int
                The length of the state stacking window.
            e_weight: float
                The penalty weight on energy.
            p_weight: float
                The penalty weight on comfort.

        """
        self._sess = sess;
        self._global_network = global_network;
        # Print debug
        #print ('Eval global network ..............................................................')
        #global_collection = self._global_network.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
        #                                   'global');
        #print (sess.run(global_collection[0]));
        self._env = env;
        self._num_episodes = num_episodes;
        self._histProcessor = HistoryPreprocessor(window_len, forecast_len);
        # Prepare env-related information
        self._env_st_yr = env.start_year;
        self._env_st_mn = env.start_mon;
        self._env_st_dy = env.start_day;
        self._env_st_wd = env.start_weekday;
        env_state_limits = env.min_max_limits;
        env_state_limits.insert(0, (0, 23)); # Add hour limit
        env_state_limits.insert(0, (0, 6)); # Add weekday limit
        self._pcd_state_limits = np.transpose(env_state_limits);
        self._e_weight = e_weight;
        self._p_weight = p_weight;
        

    def evaluate(self, local_logger, action_space_name, reward_func, rewardArgs, 
                action_func, action_limits, raw_state_process_func, debug_log_prob):
        """
        This method do the evaluation for the trained agent. 

        Args:
            local_logger: util.Logger
                The logger object for local logging. 
            reward_mode: str
                The reward mode.
            action_space_name: str
                The name for the action space.
            ppd_penalty_limit: float
                If the PPD exceed this limit, then it will be set to 1.0.
        Return: tuple
            The average reward for each controlled zone. 
        """
        action_space = ACTION_MAP[action_space_name];
        action_size = len(action_space)
        episode_counter = 1;
        average_reward = 0;
        #average_max_ppd = 0;
        env_interact_wrapper = IWEnvInteract(self._env, raw_state_process_func);
        # Reset the env
        time_this, ob_this_raw, is_terminal = env_interact_wrapper.reset();
        # Process and normalize the raw observation
        ob_this_prcd = process_raw_state_cmbd(ob_this_raw, [time_this], 
                                              self._env_st_yr, self._env_st_mn, 
                                              self._env_st_dy, self._env_st_wd, 
                                              self._pcd_state_limits); # 1-D list
        # Get the history stacked state
        self._histProcessor.reset();
        ob_this_hist_prcd = self._histProcessor.\
                            process_state_for_network(ob_this_prcd) # 2-D array
        # Do the eval
        this_ep_reward = 0;
        #this_ep_max_ppd = 0;
        while episode_counter <= self._num_episodes:
            dbg_rdm = np.random.uniform();
            #################FOR DEBUG#######################
            is_dbg_out = False;
            noForecastDim = 13;
            if dbg_rdm < debug_log_prob:
                is_dbg_out = True;
            if is_dbg_out:
                local_logger.debug('Observation this: %s' %(ob_this_raw[0: noForecastDim]));
                local_logger.debug('Observation forecast: %s' %(ob_this_raw[noForecastDim:]));
            #################################################
            # Get the action
            action_raw_out = self._select_sto_action(ob_this_hist_prcd, local_logger, is_dbg_out);
            action_raw_idx = action_raw_out if isinstance(action_raw_out, int) else action_raw_out[0]
            if action_raw_idx is not None:
                action_raw_tup = action_space[action_raw_idx];
            else:
                # Select action returns None, indicating the net work output is not valid
                random_act_idx = np.random.choice(action_size)
                action_raw_idx = random_act_idx;
                action_raw_tup = action_space[random_act_idx];
                local_logger.warning('!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!\n'
                                           'Select action function returns None, indicating the network output may not be valid!\n'
                                           'Network output is %s.'
                                           'A random action is taken instead, index is %s.'
                                           '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                                            %(action_raw_out[1], random_act_idx));

            action_stpt_prcd, action_effect_idx = action_func(action_raw_tup, action_raw_idx, action_limits, ob_this_raw);
            action_stpt_prcd = list(action_stpt_prcd);
            # Perform the action
            time_next, ob_next_raw, is_terminal = env_interact_wrapper.step(action_stpt_prcd);
            # Process and normalize the raw observation
            ob_next_prcd = process_raw_state_cmbd(ob_next_raw, [time_next], 
                                              self._env_st_yr, self._env_st_mn, 
                                              self._env_st_dy, self._env_st_wd, 
                                              self._pcd_state_limits); # 1-D list
            # Get the reward
            reward_next = reward_func(ob_next_prcd, self._e_weight, self._p_weight, *rewardArgs);
            this_ep_reward += reward_next;
            #this_ep_max_ppd = max(normalized_ppd if occupancy_status > 0 else 0,
            #                      this_ep_max_ppd);
            # Get the history stacked state
            ob_next_hist_prcd = self._histProcessor.\
                            process_state_for_network(ob_next_prcd) # 2-D array
            # Check whether to start a new episode
            if is_terminal:
                # Update the average reward
                average_reward = (average_reward * (episode_counter - 1) 
                                  + this_ep_reward) / episode_counter;
                #average_max_ppd = (average_max_ppd * (episode_counter - 1)
                #                  + this_ep_max_ppd) / episode_counter;
                local_logger.info('Evaluation: average reward by now is %0.04f'
                                  %(average_reward));
                episode_counter += 1;
                if episode_counter <= self._num_episodes:
                    time_this, ob_this_raw, is_terminal = env_interact_wrapper.reset();
                    # Process and normalize the raw observation
                    ob_this_prcd = process_raw_state_cmbd(ob_this_raw, [time_this], 
                                              self._env_st_yr, self._env_st_mn, 
                                              self._env_st_dy, self._env_st_wd, 
                                              self._pcd_state_limits); # 1-D list
                    # Get the history stacked state
                    self._histProcessor.reset();
                    ob_this_hist_prcd = self._histProcessor.\
                                process_state_for_network(ob_this_prcd) # 2-D array
                
                    this_ep_reward = 0;
                    #this_ep_max_ppd = 0;
                 
            else:
                time_this = time_next;
                ob_this_hist_prcd = ob_next_hist_prcd;
                ob_this_raw = ob_next_raw;
                
        return (average_reward);
    
    def _select_sto_action(self, state, local_logger, is_dbg_out):
        """
        Given a state, run stochastic policy network to give an action.
        
        Args:
            state: np.ndarray, 1*m where m is the state feature dimension.
                Processed normalized state.
        
        Return: int 
            The action index.
        """
        
        softmax_a = self._sess.run(self._global_network.policy_pred, 
                        feed_dict={self._global_network.state_placeholder:state,
                                   self._global_network.keep_prob: 1.0})\
                        .flatten();
        ### DEBUG
        uni_rdm = np.random.uniform();
        if is_dbg_out:
            local_logger.info('Softmax %s, sampled %s'%(softmax_a, uni_rdm))
        imd_x = uni_rdm;
        for i in range(softmax_a.shape[-1]):
            imd_x -= softmax_a[i];
            if imd_x <= 0.0:
                return i;
        return (None, softmax_a); # Return if network output is not valid
