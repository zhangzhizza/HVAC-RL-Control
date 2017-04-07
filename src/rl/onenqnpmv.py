"""Main OneDQN agent."""
import os
import logging
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import rl as tfrl
import rl.utils
from rl.policy import (Policy, UniformRandomPolicy, GreedyEpsilonPolicy
                               , GreedyPolicy, LinearDecayGreedyEpsilonPolicy)
from rl.utils import (init_weights_uniform, get_hard_target_model_updates
                              , get_uninitialized_variables)
from rl.core import Sample

logging.getLogger().setLevel(logging.INFO)

def create_model(input_state, num_actions,
                 model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    input_state: state placeholder.
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
  
   
    """

    with tf.name_scope('hidden1'):
        hidden1 = Dense(100, activation='sigmoid')(input_state)
    with tf.name_scope('output'):
        output = Dense(num_actions, activation='softmax')(hidden1)
    return output;


def create_training_op(loss, optimizer, learning_rate):
    """
    Some of the codes are modified based on Tensorflow sample code from
    https://www.tensorflow.org/versions/r0.10/tutorials/mnist/tf/
    
    Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    Args:
        loss: Loss tensor.
        learning_rate: The learning rate to use for gradient descent.
        optimizer: tf.train.Optimizer object

    Returns:
        train_op: The Op for training.
     """
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    _optimizer = optimizer(learning_rate = learning_rate)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    grads_and_vars = _optimizer.compute_gradients(loss);
    train_op = _optimizer.apply_gradients(grads_and_vars
                                        , global_step=global_step)
    return train_op;
    

class OneNQNAgent:
    """Class implementing One NQN.


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
    state_size:int 
      Number of features in the state.
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
                 train_set_size,
                 preprocessor,
                 memory,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 eval_freq,
                 eval_epi_num,
                 batch_size,
                 state_size,
                 action_size,
                 learning_rate,
                 start_e,
                 end_e,
                 num_steps,
                 log_dir,
                 save_freq):
        self._train_set_size = train_set_size;
        self._state_size = state_size;
        self._action_size = action_size;
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


        
    def compile(self, optimizer, loss_func, is_warm_start, model_dir = None):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        g = tf.Graph();
        with g.as_default():
        # Generate placeholders for the x and y.
            state_placeholder = tf.placeholder(tf.float32
                                             , shape=(None, self._state_size)
                                             , name='state_pl');

            q_placeholder = tf.placeholder(tf.float32
                                           , shape=(None, self._action_size)
                                           , name='q_pl');

            # Build a Graph that computes predictions from the cnn model.
            q_pred_0 = create_model(state_placeholder
                                  , self._action_size
                                  , model_name='q_network_0');
            q_pred_1 = create_model(state_placeholder
                                  , self._action_size
                                  , model_name='q_network_1');
            
            # Add to the Graph the Ops for loss calculation.
            loss = loss_func(q_placeholder, q_pred_0, max_grad=1.);

            # Add to the Graph the Ops that calculate and apply gradients.
            train_op = create_training_op(loss
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
                                           , 'q_network_0')

            copy_op = get_hard_target_model_updates(g
                                                    , 'q_network_1'
                                                    , 'q_network_0');
            #init cnn networks variables or warm start
            if not is_warm_start:
                sess.run(init_op);
                sess.run(copy_op);
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
        self._loss = loss;
        self._hard_copy_to_target_op = copy_op;
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

    def select_action(self, state, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        if kwargs['stage'] == 'collecting':
            return self._uniformRandomPolicy.select_action();
        elif kwargs['stage'] == 'training':
            q_values = self.calc_q_values(state);
            return self._linearDecayGreedyEpsilonPolicy.select_action(q_values
                                                                      , True);
        elif kwargs['stage'] == 'testing':
            q_values = self.calc_q_values(state);
            return self._linearDecayGreedyEpsilonPolicy.select_action(q_values
                                                                      , False);
        elif kwargs['stage'] == 'greedy':
            q_values = self.calc_q_values(state);
            return self._greedyPolicy.select_action(q_values);
            

    def update_policy(self):
        """
        Update the target network parameter.
        """
        self._sess.run(self._hard_copy_to_target_op);

    
    def calc_trainSet(self, env):
        """
        1. Filling states in training set, 
        which is used for sample standarization
        2. Calculate mean and standanr deviation of all features
 
         Note: training set should be released after using to save memory
         so a training set class is not designed as replay memory
 
         Parameters
         ----------
         env: gym.Env
           This is Eplus environment. 
  
         Parameters
         ----------
         mean: numpy array
           The array of mean of each feature
         standard deviataion: numpy array
           The mean of standard deviataion of each feature
  
        """
 
        time_this, ob_this, is_terminal = env.reset()

        ob_this = self._preprocessor.process_observation(time_this, ob_this)

        setpoint_this = ob_this[8:10]
 
         # save the first state in the training set 
        training_set = np.array(ob_this)
   
   
        for step in range(1, self._train_set_size):

          # get action command 
          command = self._uniformRandomPolicy.select_action()
          # covert command to setpoint action 
          action = self._policy.process_action(setpoint_this, command)
          # take action, get new observation 
          time_next, ob_next, is_terminal = env.step(action)

          ob_next = self._preprocessor.process_observation(time_next, ob_next)
            
          setpoint_next = ob_next[8:10]
          
          training_set = np.append(training_set, ob_next)
           
          if is_terminal:
              time_this, ob_this, is_terminal = env.reset()

              ob_this = self._preprocessor.process_observation(time_this, ob_this)

              setpoint_this = ob_this[8:10]
          else:
              ob_this = ob_next
              setpoint_this = setpoint_next
              time_this = time_next
        
        # cacualte average and standard diviation for each features   
        return (np.mean(training_set.reshape(self._train_set_size, 
                        len(ob_next)).transpose(), axis=1), 
                 np.std(training_set.reshape(self._train_set_size, 
                        len(ob_next)).transpose(), axis=1))


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
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """

        # caculate mean and standard deviation
        self._mean_array, self._std_array = self.calc_trainSet(env)
    
        train_counter = 0;
        eval_res_hist = np.zeros((1,3));

        time_this, ob_this, is_terminal = env.reset()

        ob_this = self._preprocessor.process_observation(time_this, ob_this)

        setpoint_this = ob_this[8:10]
                                                    
        this_ep_length = 0;
        flag_print_1 = True;
        flag_print_2 = True;
        action_counter = 0;
   
        for step in range(num_iterations):
            #Check which stage is the agent at. If at the collecting stage,
            #then the actions will be random action.
            if step <= self._num_burn_in:
                if flag_print_1:
                    logging.info ("Collecting samples to fill the replay memory...");
                    flag_print_1 = False;

                action_mem = self._uniformRandomPolicy.select_action()
                action = self._policy.process_action(setpoint_this, action_mem)

            else:

                if flag_print_2:
                    logging.info ("Start training process...");
                    flag_print_2 = False;

                obs_this_net = self._preprocessor.process_observation_for_network(
                ob_this, self._mean_array,  self._std_array)
         
                state_this_net = np.append(obs_this_net[0:11], obs_this_net[12:]).reshape(1,14)

                action_mem = self.select_action(state_this_net, stage = 'training')
                # covert command to setpoint action 
                action = self._policy.process_action(setpoint_this, action_mem)

            action_counter = action_counter + 1 if action_counter < 4 else 1;

            time_next, ob_next, is_terminal = env.step(action)
            ob_next = self._preprocessor.process_observation(time_next, ob_next)
            
            setpoint_next = ob_next[8:10]

            
            #check if exceed the max_episode_length
            if max_episode_length != None and \
                this_ep_length >= max_episode_length:
                is_terminal = True;

            #save sample into memory 
            self._memory.append(Sample(ob_this, action_mem, ob_next
                                       , is_terminal))

            
            #Check which stage is the agent at. If at the training stage,
            #then do the training
            if step > self._num_burn_in:
                #Check the train frequency
                if action_counter % self._train_freq == 0 \
                    and action_counter > 0:
                    action_counter = 0;
                    #Eval the model
                    if train_counter % self._eval_freq == 0:
                        eval_res = self.evaluate(env_eval, self._eval_epi_num
                                             , show_detail = True);
                        eval_res_hist = np.append(eval_res_hist
          , np.array([step
          , eval_res[0], eval_res[1]]).reshape(1, 3)
          , axis = 0);
                        np.savetxt(self._log_dir + '/eval_res_hist.csv'
                , eval_res_hist, delimiter = ',');
                        logging.info ('Global Step: %d, '%(step), 'evaluation average \
                               reward is %0.04f, average episode length is %d.'\
                                   %eval_res);
                        
                    
                    #Sample from the replay memory
                    samples = self._preprocessor.process_batch(
                        self._memory.sample(self._batch_size), 
                        self._mean_array, self._std_array);
                    #Construct target values, one for each of the sample 
                    #in the minibatch
                    samples_x = None;
                    targets = None;
                    for sample in samples:
                        sample_s = np.append(sample.obs[0:11], sample.obs[12:]).reshape(1,14)
                        sample_s_nex = np.append(sample.obs_nex[0:11], 
                          sample.obs_nex[12:]).reshape(1,14)
                        sample_r = self._preprocessor.process_reward(sample.obs_nex[10:13])

                        target = self.calc_q_values(sample_s);
                        a_max = self.select_action(sample_s_nex, stage = 'greedy');
                        if sample.is_terminal:
                            target[0, sample.a] = sample_r;
                        else:
                            target[0, sample.a] = (sample_r
                                                + self._gamma 
                                                * self.calc_q_values_1(
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
                    sess_res = self._sess.run([self._train_op, self._loss]
                                              , feed_dict = feed_dict);
                    
                    #Update the target parameters
                    if train_counter % self._target_update_freq == 0:
                        self.update_policy();
                        logging.info('Global Step %d: update target network.' 
                                     %(step));
                    #Save the parameters
                    if train_counter % self._save_freq == 0 or step + 1 == num_iterations:
                        checkpoint_file = os.path.join(self._log_dir
                                                       , 'model_data/model.ckpt');
                        self._saver.save(self._sess
                                         , checkpoint_file, global_step=step);
                    
                    if train_counter % 100 == 0:
                        print(self._q_pred_0)
                        print(self._q_pred_1)
                        logging.info ("Global Step %d: loss %0.04f"%(step, sess_res[1]));
                        # Update the events file.
                        summary_str = self._sess.run(self._summary, feed_dict=feed_dict)
                        self._summary_writer.add_summary(summary_str, train_counter);
                        self._summary_writer.flush()
                    
                    train_counter += 1;
            
            #check whether to start a new episode
            if is_terminal:
                time_this, ob_this, is_terminal = env.reset()
                ob_this = self._preprocessor.process_observation(time_this, ob_this)
                setpoint_this = ob_this[8:10]

                this_ep_length = 0;
                action_counter = 0;
            else:
                ob_this = ob_next
                setpoint_this = setpoint_next
                time_this = time_next
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
        time_this, ob_this, is_terminal = env.reset()

        ob_this = self._preprocessor.process_observation(time_this, ob_this)

        obs_this_net = self._preprocessor.process_observation_for_network(
                  ob_this, self._mean_array,  self._std_array)

        state_this_net = np.append(obs_this_net[0:11], obs_this_net[12:]).reshape(1,14)
        setpoint_this = ob_this[8:10]
        
        this_ep_reward = 0;
        this_ep_length = 0;
        while episode_counter <= num_episodes:
            action_mem = self.select_action(state_this_net, stage = 'testing');
            # covert command to setpoint action 
            action = self._policy.process_action(setpoint_this, action_mem)

            time_next, ob_next, is_terminal = env.step(action)

            ob_next = self._preprocessor.process_observation(time_next, ob_next)

            setpoint_next = ob_next[8:10]

            obs_next_net = self._preprocessor.process_observation_for_network(
                  ob_next, self._mean_array,  self._std_array)
  

            state_next_net = np.append(obs_next_net[0:11], obs_next_net[12:]).reshape(1,14)
    
            
            #10:PMV, 11: Occupant number , -2: power
            reward = self._preprocessor.process_reward(obs_next_net[10:13])
            this_ep_reward += reward;

 
            #Check if exceed the max_episode_length
            if max_episode_length is not None and \
                this_ep_length >= max_episode_length:
                is_terminal = True;
            #Check whether to start a new episode
            if is_terminal:
                time_this, ob_this, is_terminal = env.reset()
                setpoint_this = ob_this[8:10]

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
                ob_this = ob_next
                setpoint_this = setpoint_next
                time_this = time_next
                this_ep_length += 1;
        return (average_reward, average_episode_length);
