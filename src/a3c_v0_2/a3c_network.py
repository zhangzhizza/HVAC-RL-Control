import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Dense, Flatten, Input,
                          Permute)
from keras.models import Model

from a3c_v0_2.objectives import a3c_loss

NN_WIDTH = 512;

class A3C_Network:
    """
    The class that creates the policy and value network for the A3C. 
    """
    
    def __init__(self, graph, scope_name, state_dim, action_size, net_length):
        """
        Constructor.
        
        Args:
            graph: tf.Graph
                The computation graph.
            scope_name: String
                The name of the scope.
            state_dim, action_size: int
                The number of the state dimension and number of action choices. 
        """
        with graph.as_default(), tf.name_scope(scope_name):
            # Generate placeholder for state
            self._state_placeholder = tf.placeholder(tf.float32,
                                                     shape=(None, 
                                                            state_dim),
                                                     name='state_pl');
            self._keep_prob = tf.placeholder(tf.float32, name='keep_prob');
            # Build the operations that computes predictions from the nn model.
            self._policy_pred, self._v_pred, self._shared_layer= \
                self._create_model(self._state_placeholder, self._keep_prob, action_size, net_length);
            
    @property
    def state_placeholder(self):
        return self._state_placeholder;

    @property
    def keep_prob(self):
        return self._keep_prob;
    
    @property
    def policy_pred(self):
        return self._policy_pred;
    
    @property
    def value_pred(self):
        return self._v_pred;

    @property
    def shared_layer(self):
        return self._shared_layer;

    def _create_model(self, input_state, keep_prob, num_actions, net_length): 
        """
        Create the model for the policy network and value network.
        The policy network and the value network share the model for feature
        extraction from the raw state, and then the policy network uses a 
        softmax layer to provide the probablity of taking each action, and the 
        value network uses a linear layer to provide a scalar for the value. 
        
        Args:
            input_state: tf tensor or placeholder.
                Represent the input to the network, which is the state observation.
            keep_prob: tf tensor or placeholder.
                The 1 - dropout probability.
            num_actions: int.
                Number of actions.
        
        Return: (tf tensor, tf tensor)
            The policy and the value for the state. 
            
        """
        with tf.name_scope('shared_layers'):
            # Dropout layer for the first relu layer.
            layer = tf.nn.dropout(input_state, keep_prob);
            for _ in range(net_length):
                layer = Dense(NN_WIDTH, activation = 'relu')(layer);
        with tf.name_scope('policy_network'):
            policy = Dense(num_actions, activation = 'softmax')(layer);
        with tf.name_scope('value_network'):
            value = Dense(1)(layer);
        return (policy, value, layer);

class A3C_Network_Multiagent:
    """
    The class that creates the policy and value network for the A3C. 
    """
    
    def __init__(self, graph, scope_name, state_dim, action_size, 
                net_length_global, net_length_local, agt_num):
        """
        Constructor.
        
        Args:
            graph: tf.Graph
                The computation graph.
            scope_name: String
                The name of the scope.
            state_dim, action_size: int
                The number of the state dimension and number of action choices. 
            net_length_global, net_length_local: int
                The number of layers of the globally and the locally shared network.
            agt_num: int
                The number of agent. 
        """
        with graph.as_default(), tf.name_scope(scope_name):
            # Generate placeholder for state
            self._state_placeholder = tf.placeholder(tf.float32,
                                                     shape=(None, 
                                                            state_dim),
                                                     name='state_pl');
            self._keep_prob = tf.placeholder(tf.float32, name='keep_prob');
            # Build the operations that computes predictions from the nn model.
            self._policy_pred_list, self._v_pred_list, self._shared_layer= \
                self._create_model(self._state_placeholder, self._keep_prob, 
                                   action_size, net_length_global, net_length_local, 
                                   agt_num);
            
    @property
    def state_placeholder(self):
        return self._state_placeholder;

    @property
    def keep_prob(self):
        return self._keep_prob;
    
    @property
    def policy_pred_list(self):
        return self._policy_pred_list;
    
    @property
    def value_pred_list(self):
        return self._v_pred_list;

    @property
    def shared_layer(self):
        return self._shared_layer;

    def _create_model(self, input_state, keep_prob, num_actions, net_length_global, 
                    net_length_local, agt_num): 
        """
        Create the model of the policy network and value network for each room.
        All networks share the same global network for feature extraction; then, agt_num
        pnv shared networks are created in parallel from the shared global network.
        Like the ordinary A3C, the policy network and the value network of each room share 
        the a pnv shared network and then the policy network and value network of each 
        room are created. The policy network uses a softmax layer to provide the probablity 
        of taking each action, and the value network uses a linear layer to provide a scalar 
        for the value. 
        
        Args:
            input_state: tf tensor or placeholder.
                Represent the input to the network, which is the state observation.
            keep_prob: tf tensor or placeholder.
                The 1 - dropout probability.
            num_actions: int.
                Number of actions.
            net_length_global, net_length_local: int
                The number of layers of the globally and the locally shared network.
            agt_num: int
                The number of agent. 
        
        Return: (list, list, tf tensor)
            The policy network list and the value network list, and the globally shared
            network. 
            
        """
        policy_pred_list = [];
        value_pred_list = [];
        # The global shared layers
        with tf.name_scope('global_shared_layers'):
            # Dropout layer for the first relu layer.
            global_shared_layer = tf.nn.dropout(input_state, keep_prob);
            for _ in range(net_length_global):
                global_shared_layer = Dense(NN_WIDTH, activation = 'relu')(global_shared_layer);
        # The layers for each agent (room)
        for agent_i in range(agt_num):
            with tf.name_scope('agent_%d'%(agent_i)):
                local_shared_layer_i = Dense(NN_WIDTH, activation = 'relu')(global_shared_layer);
                for layer_i in range(net_length_local - 1):
                    local_shared_layer_i = Dense(NN_WIDTH, activation = 'relu')(local_shared_layer_i);
                # Create the policy and value network for each room
                with tf.name_scope('policy_network_%d'%(agent_i)):
                    policy_i = Dense(num_actions, activation = 'softmax')(local_shared_layer_i);
                with tf.name_scope('value_network_%d'%(agent_i)):
                    value_i = Dense(1)(local_shared_layer_i);
                policy_pred_list.append(policy_i);
                value_pred_list.append(value_i);
        return (policy_pred_list, value_pred_list, global_shared_layer);
