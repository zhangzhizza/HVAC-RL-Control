import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Dense, Flatten, Input,
                          Permute)
from keras.models import Model

from a3c.objectives import a3c_loss

NN_WIDTH = 512;

class A3C_Network:
    """
    The class that creates the policy and value network for the A3C. 
    """
    
    def __init__(self, graph, scope_name, state_dim, action_size):
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
            # Build the operations that computes predictions from the nn model.
            self._policy_pred, self._v_pred = \
                self._create_model(self._state_placeholder, action_size);
            
    @property
    def state_placeholder(self):
        return self._state_placeholder;
    
    @property
    def policy_pred(self):
        return self._policy_pred;
    
    @property
    def value_pred(self):
        return self._v_pred;
    
    def _create_model(self, input_state, num_actions): 
        """
        Create the model for the policy network and value network.
        The policy network and the value network share the model for feature
        extraction from the raw state, and then the policy network uses a 
        softmax layer to provide the probablity of taking each action, and the 
        value network uses a linear layer to provide a scalar for the value. 
        
        Args:
            input_state: tf tensor or placeholder.
                Represent the input to the network, which is the state observation.
            num_actions: int.
                Number of actions.
        
        Return: (tf tensor, tf tensor)
            The policy and the value for the state. 
            
        """
        with tf.name_scope('shared_layers'):
            layer = Dense(NN_WIDTH, activation = 'relu')(input_state);
            layer = Dense(NN_WIDTH, activation = 'relu')(layer);
            layer = Dense(NN_WIDTH, activation = 'relu')(layer);
            layer = Dense(NN_WIDTH, activation = 'relu')(layer);
        with tf.name_scope('policy_network'):
            policy = Dense(num_actions, activation = 'softmax')(layer);
        with tf.name_scope('value_network'):
            value = Dense(1)(layer);
        return (policy, value);