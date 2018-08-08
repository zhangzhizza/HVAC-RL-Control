import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import (Activation, Dense, Flatten, Input,
                          Permute)
from keras.models import Model
from copy import deepcopy

from a3c_v0_1.objectives import a3c_loss
from a3c_v0_1.layers import NoisyDense

class A3C_Network_NN:
    """
    The class that creates the policy and value network for the A3C. 
    """
    
    def __init__(self, graph, scope_name, state_dim, action_size, 
                 activation = 'relu', model_param = [512, 4], noisy_layer = False,
                 kernel_initializer = 'glorot_uniform'):
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
        self._activation = activation;
        self._model_param = model_param;
        self._graph = graph;
        self._noisy_layer = noisy_layer;
        self._kernel_initializer = kernel_initializer;
        with graph.as_default(), tf.name_scope(scope_name):
            # Generate placeholder for state
            self._state_placeholder = tf.placeholder(tf.float32,
                                                     shape=(None, 
                                                            state_dim),
                                                     name='state_pl');
            self._keep_prob = tf.placeholder(tf.float32, name='keep_prob');
            # Build the operations that computes predictions from the nn model.
            self._policy_pred, self._v_pred, self._shared_layer= \
                self._create_model(self._state_placeholder, self._keep_prob, action_size, self._noisy_layer);
            
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
    def graph(self):
        return self._graph;

    @property
    def shared_layer(self):
        return self._shared_layer;

    @property
    def policy_network_finalLayer(self):
        return self._policy_network_finalLayer

    @property
    def value_network_finalLayer(self):
        return self._value_network_finalLayer
    
    
    
    def _create_model(self, input_state, keep_prob, num_actions, noisy_layer): 
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
        # Build shared layers
        with tf.name_scope('shared_layers'):
            # Dropout layer for the first relu layer.
            layer = tf.nn.dropout(input_state, keep_prob);
            for layer_i in range(self._model_param[1]):
                layer = Dense(self._model_param[0], activation = self._activation, 
                              kernel_initializer = self._kernel_initializer)(layer);
        # Choose from Dense or NoisyDense
        if noisy_layer:
            finalLayer = NoisyDense;
        else:
            finalLayer = Dense;
        # Build policy and value network
        with tf.name_scope('policy_network'):
            self._policy_network_finalLayer = finalLayer(num_actions, 
                                                         kernel_initializer = self._kernel_initializer,   
                                                         activation = 'softmax')
            policy = self._policy_network_finalLayer(layer);
        with tf.name_scope('value_network'):
            self._value_network_finalLayer = finalLayer(1, 
                                                        kernel_initializer = self._kernel_initializer);
            value = self._value_network_finalLayer(layer);

        return (policy, value, layer);


    """
    with tf.name_scope('policy_network_noisy'):
        self._policy_weight_noise = tf.Variable(tf.random_normal(shape=[self._model_param[0], num_actions]), 
                                            name='policy_weight_noise', trainable = False);
        self._policy_bias_noise = tf.Variable(tf.random_normal(shape=[num_actions]), 
                                          name='policy_bias_noise', trainable = False);
        noisyNet_policy = self._noisyNet(layer, num_actions, self._policy_weight_noise, self._policy_bias_noise);
        policy =  tf.nn.softmax(noisyNet_policy);
    with tf.name_scope('value_network_noisy'):
        self._value_weight_noise = tf.Variable(tf.random_normal(shape=[self._model_param[0], 1]), 
                                           name='value_weight_noise', trainable = False);
        self._value_bias_noise = tf.Variable(tf.random_normal(shape=[1]), 
                                         name='value_bias_noise', trainable = False);
        value = self._noisyNet(layer, 1, self._value_weight_noise, self._value_bias_noise);
return (policy, value, layer);

def _noisyNet(self, input, outShape, weight_noise, bias_noise):
""
Create NoisyNet.

Args:
    outShape: 1-D python list
""
inShape = tf.shape(input);
weight_shape = tf.concat([inShape[1:], [outShape]], axis = 0)
p = tf.cast(weight_shape[0], dtype = tf.float32) # Number of inputs to the linear layer

mu_weight = tf.Variable(tf.random_uniform(shape = weight_shape, minval = -tf.sqrt(3.0/p), 
                                          maxval = tf.sqrt(3.0/p), name = 'mu_weight_init_rdmUni'),
                                           trainable = True, name = 'mu_weight');
sigma_weight = tf.Variable(tf.multiply(0.017, tf.ones(shape = weight_shape), name = 'sigma_weight_init_0.017'),
                                       trainable = True, name = 'sigma_weight');
with tf.name_scope('noisy_weight'):
    noisy_weight = tf.multiply(sigma_weight, weight_noise) + mu_weight;

mu_bias = tf.Variable(tf.random_uniform(shape = [outShape], minval = -tf.sqrt(3.0/p), 
                                        maxval = tf.sqrt(3.0/p), name = 'mu_bias_init_rdmUni'), 
                                         trainable = True, name = 'mu_bias');
sigma_bias = tf.Variable(tf.multiply(0.017, tf.ones(shape = [outShape]), name = 'sigma_bias_init_0.017'), 
                         trainable = True, name = 'sigma_bias');
with tf.name_scope('noisy_bias'):
    noisy_bias = tf.multiply(sigma_bias, bias_noise) + mu_bias;

with tf.name_scope('noisy_linear'):
    noisy_y = tf.matmul(input, noisy_weight) + noisy_bias;

return noisy_y;
    """
