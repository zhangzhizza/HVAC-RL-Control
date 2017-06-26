"""Loss functions."""

import tensorflow as tf

def a3c_loss(R, v_pred, pi, pi_one_hot, vloss_frac, ploss_frac, hregu_frac):
    """
    The loss function for the A3C.
        loss = vloss_frac * value_loss - (ploss_frac * policy_loss 
                + hregu_frac * entropy) # We want minimize the value_loss
                                        # but maximize the "policy loss"
        value_loss = 0.5 * sum_i((R_i - v_pred_i)**2)
        policy_loss = sum_i(log(pi(s_i, a_i) * (R_i - v_pred_i)))
        entropy = -sum_j((sum_i(pi(s_j, a_i) * log(s_j, a_i))))
    Reference: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
    
    Args:
        R: tf tensor, m*1 where m is sample size.
            The target Q value for each state in the sample.
        v_pred: tf tensor, m*1 where m is sample size.
            The predicted V value for each state in the sample.
        pi: tf tensor, m*a where m and a are the sample size and action size
            The probability of taking actions at a state. 
        pi_one_hot: tf tensor, m*1 where m is sample size.
            The probability of taking the action a_i at the state i.
        vloss_frac, ploss_frac, hregu_frac: float.
            The hyperparameter for the weights of each loss. 
    
    Return:
        tf tensor.
            The a3c loss. 
    
    """
    pi = pi + 1e-10; # To avoid log zero
    pi_one_hot = pi_one_hot + 1e-10;
    with tf.name_scope('a3c_loss'):
        with tf.name_scope('value_mse_loss'):
            v_mse_loss = 0.5 * tf.reduce_sum(tf.square(R - v_pred));
        with tf.name_scope('policy_loss'):
            policy_loss = tf.reduce_sum(tf.log(pi_one_hot) * tf.stop_gradient(R - v_pred));
        with tf.name_scope('entropy'):
            entropy = -tf.reduce_sum(pi * tf.log(pi));
        with tf.name_scope('weighted_loss'):
            loss = vloss_frac * v_mse_loss - (ploss_frac * policy_loss + 
                                              hregu_frac * entropy); 
        
    return loss;
