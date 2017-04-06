"""Common functions you may find useful in your implementation."""

import semver
import tensorflow as tf


def get_uninitialized_variables(sess, variables=None):
    """Return a list of uninitialized tf variables.

    Parameters
    ---------- 
    variables: tf.Variable, list(tf.Variable), optional
      Filter variable list to only those that are uninitialized. If no
      variables are specified the list of all variables in the graph
      will be used.

    Returns
    -------
    list(tf.Variable)
      List of uninitialized tf variables.
    """
    #sess = tf.get_default_session()
    if variables is None:
        variables = tf.global_variables()
    else:
        variables = list(variables)

    if len(variables) == 0:
        return []

    if semver.match(tf.__version__, '<1.0.0'):
        init_flag = sess.run(
            tf.pack([tf.is_variable_initialized(v) for v in variables]))
    else:
        init_flag = sess.run(
            tf.stack([tf.is_variable_initialized(v) for v in variables]))
    return [v for v, f in zip(variables, init_flag) if not f]


def get_soft_target_model_updates(graph, target_scope, source_scope, tau):
    r"""Return list of target model update ops.

    These are soft target updates. Meaning that the target values are
    slowly adjusted, rather than directly copied over from the source
    model.

    The update is of the form:

    $W' \gets (1- \tau) W' + \tau W$ where $W'$ is the target weight
    and $W$ is the source weight.

    Parameters
    ----------
    graph: tf.Graph
      The graph object.
    target_scope: String
      The target scope name.
    source_scope: String
      The source scope name.
    tau: float
      The weight of the source weights to the target weights used
      during update.

    Returns
    -------
    list(tf.Tensor)
      List of tensor update ops.
    """
    
    
    ops = [];
    source_trainable_ws = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES
                                               , scope=source_scope);
    target_trainable_ws = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES
                                               , scope=target_scope);
    
    for i in range(len(source_trainable_ws)):
        source_w = source_trainable_ws[i];
        target_w = target_trainable_ws[i];
        target_w_new = (1 - tau)*target_w + tau * source_w;
        op = tf.assign(target_w, target_w_new);
        ops.append(op);
    
    return ops;


def get_hard_target_model_updates(graph, target_scope, source_scope):
    """Return list of target model update ops.

    These are hard target updates. The source weights are copied
    directly to the target network.

    Parameters
    ----------
    graph: tf.Graph
      The graph object.
    target_scope: String
      The target scope name.
    source_scope: String
      The source scope name.
    tau: float
      The weight of the source weights to the target weights used
      during update.

    Returns
    -------
    list(tf.Tensor)
      List of tensor update ops.
    """
    ops = [];
    source_trainable_ws = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES
                                               , scope=source_scope);
    target_trainable_ws = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES
                                               , scope=target_scope);
    
    for source_w, target_w in zip(source_trainable_ws, target_trainable_ws):
        op = tf.assign(target_w, source_w);
        ops.append(op);
    
    return ops;

def init_weights_uniform(graph, scope):
    """Init the trainable variables in the scope with uniform random
    number.
    
    Parameters
    ----------
    graph: tf.Graph
      The graph object.
    scope: String
      The scope name.


    Returns
    -------
    list(tf.Tensor)
      List of tensor update ops.
    
    """
    #ops = [];
    trainable_ws = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES
                                               , scope=scope);
    

    return tf.variables_initializer(trainable_ws);
    #return ops;


