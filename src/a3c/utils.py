"""Common functions you may find useful in your implementation."""

import tensorflow as tf
import semver

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


def init_variables(graph, scope):
    """Init the trainable variables in the scope.
    
    Parameters
    ----------
    graph: tf.Graph
      The graph object.
    scope: String
      The scope name.

    Returns
    -------
    tf.Tensor
      Tensor update ops.
    
    """
    trainable_ws = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES
                                               , scope=scope);

    return tf.variables_initializer(trainable_ws);

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


