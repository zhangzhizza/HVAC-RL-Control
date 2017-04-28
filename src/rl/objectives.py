"""Loss functions."""

import tensorflow as tf

def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    with tf.name_scope('huber_loss'):
        residual = y_true - y_pred;
        constant1 = tf.constant(0.5);#Type mismatch error? Because numpy array float
                                 #is float64 by default, and tf is float32 by
                                 #default
    
        op1 = tf.multiply(constant1, tf.square(residual));
        op2 = (tf.constant(max_grad) * tf.abs(residual) 
           - constant1 * tf.square(max_grad));
    
        condition = tf.less(tf.abs(residual), tf.constant(max_grad));
        loss = tf.where(condition, op1, op2);
        
    return loss


def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    with tf.name_scope("mean_huber_loss"):
        loss = tf.reduce_mean(huber_loss(y_true, y_pred, max_grad));
    
    return loss