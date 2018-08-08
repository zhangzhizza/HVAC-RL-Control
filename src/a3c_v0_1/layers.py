import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import multiply, add
from keras import activations, initializers, regularizers, constraints
from keras.engine.topology import Layer, InputSpec
from copy import deepcopy

from a3c_v0_1.objectives import a3c_loss

class NoisyDense(Layer):
    """
    NoisyNet layer. Developed based on Keras Dense implementation.
    ```
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the noise weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        nD tensor with shape: `(batch_size, ..., self._input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, self._input_dim)`.
    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, self._input_dim)`,
        the output would have shape `(batch_size, units)`.
    """
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'self._input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('self._input_dim'),)
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self._input_dim = input_shape[-1]

        self._weight_noise = self.add_weight(shape=(self._input_dim, self.units),
                                      initializer=initializers.Constant(0.0),
                                      name='weigh_noise',
                                      regularizer=self.kernel_regularizer,
                                      trainable = False,
                                      constraint=self.kernel_constraint)
        self._mu_weight = self.add_weight(shape=(self._input_dim, self.units),
                                      initializer=initializers.RandomUniform(minval = -np.sqrt(3.0/self._input_dim), 
                                                                                   maxval = np.sqrt(3.0/self._input_dim), 
                                                                                   seed=None),
                                      name='mu_weight',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self._sigma_weight = self.add_weight(shape=(self._input_dim, self.units),
                                      initializer=initializers.Constant(0.017),
                                      name='sigma_weight',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self._bias_noise = self.add_weight(shape=(self.units,),
                                      initializer=initializers.Constant(0.0),
                                      name='bias_noise',
                                      regularizer=self.bias_regularizer,
                                      trainable = False,
                                      constraint=self.bias_constraint)
            self._mu_bias = self.add_weight(shape=(self.units,),
                                        initializer=initializers.RandomUniform(minval = -np.sqrt(3.0/self._input_dim), 
                                                                                     maxval = np.sqrt(3.0/self._input_dim), 
                                                                                     seed=None),
                                        name='mu_bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self._sigma_bias = self.add_weight(shape=(self.units,),
                                        initializer=initializers.Constant(0.017),
                                        name='sigma_bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: self._input_dim})
        self.built = True
        self.sample_noise(K.get_session()); # Just add the assignment ops to the graph, not really do sampling

    def call(self, inputs):
        noisy_weight = add([multiply([self._sigma_weight, self._weight_noise]), 
                                      self._mu_weight]);
        output = K.dot(inputs, noisy_weight)
        if self.use_bias:
            noisy_bias = add([multiply([self._sigma_bias, self._bias_noise]),
                                        self._mu_bias]);
            output = K.bias_add(output, noisy_bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(NoisyDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def sample_noise(self, session):
        with session.as_default():
            weight_noise = np.random.normal(0, 1, size=(self._input_dim, self.units))
            K.set_value(self._weight_noise, weight_noise)
            if self.use_bias:
                bias_noise = np.random.normal(0, 1, size=(self.units,))
                K.set_value(self._bias_noise, bias_noise);

    def remove_noise(self, session):
        with session.as_default():
            K.set_value(self._weight_noise, 
                        np.zeros(shape=(self._input_dim, self.units)))
            if self.use_bias:
                K.set_value(self._bias_noise, 
                            np.zeros(shape=(self.units,)))

    def debug(self):
        return [self._weight_noise, self._bias_noise]
     