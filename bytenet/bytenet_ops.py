from __future__ import division

import tensorflow as tf


def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)


def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)


def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum,
                                     epsilon=1e-5)


optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}


def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        shape_list = value.get_shape().as_list()
        pad_elements = dilation - 1 - (shape_list[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape_list[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape_list[2]])


def batch_to_time(value, dilation, name=None):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        shape_list = value.get_shape().as_list()
        prepared = tf.reshape(value, [dilation, -1, shape_list[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape_list[2]]) #unknown batch size


def causal_conv(value, filter_, dilation, name='causal_conv'):
    with tf.name_scope(name):
        # Pad beforehand to preserve causality.
        value_shape_list = value.get_shape().as_list()
        # filter_width = tf.shape(filter_)[0]
        filter_width = filter_.get_shape().as_list()[0]
        padding = [[0, 0], [(filter_width - 1) * dilation, 0], [0, 0]]
        padded = tf.pad(value, padding)
        if dilation > 1:
            transformed = time_to_batch(padded, dilation)
            conv = tf.nn.conv1d(transformed, filter_, stride=1, padding='SAME')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(padded, filter_, stride=1, padding='SAME')
        # Remove excess elements at the end.
        result = tf.slice(restored,
                          [0, 0, 0],
                          [-1, value_shape_list[1], -1])
        return result


def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = quantization_channels - 1
        # Perform mu-law companding transformation (ITU-T, 1988).
        magnitude = tf.log(1 + mu * tf.abs(audio)) / tf.log(1. + mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.cast((signal + 1) / 2 * mu + 0.5, tf.int32)


def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        casted = tf.cast(output, tf.float32)
        signal = 2 * (casted / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude


def create_simple_bytenet_dilation_layer(input_batch, layer_index, dilation, all_variables, use_batch_norm, train):
    '''Creates a single causal dilated convolution layer that mimics figure 3 left in bytenet paper

    note that in the source network, there is no masking on causal convolution. However, on the decoder, there is indeed causal convolutions 
    '''
    variables = all_variables['dilated_stack'][layer_index]

    if use_batch_norm:
        input_batch = variables['dense_batch_norm'](input_batch, train = train)
    activated = tf.nn.relu(input_batch)
    weights_dense = variables['dense']
    first_flat_conv = tf.nn.conv1d(activated, weights_dense, stride=1, padding="SAME", name="dense")


    if use_batch_norm:
        first_flat_conv = variables['filter_batch_norm'](first_flat_conv, train = train)
    activated_first_flat_conv = tf.nn.relu(first_flat_conv)
    #calls for masked 1 x k here for decoder -- TODO later
    weights_filter = variables['filter']
    causal_conv_filter = causal_conv(activated_first_flat_conv, weights_filter, dilation)


    if use_batch_norm:
        causal_conv_filter = variables['skip_batch_norm'](causal_conv_filter, train = train)
    activated_causal_conv_filter = tf.nn.relu(causal_conv_filter)
    weights_skip = variables['skip']
    final_block_output = tf.nn.conv1d(
        activated_causal_conv_filter, weights_skip, stride=1, padding="SAME", name="skip")

    return input_batch + final_block_output

def create_wavenet_dilation_layer(self, input_batch, layer_index, dilation, all_variables, use_biases):
    '''Creates a single causal dilated convolution layer.

    Nick you may have to modify this to produce the same blocks that bytenet had

    The layer contains a gated filter that connects to dense output
    and to a skip connection:

           |-> [gate]   -|        |-> 1x1 conv -> skip output
           |             |-> (*) -|
    input -|-> [filter] -|        |-> 1x1 conv -|
           |                                    |-> (+) -> dense output
           |------------------------------------|

    Where `[gate]` and `[filter]` are causal convolutions with a
    non-linear activation at the output.
    '''
    variables = all_variables['dilated_stack'][layer_index]

    weights_filter = variables['filter']
    weights_gate = variables['gate']

    conv_filter = causal_conv(input_batch, weights_filter, dilation)
    conv_gate = causal_conv(input_batch, weights_gate, dilation)

    if use_biases:
        filter_bias = variables['filter_bias']
        gate_bias = variables['gate_bias']
        conv_filter = tf.add(conv_filter, filter_bias)
        conv_gate = tf.add(conv_gate, gate_bias)

    out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

    # The 1x1 conv to produce the residual output
    weights_dense = variables['dense']
    transformed = tf.nn.conv1d(
        out, weights_dense, stride=1, padding="SAME", name="dense")

    # The 1x1 conv to produce the skip output
    weights_skip = variables['skip']
    skip_contribution = tf.nn.conv1d(
        out, weights_skip, stride=1, padding="SAME", name="skip")

    if use_biases:
        dense_bias = variables['dense_bias']
        skip_bias = variables['skip_bias']
        transformed = transformed + dense_bias
        skip_contribution = skip_contribution + skip_bias

    return skip_contribution, input_batch + transformed