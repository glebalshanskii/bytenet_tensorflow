from __future__ import division

import tensorflow as tf
import numpy as np
import convolution_ops

def create_simple_dilation_layer(input_batch, layer_index, dilation, all_variables, use_batch_norm, train, return_residual_output = False):
    '''For experimenting this simple dilation layer is used to avoid 1x1 convolutions as called for in the paper. 
    The point of the 1x1 convolutions is to reduce d by two. However, a cheaper implementation may be to avoid these 1x1 convolutions all together and just chain together simple dilation layers.
    '''
    variables = all_variables['dilated_stack'][layer_index]

    if use_batch_norm:
        input_batch = variables['filter_batch_norm'](input_batch, train = train)
    activated_input_batch = tf.nn.relu(input_batch)
    #calls for masked 1 x k here for decoder -- TODO later

    weights_filter = variables['filter']
    causal_conv_filter = convolution_ops.causal_conv(activated_input_batch, 
        weights_filter, 
        dilation, 
        name='dilated_filter_lyr{}_dilation{}'.format(layer_index, dilation))

    final_block_output = input_batch + causal_conv_filter

    if return_residual_output:
        return final_block_output, final_block_output
    else:
        return final_block_output


def create_simple_bytenet_dilation_layer(input_batch, layer_index, dilation, all_variables, use_batch_norm, train):
    '''Creates a single causal dilated convolution layer that mimics figure 3 left in bytenet paper

    note that in the source network, there is no masking on causal convolution. However, on the decoder, there is indeed causal convolutions 
    '''
    variables = all_variables['dilated_stack'][layer_index]

    '''in this section we increase out channels to 1d -> 2d'''
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
    causal_conv_filter = convolution_ops.causal_conv(activated_first_flat_conv, 
        weights_filter, 
        dilation, 
        name='dilated_filter_lyr{}_dilation{}'.format(layer_index, dilation))

    '''in this section we increase out channels to 1d -> 2d'''
    if use_batch_norm:
        causal_conv_filter = variables['skip_batch_norm'](causal_conv_filter, train = train)
    activated_causal_conv_filter = tf.nn.relu(causal_conv_filter)
    weights_skip = variables['skip'] 
    final_block_output = tf.nn.conv1d(
        activated_causal_conv_filter, weights_skip, stride=1, padding="SAME", name="skip")

    return input_batch + final_block_output

def create_wavenet_dilation_layer(input_batch, layer_index, dilation, all_variables, use_batch_norm, train, return_residual_output = True, use_biases = False, activation = tf.tanh):
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

    conv_filter = convolution_ops.causal_conv(input_batch, weights_filter, dilation)
    conv_gate = convolution_ops.causal_conv(input_batch, weights_gate, dilation)

    if use_biases:
        filter_bias = variables['filter_bias']
        gate_bias = variables['gate_bias']
        conv_filter = tf.add(conv_filter, filter_bias)
        conv_gate = tf.add(conv_gate, gate_bias)

    out = activation(conv_filter) * tf.sigmoid(conv_gate)

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