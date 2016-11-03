from __future__ import division

import tensorflow as tf
import numpy as np
'''library from https://github.com/tomlepaine/fast-wavenet/blob/master/wavenet/layers.py
in attempt to make batch size larger than one for dilated convolution'''



def time_to_batch(inputs, rate):
    '''If necessary zero-pads inputs and reshape by rate.
    
    Used to perform 1D dilated convolution.
    
    Args:
      inputs: (tensor) 
      rate: (int)
    Outputs:
      outputs: (tensor)
      pad_left: (int)
    '''
    _, width, num_channels = inputs.get_shape().as_list()

    width_pad = int(rate * np.ceil((width + rate) * 1.0 / rate))
    pad_left = width_pad - width

    perm = (1, 0, 2)
    shape = (int(width_pad / rate), -1, num_channels) # missing dim: batch_size * rate
    padded = tf.pad(inputs, [[0, 0], [pad_left, 0], [0, 0]])
    transposed = tf.transpose(padded, perm)
    reshaped = tf.reshape(transposed, shape)
    outputs = tf.transpose(reshaped, perm)
    return outputs

def batch_to_time(inputs, rate, crop_left=0):
    ''' Reshape to 1d signal, and remove excess zero-padding.
    
    Used to perform 1D dilated convolution.
    
    Args:
      inputs: (tensor)
      crop_left: (int)
      rate: (int)
    Ouputs:
      outputs: (tensor)
    '''
    shape = tf.shape(inputs)
    batch_size = shape[0] / rate
    width = shape[1]
    
    out_width = tf.to_int32(width * rate)
    _, _, num_channels = inputs.get_shape().as_list()
    
    perm = (1, 0, 2)
    new_shape = (out_width, -1, num_channels) # missing dim: batch_size
    transposed = tf.transpose(inputs, perm)    
    reshaped = tf.reshape(transposed, new_shape)
    outputs = tf.transpose(reshaped, perm)
    cropped = tf.slice(outputs, [0, crop_left, 0], [-1, -1, -1])
    return cropped

def conv1d(inputs,
           out_channels,
           filter_width=2,
           stride=1,
           padding='VALID',
           data_format='NHWC',
           gain=np.sqrt(2),
           activation=tf.nn.relu,
           bias=False):
    '''One dimension convolution helper function.
    
    Sets variables with good defaults.
    
    Args:
      inputs:
      out_channels:
      filter_width:
      stride:
      paddding:
      data_format:
      gain:
      activation:
      bias:
      
    Outputs:
      outputs:
    '''
    in_channels = inputs.get_shape().as_list()[-1]

    stddev = gain / np.sqrt(filter_width**2 * in_channels)
    w_init = tf.random_normal_initializer(stddev=stddev)

    w = tf.get_variable(name='w',
                        shape=(filter_width, in_channels, out_channels),
                        initializer=w_init)

    outputs = tf.nn.conv1d(
    	inputs, w, stride=stride,padding=padding, data_format=data_format)

    if bias:
        b_init = tf.constant_initializer(0.0)
        b = tf.get_variable(name='b',
                            shape=(out_channels, ),
                            initializer=b_init)

        outputs = outputs + tf.expand_dims(tf.expand_dims(b, 0), 0)

    return outputs

def dilated_conv1d(inputs,
                   weights,
                   rate=1,
                   name=None,
                   gain=np.sqrt(2)):
    '''
    
    Args:
      inputs: (tensor)
      output_channels:
      filter_width:
      rate:
      padding:
      name:
      gain: not using this right now due to xavier initialization
      activation:
    Outputs:
      outputs: (tensor)
    '''
    assert name
    with tf.variable_scope(name):
    	_, _, out_channels = weights.get_shape().as_list()
        _, width, _ = inputs.get_shape().as_list()
        inputs_ = time_to_batch(inputs, rate=rate)
        outputs_ = tf.nn.conv1d(
    		inputs, weights, stride=1,padding='VALID', data_format='NHWC')

        _, conv_out_width, _ = outputs_.get_shape().as_list()
        new_width = conv_out_width * rate
        diff = new_width - width
        outputs = batch_to_time(outputs_, rate=rate, crop_left=diff)

        # Add additional shape information.
        tensor_shape = [tf.Dimension(None),
                        tf.Dimension(width),
                        tf.Dimension(out_channels)]
        outputs.set_shape(tf.TensorShape(tensor_shape))

    return outputs