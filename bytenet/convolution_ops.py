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
    # print('inputs', inputs)
    # print('rate', rate)
    # print('crop left', crop_left) #i think this is the problem with larger filter size
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


def dilated_conv1d(inputs,
                   weights,
                   rate=1,
                   name=None):
    '''
	In this op, the dimensions of outputs should be exactly that of inputs

    Args:
      inputs: (tensor)
      output_channels:
      rate:
      name:
    Outputs:
      outputs: (tensor)
    '''
    assert name
    with tf.variable_scope(name):
    	filter_width, _, out_channels = weights.get_shape().as_list()
        _, width, _ = inputs.get_shape().as_list()
        inputs_ = time_to_batch(inputs, rate=rate) #this is not padding correctly
        outputs_ = tf.nn.conv1d(
    		inputs, weights, stride=1, padding='VALID', data_format='NHWC')

        _, conv_out_width, _ = outputs_.get_shape().as_list()

        new_width = conv_out_width * rate
        diff = new_width - width + filter_width #added filter_width for larger filters
        outputs = batch_to_time(outputs_, rate=rate, crop_left=diff)

        '''Add additional shape information.'''
        tensor_shape = [tf.Dimension(None),
                        tf.Dimension(width), #the crop left relies on this being the same width size
                        tf.Dimension(out_channels)]
        outputs.set_shape(tf.TensorShape(tensor_shape))

    return outputs