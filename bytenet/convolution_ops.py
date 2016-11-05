from __future__ import division

import tensorflow as tf
import numpy as np
'''library with some functions adapted from https://github.com/tomlepaine/fast-wavenet/blob/master/wavenet/layers.py
in attempt to make batch size larger than one for dilated convolution'''


'''currently investigating to see if the causal convolution can look at BOTH sides evenly -- not just to the left'''

def time_to_batch(value, dilation, pad_both_sides_evenly = False, name=None):
    '''pad_both_sides_evenly: Will pad both sides - used for source network

    This is the first function used in causal convolution'''
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        shape_list = value.get_shape().as_list()
        pad_elements = dilation - 1 - (shape_list[1] + dilation - 1) % dilation #always an even number
        if pad_both_sides_evenly:
            pad_elements_both_sides = int(pad_elements/2)
            padded = tf.pad(value, [[0, 0], [pad_elements_both_sides, pad_elements_both_sides], [0, 0]])
        else:
            padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])

        reshaped = tf.reshape(padded, [-1, dilation, shape_list[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape_list[2]])


def batch_to_time(value, dilation, pad_both_sides_evenly = False, name=None):
    '''pad_both_sides_evenly: Will pad both sides - used for source network

    Not sure if pad_both_sides_evenly needs to be passed in here'''

    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        shape_list = value.get_shape().as_list()
        prepared = tf.reshape(value, [dilation, -1, shape_list[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape_list[2]]) #unknown batch size


def causal_conv(value, filter_, dilation, pad_both_sides_evenly = False, name='causal_conv'):
    with tf.name_scope(name):
        # Pad beforehand to preserve causality.
        value_shape_list = value.get_shape().as_list()
        # filter_width = tf.shape(filter_)[0]
        filter_width = filter_.get_shape().as_list()[0]
        padding = [[0, 0], [(filter_width - 1) * dilation, 0], [0, 0]] #this may need to be adjusted for padding on both sides 
        padded = tf.pad(value, padding)
        if dilation > 1:
            transformed = time_to_batch(padded, dilation, pad_both_sides_evenly = pad_both_sides_evenly)
            conv = tf.nn.conv1d(transformed, filter_, stride=1, padding='SAME')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(padded, filter_, stride=1, padding='SAME')

        # Remove excess elements at the end.
        result = tf.slice(restored,
                          [0, 0, 0],
                          [-1, value_shape_list[1], -1]) #for padding on both sides this may need to be adjusted
        return result