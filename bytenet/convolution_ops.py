from __future__ import division

import tensorflow as tf
import numpy as np
'''library with some functions adapted from https://github.com/tomlepaine/fast-wavenet/blob/master/wavenet/layers.py
in attempt to make batch size larger than one for dilated convolution'''

'''investigate this function to ensure that higher batch sizes are appropriate'''
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