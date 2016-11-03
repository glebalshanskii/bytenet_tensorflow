from __future__ import division

import tensorflow as tf

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))
                
                try:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                except:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1], name='moments')
                    
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

def reduce_total_output_nodes(input_tensor, reduction_rate=10):
	'''this function is useful for reducing the total number of output nodes of a source network

	if you're source network is size:

	[batch_size, num_nodes, quanization_channels]

	You're new output by summing adjacent nodes will be:

	[batch_size, num_nodes/reduction_rate, quanization_channels]

	This is useful for attention purposes for your target network if you use a rnn network. You can take 2000 timesteps to 200 outputs with a factor of 10

	@leavesbreathe investigated using tf.segment_sum. However, this requires reshaping the tensor so in the end, tf split may be more efficient'''


	input_tensor_shape = input_tensor.get_shape().as_list()

	if len(input_tensor_shape) != 3:
		raise ValueError('tensor not shaped correctly')

	if input_tensor_shape[1]%reduction_rate != 0:
		raise ValueError('num of nodes', input_tensor_shape[1], 'must be divisible by reduction rate:', reduction_rate)


	num_final_nodes = int(input_tensor_shape[1]/reduction_rate)
	input_tensor_split_list = tf.split(split_dim = 1, num_split = num_final_nodes, value =  input_tensor)

	for i,split_tensor in enumerate(input_tensor_split_list):
		reduced_sum = tf.reduce_sum(split_tensor, reduction_indices = 1, keep_dims = True)

		if i == 0:
			output_tensor = reduced_sum
		else:
			output_tensor = tf.concat(1, [output_tensor, reduced_sum])
	return output_tensor
