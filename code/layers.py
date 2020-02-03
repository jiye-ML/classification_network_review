import tensorflow as tf
import tensorflow.contrib.layers as tcl



def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1, relu=True, is_batch_norm = True):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels/groups, num_filters],
                                  initializer=tcl.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
    if relu:
        bias = tf.nn.relu(bias, name=scope.name)

    # if is_batch_norm:
    #     bias = batch_norm(bias)

    return bias

# 全连接层
def fc(x, num_out, name, relu=True):
    num_in = x.get_shape()[-1].value
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out], initializer=tcl.xavier_initializer(),trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    return tf.nn.relu(act) if relu else act

# 池化层
def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)

# Create a local response normalization layer.
def lrn(x, name,radius=5, alpha=1e-4, beta=0.75, bias=2.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

# 实现Batch Normalization
def batch_norm(x, epsilon=1e-5, momentum=0.9,train=True, name="batch_norm"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon,
                                        scale=True, is_training=train,scope=name)
