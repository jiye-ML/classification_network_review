"""
随机深度残差网络的 随机残差block
"""


import tensorflow as tf


def conv(prev, kernel_shape, bias_shape, stride, padding, scope, use_relu=True, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        weights = tf.get_variable('weights', kernel_shape)
        bias = tf.get_variable('biases', bias_shape)
        conv2d = tf.nn.conv2d(prev, weights, stride, padding)
        biased = tf.nn.bias_add(conv2d, bias)
        if use_relu:
            return tf.nn.relu(biased)
        else:
            return biased


def create_conv_weights(kernel_shape, bias_shape, scope):
    with tf.variable_scope(scope):
        tf.get_variable('weights', kernel_shape)
        tf.get_variable('biases', bias_shape)


def maxpool(prev, kernel_shape, stride, padding, scope):
    with tf.name_scope(scope):
        return tf.nn.max_pool(prev, kernel_shape, stride, padding)


def avgpool(prev, kernel_shape, stride, padding, scope):
    with tf.name_scope(scope):
        return tf.nn.avg_pool(prev, kernel_shape, stride, padding)

# 一个随机残差模块
def residual_block(prev, bottleneck_depth, output_depth, survival_rate, training, scope):
    with tf.variable_scope(scope):
        # 一个block的恒等映射部分
        res = prev
        prev_depth = prev.get_shape()[3].value
        # 留下来的比例
        survival_rate = tf.constant(survival_rate, name='survival_rate')
        # 如果深度不一样，调节
        if prev_depth != output_depth:
            res = conv(prev=prev,
                       kernel_shape=[1, 1, prev_depth, output_depth],
                       bias_shape=[output_depth],
                       stride=[1, 1, 1, 1],
                       padding='SAME',
                       scope='shortcut')
        # 创建一个block
        create_conv_weights(kernel_shape=[1, 1, prev_depth, bottleneck_depth],
                            bias_shape=[bottleneck_depth],
                            scope='bottleneck_1x1')
        create_conv_weights(kernel_shape=[3, 3, bottleneck_depth, bottleneck_depth],
                            bias_shape=[bottleneck_depth],
                            scope='bottleneck_3x3')
        create_conv_weights(kernel_shape=[1, 1, bottleneck_depth, output_depth],
                            bias_shape=[output_depth],
                            scope='output_1x1')
        # 一个block的function部分
        def perturbation(reuse=True):
            conv2d = conv(prev=prev,
                          kernel_shape=[1, 1, prev_depth, bottleneck_depth],
                          bias_shape=[bottleneck_depth],
                          padding='SAME',
                          stride=[1, 1, 1, 1],
                          scope='bottleneck_1x1',
                          reuse=reuse)
            conv2d = conv(prev=conv2d,
                          kernel_shape=[3, 3, bottleneck_depth, bottleneck_depth],
                          bias_shape=[bottleneck_depth],
                          stride=[1, 1, 1, 1],
                          padding='SAME',
                          scope='bottleneck_3x3',
                          reuse=reuse)
            conv2d = conv(prev=conv2d,
                          kernel_shape=[1, 1, bottleneck_depth, output_depth],
                          bias_shape=[output_depth],
                          stride=[1, 1, 1, 1],
                          padding='SAME',
                          scope='output_1x1',
                          use_relu=False,
                          reuse=reuse)
            return conv2d

        # 对于没有drop的层
        def not_dropped():
            return tf.nn.relu(tf.add(res, perturbation()))

        # 对于drop的层
        def dropped():
            return tf.nn.relu(res)

        def test():
            with tf.name_scope('test'):
                return tf.nn.relu(tf.add(res, tf.multiply(perturbation(), survival_rate)))

        def train():
            with tf.name_scope('train'):
                # drop的概率
                survival_roll = tf.random_uniform(shape=[], minval=0.0, maxval=1.0, name='survival')
                survive = tf.less(survival_roll, survival_rate)
                return tf.cond(survive, not_dropped, dropped)

        return tf.cond(training, train, test)


if __name__ == '__main__':

    inputs = tf.placeholder(tf.float32, [None, 55, 55, 64])
    training = tf.placeholder(tf.bool)
    output = residual_block(prev=inputs,
                            bottleneck_depth=32,
                            output_depth=128,
                            survival_rate=0.8,
                            training=training,
                            scope='stack')
    # writer = tf.train.SummaryWriter('stochastic_depth',
    #                                 graph=tf.get_default_graph())
    # writer.close()