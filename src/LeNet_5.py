'''
LeNet5 特性：
    1. 每个卷积层包含三个部分： 卷积，池化，和非线性激活函数
    2. 使用卷积提取空间特征
    3. 降采样的平均池化层
    4. sigmoid激活函数
    5. MLP作为最后的分类器
    6. 层与层之间的稀疏连接减少计算的复杂度。
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets('../data/mnist', one_hot=True)

# weight
def get_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    pass

def get_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

# conv
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层
w_conv1 = get_weight([5, 5, 1, 32])
b_conv1 = get_bias([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层
w_conv2 = get_weight([5, 5, 32, 64])
b_conv2 = get_bias([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接
w_fc1 = get_weight([7 * 7 * 64, 1024])
b_fc1 = get_bias([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool2_flat, w_fc1), b_fc1))

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax
w_fc2 = get_weight([1024, 10])
b_fc2 = get_bias([10])
y = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# cross_entroy
cross_entry = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entry)

# 准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

epochs = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        x_s, y_s = mnist.train.next_batch(50)

        if epoch % 100 == 0:
            accuracy = sess.run(accuracy_op, feed_dict={x: x_s, y_: y_s, keep_prob: 1})
            print("step {} accuracy: {}".format(epoch, accuracy))

        sess.run(train_step, feed_dict={x: x_s, y_: y_s, keep_prob: 0.5})

    sess.run(accuracy_op, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})