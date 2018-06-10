'''
使用cnn分类cifar10
'''
import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time

num_examples = 10000
max_steps = 3000
batch_size = 1
data_dir = '../data/cifar10/cifar-10-batches-bin'

tf.logging.set_verbosity(tf.logging.INFO)

#下载并解压数据
cifar10.maybe_download_and_extract()

# 定义一个有正则化的权重变量类型
def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if wl is not None:
        # 对权重进行L2正则，产生一个loss，加到最后总体loss上
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(var),wl,name='weight_loss'))
    return var

# 数据
images_train, labels_train = cifar10_input.distorted_inputs(data_dir = data_dir, batch_size = batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir = data_dir, batch_size = batch_size)

# placeholder
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

# 第一层
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev = 5e-2, wl = 0.0)
kernel1 = tf.nn.conv2d(image_holder,weight1,[1, 1, 1, 1], padding = 'SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1,ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

# 第二层
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev = 5e-2,wl = 0.0)
kernel2 = tf.nn.conv2d(norm1,weight2,[1, 1, 1, 1],padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0,alpha = 0.001/9.0, beta = 0.75)
pool2 = tf.nn.max_pool(norm2,ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME')

weight3 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev = 5e-2,wl = 0.0)
kernel3 = tf.nn.conv2d(pool2,weight3,[1, 1, 1, 1],padding='SAME')
bias3 = tf.Variable(tf.constant(0.1, shape=[64]))
conv3 = tf.nn.relu(tf.nn.bias_add(kernel3, bias3))
norm3 = tf.nn.lrn(conv3, 4, bias=1.0,alpha = 0.001/9.0, beta = 0.75)
pool3 = tf.nn.max_pool(norm2,ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME')

# 全连接
reshape = tf.reshape(pool3, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 全连接层
weight4 = variable_with_weight_loss(shape=[384, 192], stddev = 0.04, wl = 0.04)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# 分类层
weight5 = variable_with_weight_loss(shape=[192, 10], stddev = 0.04, wl = 0.04)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels)
    tf.add_to_collection('losses', tf.reduce_mean(cross_entropy))
    return tf.add_n(tf.get_collection('losses'))

loss = loss(logits, label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

    sess.run(tf.global_variables_initializer())
    # 启动图中的所有线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 训练
    for step in range(max_steps):
        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, loss], feed_dict = {image_holder: image_batch, label_holder: label_batch})
        if step % 10 == 0:
            print('{} step {}, loss {:.2f}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                                   step, loss_value))

    num_iter = num_examples // batch_size
    true_count = 0
    for step in range(num_iter):
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
        true_count += np.sum(predictions)

    precision = true_count / (num_iter * batch_size)
    print('precision @ 1 = %.3f' % precision)

    coord.request_stop()
    coord.join(threads)