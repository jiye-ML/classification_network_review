'''
将 cifar10 存入 tfrecord
'''
from six.moves import cPickle
import os
import sys
import tarfile

import numpy as np
import urllib
import tensorflow as tf

from datasets import dataset_utils

# The URL where the CIFAR data can be downloaded.
_DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

# The number of training files.
_NUM_TRAIN_FILES = 5

# The height and width of each image.
_IMAGE_SIZE = 32

# The names of the classes.
_CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

# 加载cifar10 pickle files， 写入 TFRecord.
def _add_to_tfrecord(filename, tfrecord_writer, offset=0):

    with tf.gfile.Open(filename, 'rb') as f:
        if sys.version_info < (3,):
            data = cPickle.load(f)
        else:
            data = cPickle.load(f, encoding='bytes')
    # 读取数据
    images = data[b'data']
    num_images = images.shape[0]

    images = images.reshape((num_images, 3, 32, 32))
    labels = data[b'labels']

    with tf.Graph().as_default():
        image_placeholder = tf.placeholder(dtype=tf.uint8)
        # 解码
        encoded_image = tf.image.encode_png(image_placeholder)

        with tf.Session('') as sess:
            for j in range(num_images):
                print("Reading file [{}] image {}/{}".format(filename, offset + j + 1, offset + num_images))
                # 转换第三通道
                image = np.squeeze(images[j]).transpose((1, 2, 0))
                label = labels[j]
                # 读取数据
                png_string = sess.run(encoded_image, feed_dict={image_placeholder: image})
                # 存入
                example = dataset_utils.image_to_tfexample(png_string, b'png', _IMAGE_SIZE, _IMAGE_SIZE, label)
                tfrecord_writer.write(example.SerializeToString())

    return offset + num_images

# 输出的文件名
def _get_output_filename(dataset_dir, split_name):
    return '%s/cifar10_%s.tfrecord' % (dataset_dir, split_name)

# 如果文件压缩包不再目录中下载并解压
def _download_and_uncompress_dataset(dataset_dir):

    filename = _DATA_URL.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(_DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dataset_dir)

# 删除临时文件
def _clean_up_temporary_files(dataset_dir):

    filename = _DATA_URL.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)
    tf.gfile.Remove(filepath)

    tmp_dir = os.path.join(dataset_dir, 'cifar-10-batches-py')
    tf.gfile.DeleteRecursively(tmp_dir)
    pass

#  如果目录下存在cifar10文件， 保存为tfrecord格式， 没有就下载一个
def run(dataset_dir):

    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    training_filename = _get_output_filename(dataset_dir, 'train')
    testing_filename = _get_output_filename(dataset_dir, 'test')

    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    # dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)

    # 训练数据
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        offset = 0
        for i in range(_NUM_TRAIN_FILES):
            filename = os.path.join(dataset_dir, 'cifar-10-batches-py', 'data_batch_%d' % (i + 1))  # 1-indexed.
            offset = _add_to_tfrecord(filename, tfrecord_writer, offset)

    # 测试数据
    with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
        _add_to_tfrecord(os.path.join(dataset_dir, 'cifar-10-batches-py', 'test_batch'), tfrecord_writer)

    # 标签 {1： bird。。。}
    labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    _clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the Cifar10 dataset!')
    pass



if __name__ == '__main__':

    run('data')