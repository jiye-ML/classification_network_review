import tensorflow as tf

from datasets import download_and_convert_cifar10
from datasets import download_and_convert_flowers
from datasets import download_and_convert_mnist
from datasets import convert_tfrecord

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string( 'dataset_name', 'cifar10',
                            'The name of the dataset to convert, one of "cifar10", "flowers", "mnist".')

tf.app.flags.DEFINE_string('dataset_dir', 'data',
                           'The directory where the output TFRecords and temporary files are saved.')

def main(_):
    if not FLAGS.dataset_name:
        raise ValueError('You must supply the dataset name with --dataset_name')
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    if FLAGS.dataset_name == 'cifar10':
        download_and_convert_cifar10.run(FLAGS.dataset_dir)
    elif FLAGS.dataset_name == 'flowers':
        download_and_convert_flowers.run(FLAGS.dataset_dir)
    elif FLAGS.dataset_name == 'mnist':
        download_and_convert_mnist.run(FLAGS.dataset_dir)
    elif FLAGS.dataset_name == 'custom':
        convert_tfrecord.run(FLAGS.dataset_dir, dataset_name=FLAGS.dataset_name)
    else:
        convert_tfrecord.run(FLAGS.dataset_dir, dataset_name=FLAGS.dataset_name)


if __name__ == '__main__':
    tf.app.run()