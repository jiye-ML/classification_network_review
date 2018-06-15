import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('train_dir', 'model/',
                           'Directory where checkpoints and event logs are written to.')


tf.app.flags.DEFINE_integer('num_readers', 32,
                            'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer('num_preprocessing_threads', 32,
                            'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer('log_every_n_steps', 10, 'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer('save_summaries_secs', 600,
                            'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer('save_interval_secs', 600,
                            'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_boolean('gpu_allow_growth', True, 'Allow dynamic usage of GPU memory')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'The weight decay on the model weights.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential',
                           'learning rate is decayed: "fixed", "exponential",  or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')

tf.app.flags.DEFINE_float('end_learning_rate', 0.0001,
                          'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float('label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.76, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float('num_epochs_per_decay', 50, 'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          'The decay to use for the moving average.If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string('dataset_name', 'cifar10', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string('dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string('dataset_dir', 'data', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer('labels_offset', 0,
                            'An offset for the labels in the dataset. This flag is primarily used to '
                            'evaluate the VGG and ResNet architectures which do not use a background '
                            'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string('model_name', 'inception_v4', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string('preprocessing_name', None,
                           'The name of the preprocessing to use. If left as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer('batch_size', 256, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer('train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', 200000, 'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', 'model/inception_v4.ckpt',
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', 'InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits',
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', 'InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits',
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean('ignore_missing_vars', True, 'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS



class Train():

    def __init__(self):
        pass

    # summary操作
    def summary(self):
        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for end_points.
        for end_point in self._end_points:
            x = self._end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point, tf.nn.zero_fraction(x)))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        summaries.add(tf.summary.scalar('learning_rate', self._learning_rate))

        # Add total_loss to summary.
        summaries.add(tf.summary.scalar('total_loss', self._total_loss))

        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Merge all summaries together.
        return tf.summary.merge(list(summaries), name='summary_op')
        pass

    def run(self):

        with tf.Graph().as_default():

            # data
            if not FLAGS.dataset_dir:
                raise ValueError('You must supply the dataset directory with --dataset_dir')
            self._dataset_dir = FLAGS.dataset_dir
            # dataset
            self._dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

            # net
            network_fn = nets_factory.get_network_fn(
                FLAGS.model_name, num_classes=(self._dataset.num_classes - FLAGS.labels_offset),
                weight_decay=FLAGS.weight_decay, is_training=True)

            # 日志
            tf.logging.set_verbosity(tf.logging.INFO)

            # Create global_step
            global_step = tf.train.create_global_step()

            preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
            image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=True)

            # dataset provider
            provider = slim.dataset_data_provider.DatasetDataProvider(self._dataset, num_readers=FLAGS.num_readers,
                                                                      common_queue_capacity=20 * FLAGS.batch_size,
                                                                      common_queue_min=10 * FLAGS.batch_size)
            [image, label] = provider.get(['image', 'label'])
            label -= FLAGS.labels_offset

            train_image_size = FLAGS.train_image_size or network_fn.default_image_size

            image = image_preprocessing_fn(image, train_image_size, train_image_size)

            images, labels = tf.train.batch(
                [image, label], batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads, capacity=5 * FLAGS.batch_size)
            labels = slim.one_hot_encoding(labels, self._dataset.num_classes - FLAGS.labels_offset)
            batch_queue = slim.prefetch_queue.prefetch_queue([images, labels], capacity=2)

            # model
            images, labels = batch_queue.dequeue()
            self._logits, self._end_points = network_fn(images)

            # loss
            if 'AuxLogits' in self._end_points:
                tf.losses.softmax_cross_entropy(logits=self._end_points['AuxLogits'], onehot_labels=labels,
                                                label_smoothing=FLAGS.label_smoothing, weights=0.4, scope='aux_loss')
            tf.losses.softmax_cross_entropy(logits=self._logits, onehot_labels=labels,
                                            label_smoothing=FLAGS.label_smoothing, weights=1.0)

            # the updates for the batch_norm variables created by network_fn.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            # Configure the moving averages
            if FLAGS.moving_average_decay:
                moving_average_variables = slim.get_model_variables()
                # 滑动平均， decay=min(decay, (1+num_updates) / (10+num_updates)) shadow_var = decay * shadow_var + (1-decay)*var
                variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
            else:
                moving_average_variables, variable_averages = None, None

            # Configure the optimization procedure.
            self._learning_rate = self._configure_learning_rate(global_step)
            optimizer = self._configure_optimizer(self._learning_rate)

            if FLAGS.moving_average_decay:
                # Update ops executed locally by trainer.
                update_ops.append(variable_averages.apply(moving_average_variables))

            # Variables to train.
            variables_to_train = self._get_variables_to_train()

            #  and returns a train_tensor and summary_op
            self._total_loss, gradients = self.optimize(optimizer, var_list=variables_to_train)

            # Create gradient updates.
            grad_updates = optimizer.apply_gradients(gradients, global_step=global_step)
            update_ops.append(grad_updates)

            update_op = tf.group(*update_ops)
            with tf.control_dependencies([update_op]):
                self._train_tensor = tf.identity(self._total_loss, name='train_op')

            # summary
            self._summary_op = self.summary()

            # session
            self._session_config = tf.ConfigProto()
            self._session_config.gpu_options.allow_growth = FLAGS.gpu_allow_growth

            # Kicks off the training. #
            slim.learning.train(self._train_tensor, logdir=FLAGS.train_dir, init_fn=self._get_init_fn(),
                                summary_op=self._summary_op, number_of_steps=FLAGS.max_number_of_steps,
                                log_every_n_steps=FLAGS.log_every_n_steps, save_summaries_secs=FLAGS.save_summaries_secs,
                                save_interval_secs=FLAGS.save_interval_secs, session_config=self._session_config)
            pass


    # Configures the learning rate.
    def _configure_learning_rate(self, global_step):

        decay_steps = int(self._dataset.num_samples / FLAGS.batch_size * FLAGS.num_epochs_per_decay)

        if FLAGS.learning_rate_decay_type == 'exponential':
            return tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps,
                                              FLAGS.learning_rate_decay_factor, staircase=True,
                                              name='exponential_decay_learning_rate')
        elif FLAGS.learning_rate_decay_type == 'fixed':
            return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
        elif FLAGS.learning_rate_decay_type == 'polynomial':
            return tf.train.polynomial_decay(FLAGS.learning_rate, global_step, decay_steps, FLAGS.end_learning_rate,
                                             power=1.0, cycle=False, name='polynomial_decay_learning_rate')
        else:
            raise ValueError('learning_rate_decay_type [%s] was not recognized', FLAGS.learning_rate_decay_type)

    # 配置优化函数
    def _configure_optimizer(self, learning_rate):
        return tf.train.AdamOptimizer(learning_rate)

    # 初始化函数： 当模型最初始加载的时候才需要
    def _get_init_fn(self):

        if FLAGS.checkpoint_path is None:
            return None

        # checkpoint存在的化就忽略
        if tf.train.latest_checkpoint(FLAGS.train_dir):
            tf.logging.info('Ignoring --checkpoint_path because a checkpoint already exists in %s' % FLAGS.train_dir)
            return None
        # 不需要加载层
        exclusions = []
        if FLAGS.checkpoint_exclude_scopes:
            exclusions = [scope.strip() for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

        # 需要加载的变量
        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        tf.logging.info('Fine-tuning from %s' % checkpoint_path)

        return slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore,
                                              ignore_missing_vars=FLAGS.ignore_missing_vars)

    # 返回需要训练的变量： 在初始化fine tune的时候有些层变量没有加载，需要在这里加入到train的集合中
    def _get_variables_to_train(self):

        if FLAGS.trainable_scopes is None:
            return tf.trainable_variables()
        else:
            scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        return variables_to_train

    # 计算损失
    def _optimize(self, optimizer, regularization_losses, **kwargs):
        """
        Returns:
          A tuple (loss, grads_and_vars).
        """
        sum_loss = None
        loss = None
        regularization_loss = None
        all_losses = []
        losses = tf.get_collection(tf.GraphKeys.LOSSES)
        if losses:
            loss = tf.add_n(losses)
            all_losses.append(loss)
        if regularization_losses:
            regularization_loss = tf.add_n(regularization_losses, name='regularization_loss')
            all_losses.append(regularization_loss)
        if all_losses:
            sum_loss = tf.add_n(all_losses)
        # Add the summaries out of the clone device block.
        if loss is not None:
            tf.summary.scalar('/clone_loss', loss)
        if regularization_loss is not None:
            tf.summary.scalar('regularization_loss', regularization_loss)

        grad = None
        if sum_loss is not None:
            grad = optimizer.compute_gradients(sum_loss, **kwargs)

        return sum_loss, grad

    # 优化
    def optimize(self, optimizer, regularization_losses=None, **kwargs):
        """
         A tuple (total_loss, grads_and_vars).
        """
        if regularization_losses is None:
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        return self._optimize(optimizer, regularization_losses, **kwargs)

    pass


if __name__ == '__main__':

    Train().run()
