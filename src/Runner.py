import argparse
import numpy as np
import tensorflow as tf

from src.Tools import Tools
from src.Data import Cifar10Data
from src.VggNet import VGGNet


class Runner:

    def __init__(self, data, classifies, learning_rate=0.001, decay_steps=1000, **kw):
        self._data = data
        self._type_number = self._data.type_number
        self._image_size = self._data.image_size
        self._image_channel = self._data.image_channel
        self._batch_size = self._data.batch_size
        self._classifies = classifies

        input_shape = [self._batch_size, self._image_size, self._image_size, self._image_channel]
        self._images = tf.placeholder(shape=input_shape, dtype=tf.float32)
        self._labels = tf.placeholder(dtype=tf.int32, shape=[self._batch_size])

        # 学习率衰减
        self._global_step = tf.Variable(0, trainable=False)
        self._learning_rate = tf.train.exponential_decay(learning_rate, global_step=self._global_step,
                                                         decay_steps=decay_steps, decay_rate=0.9)
        self._keep_prob = tf.placeholder(dtype=tf.float32)
        # loss
        self._logits, self._softmax, self._prediction = self._classifies.fit(self._images, keep_prob=self._keep_prob, **kw)
        self._entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._labels, logits=self._logits)

        self._loss = tf.reduce_mean(self._entropy)
        self._solver = tf.train.AdamOptimizer(learning_rate = self._learning_rate).minimize(self._loss, global_step=self._global_step)


        # sess
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self._saver = tf.train.Saver()
        pass

    # 训练网络
    def train(self, epochs, save_model, min_loss, print_loss, test, save, keep_prob=0.5):
        self._sess.run(tf.global_variables_initializer())
        # 如果没有训练， 就先加载
        # if os.path.isfile(save_model):
        #     self._saver.restore(self._sess, save_model)

        epoch = 0
        for epoch in range(epochs):
            images, labels = self._data.next_train()
            loss, soft_max, _, learning_rate = self._sess.run(fetches=[self._loss, self._softmax,
                                                                       self._solver, self._learning_rate],
                                                              feed_dict={self._images: images, self._labels: labels,
                                                                         self._keep_prob: keep_prob})
            if epoch % print_loss == 0:
                Tools.print_info("{} learning_rate:{} loss {}".format(epoch, learning_rate, loss))
            # if loss < min_loss:
            #     break
            if epoch % test == 0 and epoch != 0:
                self.test()
                pass
            if epoch % save == 0 and epoch != 0:
                self._saver.save(self._sess, save_path=save_model)
            pass
        Tools.print_info("{}: train end".format(epoch))
        self.test()
        Tools.print_info("test end")
        pass

    # 测试网络
    def test(self):
        all_ok = 0
        # test_epoch = self._data.test_batch_number
        test_epoch = 100
        keep_prob = 0
        for now in range(test_epoch):
            images, labels = self._data.next_test(now)
            prediction, keep_prob = self._sess.run([self._prediction, self._keep_prob], feed_dict={self._images: images,
                                                                             self._keep_prob: 1.0})
            all_ok += np.sum(np.equal(labels, prediction))
        all_number = test_epoch * self._batch_size
        Tools.print_info("the result is {} ({}/{}  keep_prob {})".format(all_ok / (all_number * 1.0), all_ok,
                                                                        all_number, keep_prob))
        pass

    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, default="VggNet", help="name")
    parser.add_argument("-epochs", type=int, default=50000000, help="train epoch number")
    parser.add_argument("-batch_size", type=int, default=256, help="batch size")
    parser.add_argument("-type_number", type=int, default=10, help="type number")
    parser.add_argument("-image_size", type=int, default=32, help="image size")
    parser.add_argument("-image_channel", type=int, default=3, help="image channel")
    parser.add_argument("-keep_prob", type=float, default=0.7, help="keep prob")
    parser.add_argument("-learning_rate", type=float, default=0.0001, help="learning_rate")
    parser.add_argument("-decay_steps", type=int, default=1000, help="decay_steps")
    parser.add_argument("-skip_layers", type=str, default=['conv5', 'fc6', 'fc7', 'fc8'], help="finetune skip these layers")
    parser.add_argument("-zip_file", type=str, default="data/resisc45.zip", help="zip file path")
    args = parser.parse_args()

    output_param = "name={},epochs={},batch_size={},type_number={},image_size={},image_channel={},zip_file={},keep_prob={}"
    Tools.print_info(output_param.format(args.name, args.epochs, args.batch_size, args.type_number,
                                         args.image_size, args.image_channel, args.zip_file, args.keep_prob))

    # now_train_path, now_test_path = PreData.main(zip_file=args.zip_file)
    # now_data = Data(batch_size=args.batch_size, type_number=args.type_number, image_size=args.image_size,
    #                 image_channel=args.image_channel, train_path=now_train_path, test_path=now_test_path)

    now_data = Cifar10Data(batch_size=args.batch_size, type_number=args.type_number, image_size=args.image_size,
                    image_channel=args.image_channel)

    now_net = VGGNet(now_data.type_number, now_data.image_size, now_data.image_channel, now_data.batch_size)

    runner = Runner(data=now_data, classifies=now_net, learning_rate=args.learning_rate, decay_steps=args.decay_steps)
    runner.train(epochs=args.epochs, save_model=Tools.new_dir("model/") + args.name + args.name + ".ckpt",
                 min_loss=1e-10, print_loss=20, test=1000, save=10000, keep_prob=args.keep_prob)

    pass
