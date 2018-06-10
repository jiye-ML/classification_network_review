import numpy as np
from layers import *


class AlexNet:

    def __init__(self, type_number, image_size, image_channel, batch_size,
                 weight_path="model/bvlc_alexnet.npy",  skip_layer="conv5"):
        '''
        :param type_number: 需要分类的类别数目
        :param image_size:  
        :param image_channel: 
        :param batch_size: 
        :param weight_path:  如果需要从npy加载权重，
        :param skip_layer:  跳过某些不需要的层
        '''
        self._type_number = type_number
        self._image_size = image_size
        self._image_channel = image_channel
        self._batch_size = batch_size

        self._weight_path = weight_path
        self._skip_layer = skip_layer
        pass

    def fit(self, input_op, **kw):
        #  256 X 256 X 3
        conv1 = conv(input_op, 11, 11, 96, 4, 4, name='conv1')
        norm1 = lrn(conv1, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        # conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        # pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
        pool5 = conv4

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        dim = pool5.get_shape()[1].value * pool5.get_shape()[2].value * pool5.get_shape()[3].value
        flattened = tf.reshape(pool5, [-1, dim])
        dropout6 = dropout(fc(flattened, dim, 4096, name='fc6'), kw['keep_prob'])

        # 7th Layer: FC (w ReLu) -> Dropout
        dropout7 = dropout(fc(dropout6, 4096, 4096, name='fc7'), kw['keep_prob'])

        # 8th Layer: FC and return unscaled activations
        logits = fc(dropout7, 4096, self._type_number, relu=False, name='fc8')

        softmax = tf.nn.softmax(logits)
        prediction = tf.argmax(softmax, 1)
        return logits, softmax, prediction

    # 从 npy中加载模型
    def load_initial_weights(self, session):
        weights_dict = np.load(self._weight_path, encoding='bytes').item()

        for op_name in weights_dict:
            # 跳过需要重新训练的层
            if op_name not in self._skip_layer:
                with tf.variable_scope(op_name, reuse=True):
                    # 加载权重和偏置
                    for data in weights_dict[op_name]:
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))

    pass





