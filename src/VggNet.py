import tensorflow as tf
from src import layers


class VGGNet:

    def __init__(self, type_number, image_size, image_channel, batch_size):
        self._type_number = type_number
        self._image_size = image_size
        self._image_channel = image_channel
        self._batch_size = batch_size

        pass

    # 网络
    # keep_prob=0.7
    def fit(self, input_op, **kw):
        first_out = 64

        # 卷积层 1
        conv_1_1  = layers.conv(input_op, 3, 3, first_out, 1, 1, name="conv_1_1")
        conv_1_2 = layers.conv(conv_1_1, 3, 3, first_out, 1, 1, name="conv_1_2")
        pool_1 = layers.max_pool(conv_1_2, 2, 2, 2, 2, name="pool_1")

        # 卷积层 2
        conv_2_1  = layers.conv(pool_1, 3, 3, first_out * 2, 1, 1, name="conv_2_1")
        conv_2_2 = layers.conv(conv_2_1, 3, 3, first_out * 2, 1, 1, name="conv_2_2")
        pool_2 = layers.max_pool(conv_2_2, 2, 2, 2, 2, name="pool_2")

        # 卷积层 3
        conv_3_1  = layers.conv(pool_2, 3, 3, first_out * 4, 1, 1, name="conv_3_1")
        conv_3_2 = layers.conv(conv_3_1, 3, 3, first_out * 4, 1, 1, name="conv_3_2")
        conv_3_3 = layers.conv(conv_3_2, 3, 3, first_out * 4, 1, 1, name="conv_3_3")
        pool_3 = layers.max_pool(conv_3_3, 2, 2, 2, 2, name="pool_3")

        # 卷积层 4
        conv_4_1  = layers.conv(pool_3, 3, 3, first_out * 8, 1, 1, name="conv_4_1")
        conv_4_2 = layers.conv(conv_4_1, 3, 3, first_out * 8, 1, 1, name="conv_4_2")
        conv_4_3 = layers.conv(conv_4_2, 3, 3, first_out * 8, 1, 1, name="conv_4_3")
        pool_4 = layers.max_pool(conv_4_3, 2, 2, 2, 2, name="pool_4")

        # 卷积层 5
        # conv_5_1  = layers.conv(pool_4, 3, 3, first_out * 8, 1, 1, name="conv_5_1")
        # conv_5_2 = layers.conv(conv_5_1, 3, 3, first_out * 8, 1, 1, name="conv_5_2")
        # conv_5_3 = layers.conv(conv_5_2, 3, 3, first_out * 8, 1, 1, name="conv_5_3")
        # pool_5 = layers.max_pool(conv_5_3, 2, 2, 2, 2, name="pool_5")

        pool_5 = pool_4

        # 拉直
        shp = pool_5.get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        reshape_pool_5 = tf.reshape(pool_5, [-1, flattened_shape], name="reshape_pool_5")

        # 全连接层
        fc_6_drop = layers.dropout(layers.fc(reshape_pool_5, 1024, name='fc_6'), keep_prob=kw["keep_prob"])
        fc_7_drop = layers.dropout(layers.fc(fc_6_drop, 1024, name='fc_7'), keep_prob=kw["keep_prob"])
        fc_8 = layers.fc(fc_7_drop, self._type_number, name='fc_8', relu=False)

        print("64 4 512")

        softmax = tf.nn.softmax(fc_8)
        prediction = tf.argmax(softmax, 1)

        return fc_8, softmax, prediction

    pass
