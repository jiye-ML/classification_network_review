import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim


class ResNet:

    def __init__(self, type_number, image_size, image_channel, batch_size):
        self._type_number = type_number
        self._image_size = image_size
        self._image_channel = image_channel
        self._batch_size = batch_size
        pass

    # 残差基本单元
    class _Block(collections.namedtuple("Block", ["scope", "unit_fn", "args"])):
        pass

    # 卷积层
    @staticmethod
    def _bottleneck_conv2d(inputs, num_outputs, kernel_size, stride):
        if stride == 1:
            padding = "SAME"
        else:
            padding = "VALID"
            padding_begin = (kernel_size - 1) // 2
            padding_end = kernel_size - 1 - padding_begin
            inputs = tf.pad(inputs, [[0, 0], [padding_begin, padding_end], [padding_begin, padding_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding=padding)

    # Block中unit_fn的实现
    @slim.add_arg_scope
    def _bottleneck(self, inputs, depth, depth_bottleneck, stride, scope=None):
        with tf.variable_scope(scope, "bottleneck_v2", [inputs]):
            pre_activation = slim.batch_norm(inputs, activation_fn=tf.nn.relu)

            # 定义直连的x：将两者的通道数和空间尺寸处理成一致
            depth_in = inputs.get_shape()[-1].value
            if depth == depth_in:
                # 输入和输出通道数相同的情况, 那么查看特征值图大小是否一致，即下面的if else.
                shortcut = inputs if stride == 1 else slim.max_pool2d(inputs, kernel_size=[1, 1], stride=stride,
                                                                      padding="SAME")
            else:
                # 输入和输出通道数不相同的情况
                shortcut = slim.conv2d(pre_activation, depth, [1, 1], stride=stride, normalizer_fn=None,
                                       activation_fn=None)

            residual = slim.conv2d(pre_activation, depth_bottleneck, kernel_size=[1, 1], stride=1)
            residual = self._bottleneck_conv2d(residual, depth_bottleneck, kernel_size=3, stride=stride)
            residual = slim.conv2d(residual, depth, kernel_size=[1, 1], stride=1, activation_fn=None)

            output = shortcut + residual
        return output

    # 堆叠Blocks
    @staticmethod
    @slim.add_arg_scope
    def _stack_blocks_dense(net, blocks):
        for block in blocks:
            with tf.variable_scope(block.scope, "block", [net]) as sc:
                for i, unit in enumerate(block.args):
                    with tf.variable_scope("unit_%d" % (i + 1), values=[net]):
                        depth, depth_bottleneck, stride = unit
                        net = block.unit_fn(net, depth=depth, depth_bottleneck=depth_bottleneck, stride=stride)
                pass
        return net

    # 构造整个网络
    def _resnet_v2(self, inputs, blocks, global_pool=True, include_root_block=True):
        end_points = {}

        net = inputs
        # 是否加上ResNet网络最前面通常使用的7X7卷积和最大池化
        if include_root_block:
            net = slim.conv2d(net, 64, [7, 7], stride=2, padding="SAME", activation_fn=None, normalizer_fn=None)
            net = slim.max_pool2d(net, [3, 3], stride=2, padding="SAME")

        # 构建ResNet网络
        net = self._stack_blocks_dense(net, blocks)
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)

        # 全局平均池化层
        if global_pool:
            net = tf.reduce_mean(net, [1, 2], keep_dims=True)

        # 分类
        logits = slim.conv2d(net, self._type_number, kernel_size=[1, 1], activation_fn=None, normalizer_fn=None)
        logits = tf.squeeze(logits, [1, 2])  # batch_size X type_number

        softmax = slim.softmax(logits)
        end_points["softmax"] = softmax
        end_points["prediction"] = tf.argmax(softmax, 1)
        return logits, end_points["softmax"], end_points["prediction"]

    # 通用scope
    @staticmethod
    def _resnet_arg_scope(is_training=True, weight_decay=0.0001, bn_decay=0.997, bn_epsilon=1e-5, bn_scale=True):
        batch_norm_params = {
            "is_training": is_training,
            "decay": bn_decay,
            "epsilon": bn_epsilon,
            "scale": bn_scale,
            "updates_collections": tf.GraphKeys.UPDATE_OPS
        }
        conv2d_params = {
            "weights_regularizer": slim.l2_regularizer(weight_decay),
            "weights_initializer": slim.variance_scaling_initializer(),
            "activation_fn": tf.nn.relu,
            "normalizer_fn": slim.batch_norm,
            "normalizer_params": batch_norm_params
        }
        with slim.arg_scope([slim.conv2d], **conv2d_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
                return arg_sc
        pass

    def resnet_v2_50(self, input_op, **kw):
        blocks = [
            self._Block("block1", self._bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            self._Block("block2", self._bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
            self._Block("block3", self._bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
            self._Block("block4", self._bottleneck, [(2048, 512, 1)] * 3)
        ]
        with slim.arg_scope(self._resnet_arg_scope()):
            return self._resnet_v2(input_op, blocks, global_pool=True, include_root_block=True)
        pass

    def resnet_v2_101(self, input_op, **kw):
        blocks = [
            self._Block("block1", self._bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            self._Block("block2", self._bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
            self._Block("block3", self._bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
            self._Block("block4", self._bottleneck, [(2048, 512, 1)] * 3)
        ]
        with slim.arg_scope(self._resnet_arg_scope()):
            return self._resnet_v2(input_op, blocks, global_pool=True, include_root_block=True)
        pass

    def resnet_v2_152(self, input_op, **kw):
        blocks = [
            self._Block("block1", self._bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            self._Block("block2", self._bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
            self._Block("block3", self._bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
            self._Block("block4", self._bottleneck, [(2048, 512, 1)] * 3)
        ]
        with slim.arg_scope(self._resnet_arg_scope()):
            return self._resnet_v2(input_op, blocks, global_pool=True, include_root_block=True)
        pass

    def resnet_v2_200(self, input_op, **kw):
        blocks = [
            self._Block("block1", self._bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            self._Block("block2", self._bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
            self._Block("block3", self._bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
            self._Block("block4", self._bottleneck, [(2048, 512, 1)] * 3)
        ]
        with slim.arg_scope(self._resnet_arg_scope()):
            return self._resnet_v2(input_op, blocks, global_pool=True, include_root_block=True)
        pass

    pass