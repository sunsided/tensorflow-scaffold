from typing import Optional
import tensorflow as tf
from tensorflow.contrib.training import HParams
from .visualization import put_kernels_on_grid
from .Model import Model, HyperParameters


class SimpleModel(Model):
    def __init__(self, params: Optional[HyperParameters] = None):
        super().__init__(params)

    def default_hparams(self):
        return HParams(learning_rate=1e-5,
                       dropout_rate=0.5,
                       l2_regularization=1e-5,
                       fine_tuning=False)

    def build(self, features: tf.Tensor, mode: str):
        with tf.variable_scope('input_normalization'):
            # We perform global mean and variance normalization.
            features = tf.subtract(features, .5, name='mean_normalize')  # TODO: Obtain channel-correct means
            features = tf.multiply(features, 2., name='variance_normalize')

        with tf.variable_scope('model'):
            net = tf.layers.conv2d(features, filters=32, kernel_size=7, strides=5, padding='valid',
                                   data_format='channels_last',
                                   activation=tf.nn.selu,
                                   kernel_initializer=tf.initializers.orthogonal,
                                   name='conv_1')

            net = tf.layers.conv2d(net, filters=8, kernel_size=1, strides=1, padding='valid',
                                   data_format='channels_last',
                                   activation=tf.nn.selu,
                                   kernel_initializer=tf.initializers.orthogonal,
                                   name='conv_2')
            net = tf.layers.conv2d(net, filters=32, kernel_size=3, strides=3, padding='valid',
                                   data_format='channels_last',
                                   activation=tf.nn.selu,
                                   kernel_initializer=tf.initializers.orthogonal,
                                   name='conv_3')

            net = tf.layers.conv2d(net, filters=8, kernel_size=1, strides=1, padding='valid',
                                   data_format='channels_last',
                                   activation=tf.nn.selu,
                                   kernel_initializer=tf.initializers.orthogonal,
                                   name='conv_4')
            net = tf.layers.conv2d(net, filters=32, kernel_size=3, strides=3, padding='valid',
                                   data_format='channels_last',
                                   activation=tf.nn.selu,
                                   kernel_initializer=tf.initializers.orthogonal,
                                   name='conv_5')

            net = tf.reduce_mean(net, [1, 2], name='global_avg_pooling')
            net = tf.nn.selu(net)

            # To stay 2D convolutional, we need to re-add the missing dimensions.
            net = tf.reshape(net, shape=(-1, 1, 1, net.shape[1].value), name='reshape')
            net = tf.layers.conv2d(net, filters=256, kernel_size=1, strides=1, padding='valid',
                                   data_format='channels_last',
                                   activation=tf.nn.selu,
                                   kernel_initializer=tf.initializers.orthogonal,
                                   name='conv_6')

            net = tf.layers.dropout(net, rate=self.params.dropout_rate, name='dropout', training=mode == tf.estimator.ModeKeys.TRAIN)

            net = tf.layers.conv2d(net, filters=1, kernel_size=1, strides=1, padding='valid',
                                   data_format='channels_last',
                                   activation=None,
                                   kernel_initializer=tf.initializers.orthogonal,
                                   name='conv_7')

            logits = tf.squeeze(tf.layers.flatten(net, name='flatten'), axis=1, name='logits')
            prediction = tf.nn.sigmoid(logits, name='prediction')

            grid = put_kernels_on_grid(tf.get_default_graph().get_tensor_by_name('model/conv_1/kernel:0'), 4, 8)
            tf.summary.image('conv1/kernels', grid)

            grid = tf.get_default_graph().get_tensor_by_name('model/conv_1/kernel:0')
            tf.summary.histogram('conv1/weights', grid)

            grid = tf.get_default_graph().get_tensor_by_name('model/conv_2/kernel:0')
            tf.summary.histogram('conv2/weights', grid)

            grid = tf.get_default_graph().get_tensor_by_name('model/conv_3/kernel:0')
            tf.summary.histogram('conv3/weights', grid)

            grid = tf.get_default_graph().get_tensor_by_name('model/conv_4/kernel:0')
            tf.summary.histogram('conv4/weights', grid)

            grid = tf.get_default_graph().get_tensor_by_name('model/conv_5/kernel:0')
            tf.summary.histogram('conv5/weights', grid)

            grid = tf.get_default_graph().get_tensor_by_name('model/conv_6/kernel:0')
            tf.summary.histogram('conv6/weights', grid)

            grid = tf.get_default_graph().get_tensor_by_name('model/conv_7/kernel:0')
            tf.summary.histogram('conv7/weights', grid)

        return {'logits': logits,
                'predictions': prediction}
