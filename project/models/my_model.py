from argparse import Namespace
from typing import Dict
import tensorflow as tf


def my_model(features: tf.Tensor, mode: str, params: Namespace) -> Dict[str, tf.Tensor]:
    """ Base model definitions """
    with tf.variable_scope('model'):
        net = tf.layers.conv2d(features, filters=8, kernel_size=1, strides=1, padding='valid',
                               data_format='channels_last',
                               activation=None,
                               kernel_initializer=tf.initializers.orthogonal,
                               name='conv_1')
        net = tf.layers.conv2d(net, filters=32, kernel_size=3, strides=3, padding='valid',
                               data_format='channels_last',
                               activation=tf.nn.selu,
                               kernel_initializer=tf.initializers.orthogonal,
                               name='conv_2')

        net = tf.layers.conv2d(net, filters=8, kernel_size=1, strides=1, padding='valid',
                               data_format='channels_last',
                               activation=None,
                               kernel_initializer=tf.initializers.orthogonal,
                               name='conv_3')
        net = tf.layers.conv2d(net, filters=32, kernel_size=3, strides=3, padding='valid',
                               data_format='channels_last',
                               activation=tf.nn.selu,
                               kernel_initializer=tf.initializers.orthogonal,
                               name='conv_4')

        net = tf.layers.conv2d(net, filters=8, kernel_size=1, strides=1, padding='valid',
                               data_format='channels_last',
                               activation=None,
                               kernel_initializer=tf.initializers.orthogonal,
                               name='conv_5')
        net = tf.layers.conv2d(net, filters=32, kernel_size=3, strides=3, padding='valid',
                               data_format='channels_last',
                               activation=tf.nn.selu,
                               kernel_initializer=tf.initializers.orthogonal,
                               name='conv_6')

        net = tf.layers.conv2d(net, filters=8, kernel_size=1, strides=1, padding='valid',
                               data_format='channels_last',
                               activation=None,
                               kernel_initializer=tf.initializers.orthogonal,
                               name='conv_7')
        net = tf.layers.conv2d(net, filters=32, kernel_size=3, strides=3, padding='valid',
                               data_format='channels_last',
                               activation=tf.nn.selu,
                               kernel_initializer=tf.initializers.orthogonal,
                               name='conv_8')

        net = tf.reduce_mean(net, [1, 2], name='global_avg_pooling')

        # To stay 2D convolutional, we need to re-add the missing dimensions.
        net = tf.reshape(net, shape=(-1, 1, 1, net.shape[1].value), name='reshape')
        net = tf.layers.conv2d(net, filters=32, kernel_size=1, strides=1, padding='valid',
                               data_format='channels_last',
                               activation=tf.nn.selu,
                               kernel_initializer=tf.initializers.orthogonal,
                               name='conv_9')

        net = tf.layers.conv2d(net, filters=1, kernel_size=1, strides=1, padding='valid',
                               data_format='channels_last',
                               activation=tf.nn.selu,
                               kernel_initializer=tf.initializers.orthogonal,
                               name='conv_10')

        logits = tf.layers.flatten(net, name='logits')
        prediction = tf.nn.sigmoid(logits, name='prediction')

    return {
        'logits': logits,
        'predictions': prediction
    }
