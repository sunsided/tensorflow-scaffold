import tensorflow as tf


def my_model(features):
    """ Base model definitions """
    return tf.layers.dense(features, 1024)
