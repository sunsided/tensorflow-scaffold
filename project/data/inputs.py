"""Estimator inputs definitions. """
import tensorflow as tf


def input_fn(params, is_training):
    """Load data here and return features and labels tensors."""
    features = tf.constant(0)
    labels = tf.constant(0)

    return features, labels
