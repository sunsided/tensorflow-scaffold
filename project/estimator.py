import tensorflow as tf

from project.models import model_builder


def model_fn(features, labels, mode, params):
    """ Model function to be used by the estimator.

    Returns:
      An EstimatorSpec object.
    """

    model = model_builder(params.model[])

    # define loss, training ops, predictions, metrics and hooks

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=None,
                                      loss=total_loss,
                                      train_op=train_op,
                                      training_hooks=[logging_hook])
