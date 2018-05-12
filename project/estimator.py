from argparse import Namespace
import tensorflow as tf

from project.models import model_builder


def model_fn(features: tf.Tensor, labels: tf.Tensor, mode: str, params: Namespace) -> tf.estimator.EstimatorSpec:
    """ Model function to be used by the estimator.

    Returns:
      An EstimatorSpec object.
    """

    # TODO: Support multi-headed models

    model = model_builder(params.model)
    net = model(features, mode, params)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': net['predictions'],
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    with tf.variable_scope('metrics'):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=net['logits'], name='xentropy')
        total_loss = tf.add(loss, tf.losses.get_regularization_loss(), name='total_loss')

        accuracy = tf.metrics.accuracy(labels=labels, predictions=net['predictions'], name='accuracy')
        precision = tf.metrics.precision(labels=labels, predictions=net['predictions'], name='precision')
        recall = tf.metrics.recall(labels=labels, predictions=net['predictions'], name='recall')
        auc = tf.metrics.auc(labels=labels, predictions=net['predictions'], name='auc')

    total_loss = tf.identity(total_loss, name='loss')
    learning_rate = tf.constant(params.learning_rate, dtype=tf.float32, name='learning_rate')

    with tf.variable_scope('optimization'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(total_loss, tf.train.get_or_create_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=None,
                                          loss=loss,
                                          train_op=train_op,
                                          training_hooks=[])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=None,
                                          loss=loss,
                                          eval_metric_ops={
                                              'accuracy': accuracy,
                                              'precision': precision,
                                              'recall': recall,
                                              'auc': auc
                                          })

    assert False, 'An unknown mode was specified: {}'.format(mode)
