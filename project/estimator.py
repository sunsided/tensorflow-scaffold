from argparse import Namespace
import tensorflow as tf

from project.models import model_builder


def model_fn(features: tf.Tensor, labels: tf.Tensor, mode: str, params: Namespace) -> tf.estimator.EstimatorSpec:
    """ Model function to be used by the estimator.

    Returns:
      An EstimatorSpec object.
    """

    # TODO: Support multi-headed models

    # Summary for debugging
    tf.summary.image('input_image', features)
    tf.summary.histogram('features', features)

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
        prediction = net['predictions']

        xentropy = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=net['logits'])
        loss = tf.add(xentropy, tf.losses.get_regularization_loss(), name='total_loss')

        # For area under curve metrics we need the raw predictions.
        auc = tf.metrics.auc(labels=labels, predictions=net['predictions'], name='auc')

        # Accuracy, precision and recall metrics are meant for single-class classification.
        # They expect a value that can be converted to bool. Here, we cutoff at a specific threshold.
        label_hot = tf.cast(labels, dtype=tf.bool)
        prediction_hot = tf.greater_equal(prediction, 0.5)
        accuracy = tf.metrics.accuracy(labels=label_hot, predictions=prediction_hot, name='accuracy')
        precision = tf.metrics.precision(labels=label_hot, predictions=prediction_hot, name='precision')
        recall = tf.metrics.recall(labels=label_hot, predictions=prediction_hot, name='recall')

    xentropy = tf.identity(xentropy, name='xentropy')
    loss = tf.identity(loss, name='loss')

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

    learning_rate = tf.constant(params.learning_rate, dtype=tf.float32, name='learning_rate')

    with tf.variable_scope('optimization'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=None,
                                          loss=loss,
                                          train_op=train_op)

    assert False, 'An unknown mode was specified: {}'.format(mode)
