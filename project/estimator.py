from argparse import Namespace
from typing import Dict
import tensorflow as tf

from project.models import model_builder


def eval_metrics(labels: tf.Tensor, prediction: tf.Tensor) -> Dict[str, tf.Operation]:
    # For area under curve metrics we need the raw predictions.
    auc = tf.metrics.auc(labels=labels, predictions=prediction, name='auc')

    # Accuracy, precision and recall metrics are meant for single-class classification.
    # They expect a value that can be converted to bool. Here, we cutoff at a specific threshold.
    label_hot = tf.cast(labels, dtype=tf.bool)
    prediction_hot = tf.greater_equal(prediction, 0.5)
    accuracy = tf.metrics.accuracy(labels=label_hot, predictions=prediction_hot, name='accuracy')
    precision = tf.metrics.precision(labels=label_hot, predictions=prediction_hot, name='precision')
    recall = tf.metrics.recall(labels=label_hot, predictions=prediction_hot, name='recall')

    return {'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc}


def model_fn(features: tf.Tensor, labels: tf.Tensor, mode: str, params: Namespace) -> tf.estimator.EstimatorSpec:
    """ Model function to be used by the estimator.

    Returns:
      An EstimatorSpec object.
    """

    # TODO: Support multi-headed models

    # In order to validate the input images in TensorBoard, we're adding them to the images collection.
    tf.summary.image('input_image', features)

    # We now construct the models according to the configuration.
    model = model_builder(params.model, params)  # TODO: Use hyperparameters here
    net = model(features, mode)

    # During prediction, we directly return the output of the network.
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': net['predictions'],
        }
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    # For training and testing/evaluation, we need loss and accuracy metrics.
    with tf.variable_scope('metrics'):
        xentropy = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=net['logits'])
        loss = tf.add(xentropy, tf.losses.get_regularization_loss(), name='total_loss')
        metrics = eval_metrics(labels, net['predictions'])

    # TODO: Add to tensors_to_log collection
    tf.identity(xentropy, name='xentropy')
    loss = tf.identity(loss, name='loss')

    # For testing/evaluation, we can now return the
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)

    # TODO: The learning rate should be an untrainable variable
    learning_rate = tf.constant(model.params.learning_rate, dtype=tf.float32, name='learning_rate')

    # We now define the optimizer.
    with tf.variable_scope('optimization'):
        # TODO: Add additional optimizer parameters to hyper-parameters
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

    # For training, we specify the optimizer
    assert mode == tf.estimator.ModeKeys.TRAIN, 'An unknown mode was specified: {}'.format(mode)
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
