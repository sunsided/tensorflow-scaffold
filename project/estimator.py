from argparse import Namespace
import tensorflow as tf

from project.models import model_builder
from project.models.Model import Losses


def _report_losses(losses: Losses) -> tf.Tensor:
    """
    Splits the training loss from the tensors that should
    be reported as part of the regular training output.
    :param losses: The losses to process.
    :return: The training loss.
    """
    if isinstance(losses, tf.Tensor):
        return losses
    assert isinstance(losses, tuple)
    loss, to_report = losses

    assert isinstance(to_report, dict)
    for name, tensor in to_report.items():
        tf.identity(tensor, name)
    return loss


def model_fn(features: tf.Tensor, labels: tf.Tensor, mode: str, params: Namespace) -> tf.estimator.EstimatorSpec:
    """ Model function to be used by the estimator.

    Returns:
      An EstimatorSpec object.
    """

    # TODO: Support multi-headed models

    # In order to validate the input images in TensorBoard, we're adding them to the images collection.
    tf.summary.image('input_image', features)

    # We now construct the models according to the configuration.
    model = model_builder(params.model, params)  # TODO: Use "real" hyperparameters here
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
        loss = model.loss(labels, net)
        metrics = model.eval_metrics(labels, net)

    # During training, we want to report losses.
    loss = _report_losses(loss)

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
