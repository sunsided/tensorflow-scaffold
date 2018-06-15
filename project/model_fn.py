from argparse import Namespace
import tensorflow as tf

from project.models import model_builder
from project.models.Model import Losses, Metrics, Output


def prediction_spec(net: Output) -> tf.estimator.EstimatorSpec:
    """
    Returns the Estimator specification for predictions.
    :param net: The network output.
    :return: The EstimatorSpec.
    """
    predictions = {
        'probabilities': net['predictions'],
    }
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })


def eval_spec(loss: tf.Tensor, metrics: Metrics) -> tf.estimator.EstimatorSpec:
    """
    Returns the Estimator specification for evaluations.
    :param loss: The evaluation loss.
    :param metrics: The evaluation metrics.
    :return: The EstimatorSpec.
    """
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL,
                                      loss=loss, eval_metric_ops=metrics)


def train_spec(loss: tf.Tensor, train_op: tf.Operation) -> tf.estimator.EstimatorSpec:
    """
    Returns the Estimator specification for training.
    :param loss: The training loss.
    :param train_op: The training operation.
    :return: The EstimatorSpec.
    """
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN,
                                      loss=loss, train_op=train_op)


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
        return prediction_spec(net)

    # For training and testing/evaluation, we need loss and accuracy metrics.
    with tf.variable_scope('metrics'):
        loss = model.loss(labels, net)
        metrics = model.eval_metrics(labels, net)

    # During training, we want to report losses.
    loss = _report_losses(loss)

    # For testing/evaluation, we can now return the
    if mode == tf.estimator.ModeKeys.EVAL:
        return eval_spec(net, metrics)

    # We store the learning rate as a variable so that we can modify (or feed)
    # it later on.
    learning_rate = tf.Variable(model.params.learning_rate, dtype=tf.float32, trainable=False, name='learning_rate')

    # We now define the optimizer.
    with tf.variable_scope('optimization'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=model.params.adam_beta1,
                                           beta2=model.params.adam_beta2,
                                           epsilon=model.params.adam_epsilon)
        train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

        # We also want the learning rate reported in TensorBoard.
        tf.summary.scalar('learning_rate', learning_rate)

    # For training, we specify the optimizer
    assert mode == tf.estimator.ModeKeys.TRAIN, 'An unknown mode was specified: {}'.format(mode)
    return train_spec(loss, train_op)
