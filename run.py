import argparse
import random
import os
import tensorflow as tf
import numpy as np

from typing import Optional, Any, Dict
from library.hooks import ExamplesPerSecondHook, EvaluationCheckpointSaverHook
from library.parameters import get_project_parameters, YParams
from project import input_fn, model_fn, model_builder


def main(flags: argparse.Namespace):
    # Set TensorFlow logging verbosity
    tf.logging.set_verbosity(tf.logging.INFO if not flags.verbose else tf.logging.WARN)

    # Set random seed
    if flags.random_seed is not None:
        random.seed(flags.random_seed, version=2)
        np.random.seed(flags.random_seed)
        tf.set_random_seed(flags.random_seed)

    # Configure CPU prefetching
    num_gpu = flags.num_gpu if 'num_gpu' in flags else (1 if tf.test.gpu_device_name() else 0)
    if (flags.prefetch_to_device is None) and (num_gpu == 1):
        flags.prefetch_to_device = tf.test.gpu_device_name()

    # Load the hyperparameters
    hparams = YParams(flags.hyperparameter_file, flags.hyperparameter_set)

    # We now obtain the model and replace the parameter with the actual instance.
    model = model_builder(flags.model, hparams)
    flags.model = model

    # Create estimator that trains and evaluates the model
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=flags.model_dir,
        config=build_run_config(flags),
        params=flags
    )

    if not flags.validate:
        run_training_validation_loop(estimator, flags)
    else:
        run_validation(estimator, flags)

    # TODO: Add exporting capability (frozen graph)
    # TODO: Add Experiments API


def run_training_validation_loop(estimator: tf.estimator.Estimator, flags: argparse.Namespace) -> None:
    """
    Runs training and validation in a loop.
    :param estimator: The estimator to train and validate.
    :param flags: The command-line arguments.
    :return:
    """
    tf.logging.info('Running training/validation loop.')
    train_steps = flags.max_train_steps if flags.max_train_steps is not None and flags.max_train_steps >= 0 else None
    eval_steps = flags.max_eval_steps if flags.max_eval_steps is not None and flags.max_eval_steps > 0 else None

    # The tensors to log during training
    train_tensors_to_log = ['learning_rate', 'loss', 'xentropy']

    # The tensors to watch for minimization during evaluation.
    current_validation_state = get_best_validation_metrics(flags)
    eval_tensors_to_monitor = {'loss': current_validation_state.get('loss', None)}

    # Set up hook that outputs training logs every N steps.
    # TODO: Add profiler hooks support: https://www.tensorflow.org/api_docs/python/tf/train/ProfilerHook
    report_every_n_iter = 1000
    train_hooks = [
        tf.train.LoggingTensorHook(tensors=train_tensors_to_log, every_n_iter=report_every_n_iter),
        ExamplesPerSecondHook(batch_size=flags.train_batch_size, every_n_steps=report_every_n_iter)
    ]
    eval_hooks = [
        EvaluationCheckpointSaverHook(checkpoint_dir=flags.best_model_dir, tensors_to_minimize=eval_tensors_to_monitor)
    ]

    for _ in range(flags.train_epochs // flags.epochs_between_evals):
        estimator.train(input_fn=input_fn,
                        hooks=train_hooks,
                        max_steps=train_steps)
        eval_results = estimator.evaluate(input_fn=input_fn,
                                          steps=eval_steps,
                                          hooks=eval_hooks)
        print('\nEvaluation results:\n\t%s\n' % eval_results)
        # TODO: Stop running when accuracy is below flags.stop_threshold


def run_validation(estimator: tf.estimator.Estimator, flags: argparse.Namespace, verbose: bool=True) -> Dict[str, Any]:
    """
    Validates the given estimator.
    :param estimator: The estimator to validate.
    :param flags: The command-line flags.
    :param verbose: If True, prints the results.
    """
    eval_steps = flags.max_eval_steps if flags.max_eval_steps is not None and flags.max_eval_steps > 0 else None
    eval_name = flags.validation_name if flags.validation_name != '' else None
    eval_checkpoint = flags.validation_checkpoint or None

    eval_results = estimator.evaluate(input_fn=input_fn,
                                      steps=eval_steps,
                                      checkpoint_path=eval_checkpoint,
                                      name=eval_name)
    if verbose:
        print('\nEvaluation results:\n\t%s\n' % eval_results)
    return eval_results


def get_best_validation_metrics(flags: argparse.Namespace, model_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Obtains the validation metrics from the specified model directory by creating a new
    Estimator instance and running it.
    :param flags: The command-line arguments.
    :param model_dir: The model directory.
    :return: The evaluation metrics.
    """
    if model_dir is None:
        model_dir = flags.best_model_dir
    if not os.path.exists(model_dir):
        tf.logging.warning('The directory for best models doesn\'t (yet) exist: %s', model_dir)
        return {}

    tf.logging.info('Determining best validation metrics from: %s', model_dir)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=build_run_config(flags),
        params=flags
    )

    return run_validation(estimator, flags, verbose=False)


def build_run_config(flags: argparse.Namespace) -> tf.estimator.RunConfig:
    """
    Builds a tf.estimator.RunConfig from the command-line arguments.
    :param flags: The command-line arguments.
    :return: The run configuration.
    """
    config = tf.ConfigProto()

    config.gpu_options.allow_growth = flags.gpu_growth
    if flags.intra_op_parallelism_threads >= 0:
        config.intra_op_parallelism_threads = flags.intra_op_parallelism_threads
    if flags.inter_op_parallelism_threads >= 0:
        config.inter_op_parallelism_threads = flags.inter_op_parallelism_threads
    if flags.xla:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    # Run configuration
    return tf.estimator.RunConfig(
        save_summary_steps=flags.save_summary_steps,
        session_config=config,
        keep_checkpoint_max=10)


def get_cli_args():
    args = get_project_parameters()

    # We have batch_size as a global default to training and validation batch sizes.
    # Here we joint them.
    if args.train_batch_size is None or args.train_batch_size <= 0:
        args.train_batch_size = args.batch_size
    if args.eval_batch_size is None or args.eval_batch_size <= 0:
        args.eval_batch_size = args.batch_size

    return args


if __name__ == '__main__':
    cli_args = get_cli_args()
    main(cli_args)
