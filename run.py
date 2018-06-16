import argparse
import random
import tensorflow as tf
import numpy as np

from experiments.hooks import ExamplesPerSecondHook
from experiments.hooks.EvaluationCheckpointSaverHook import EvaluationCheckpointSaverHook
from experiments.parameters import get_project_parameters, YParams
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

    # Session configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = flags.gpu_growth
    if flags.intra_op_parallelism_threads >= 0:
        config.intra_op_parallelism_threads = flags.intra_op_parallelism_threads
    if flags.inter_op_parallelism_threads >= 0:
        config.inter_op_parallelism_threads = flags.inter_op_parallelism_threads
    if flags.xla:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    # Run configuration
    run_config = tf.estimator.RunConfig(save_summary_steps=200, session_config=config)

    # Load the hyperparameters
    hparams = YParams(flags.hyperparameter_file, flags.hyperparameter_set)
    # TODO: Replace parameters passed on the command line from flags

    # We now obtain the model and replace the parameter with the actual instance.
    model = model_builder(flags.model, hparams)
    flags.model = model

    # Create estimator that trains and evaluates the model
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=flags.model_dir,
        config=run_config,
        params=flags
    )

    eval_steps = flags.max_eval_steps if flags.max_eval_steps is not None and flags.max_eval_steps > 0 else None

    if not flags.validate:
        # The tensors to log during training
        tensors_to_log = ['learning_rate', 'loss', 'xentropy']

        tensor_values = {'loss': None}

        # Set up hook that outputs training logs every N steps.
        # TODO: Add profiler hooks support
        # TODO: Take snapshot whenever training accuracy increases!
        report_every_n_iter = 1000
        train_hooks = [
            tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=report_every_n_iter),
            ExamplesPerSecondHook(batch_size=flags.train_batch_size, every_n_steps=report_every_n_iter)
        ]
        eval_hooks = [
            EvaluationCheckpointSaverHook(checkpoint_dir='out-eval', tensor_values=tensor_values)
        ]

        train_steps = flags.max_train_steps if flags.max_train_steps is not None and flags.max_train_steps >= 0 else None
        for _ in range(flags.train_epochs // flags.epochs_between_evals):
            estimator.train(input_fn=lambda: input_fn(flags, is_training=True),
                            hooks=train_hooks,
                            max_steps=train_steps)
            eval_results = estimator.evaluate(input_fn=lambda: input_fn(flags, is_training=False),
                                              steps=eval_steps,
                                              hooks=eval_hooks)
            print('\nEvaluation results:\n\t%s\n' % eval_results)
            # TODO: Stop running when accuracy is below flags.stop_threshold
    else:
        eval_name = flags.validation_name if flags.validation_name != '' else None
        eval_checkpoint = flags.validation_checkpoint or None
        eval_results = estimator.evaluate(input_fn=lambda: input_fn(flags, is_training=False),
                                          steps=eval_steps,
                                          checkpoint_path=eval_checkpoint,
                                          name=eval_name)
        print('\nEvaluation results:\n\t%s\n' % eval_results)

    # TODO: Add exporting capability (frozen graph)
    # TODO: Add Experiments API


def get_cli_args():
    args = get_project_parameters()

    # We have batch_size as a global default to training and validation batch sizes.
    # Here we joint them.
    if args.train_batch_size is None or args.train_batch_size <= 0:
        args.train_batch_size = args.batch_size
    if args.validation_batch_size is None or args.validation_batch_size <= 0:
        args.validation_batch_size = args.batch_size

    return args


if __name__ == '__main__':
    cli_args = get_cli_args()
    main(cli_args)
