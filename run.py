import argparse
import random
import tensorflow as tf
import numpy as np

from experiments.hooks import ExamplesPerSecondHook
from experiments.parameters import ProjectArgParser
from project.data.inputs import input_fn
from project.estimator import model_fn


def main(flags: argparse.Namespace):
    # Set TensorFlow logging verbosity
    tf.logging.set_verbosity(tf.logging.INFO if not flags.verbose else tf.logging.DEBUG)

    # Set random seed
    if flags.random_seed is not None:
        random.seed(flags.random_seed, version=2)
        np.random.seed(flags.random_seed)
        tf.set_random_seed(flags.random_seed)

    # TODO: Serialize model parameters into configuration file (check e.g. tf.contrib.training.HParams)

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
        tensors_to_log = ['learning_rate', 'loss']

        # Set up hook that outputs training logs every 100 steps.
        # TODO: Add profiler hooks support
        train_hooks = [
            tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100),
            ExamplesPerSecondHook(batch_size=flags.train_batch_size, every_n_steps=100)
        ]

        train_steps = flags.max_train_steps if flags.max_train_steps is not None and flags.max_train_steps >= 0 else None
        for _ in range(flags.train_epochs // flags.epochs_between_evals):
            estimator.train(input_fn=lambda: input_fn(flags, is_training=True),
                            hooks=train_hooks,
                            max_steps=train_steps)
            eval_results = estimator.evaluate(input_fn=lambda: input_fn(flags, is_training=False),
                                              steps=eval_steps)
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


def get_cli_args():
    # For hyperparameters with JSON export functionality,
    # we could use HParams. However, they do not fit in easily with argparse.
    # https://www.tensorflow.org/api_docs/python/tf/contrib/training/HParams
    parser = ProjectArgParser()
    args = parser.parse_args()

    if args.train_batch_size is None or args.train_batch_size <= 0:
        args.train_batch_size = args.batch_size

    if args.validation_batch_size is None or args.validation_batch_size <= 0:
        args.validation_batch_size = args.batch_size

    return args


if __name__ == '__main__':
    cli_args = get_cli_args()
    main(cli_args)