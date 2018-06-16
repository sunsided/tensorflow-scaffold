import argparse
import multiprocessing
import tempfile
from .tf_official import BaseParser, ImageModelParser, ExportParser, PerformanceParser


class ProjectArgParser(argparse.ArgumentParser):
    """Argument parser for running orientation detection model."""

    def __init__(self):
        super().__init__(parents=[
            BaseParser(add_help=False, multi_gpu=False, num_gpu=False,
                       stop_threshold=False,
                       train_epochs=True, epochs_between_evals=True),
            PerformanceParser(num_parallel_calls=True, use_synthetic_data=False, max_train_steps=True, intra_op=True),
            ImageModelParser(),
            ExportParser(),
        ], add_help=True)

        self.add_argument(
            '--model', '-m', default='latest',
            help='[default: %(default)s] The model to use.',
            metavar='<M>',
        )

        self.add_argument(
            '--hyperparameters_file', '-hpf', default='hyperparameters.yaml',
            help='[default: %(default)s] The hyperparameter file to use.',
            metavar='<HPF>',
        )

        self.add_argument(
            '--hyperparameter_set', '-hps', default='default',
            help='[default: %(default)s] The hyperparameter set to use.',
            metavar='<HPS>',
        )

        self.add_argument(
            '--validation_dir', '-vd', default=tempfile.tempdir,
            help='[default: %(default)s] The location of the validation data.',
            metavar='<VD>',
        )

        self.add_argument(
            '--validation_checkpoint', '-vc', type=str, default=None,
            help='[default: %(default)s] The location of the validation checkpoint to use. '
                 'This option is ignored when training.',
            metavar='<VC>',
        )

        self.add_argument(
            '--validation_name', '-vn', type=str, default='',
            help='[default: %(default)s] The name of the evaluation when multiple data sets '
                 'are used (e.g. "validation", "test", ...).',
            metavar='<VN>',
        )

        self.add_argument(
            '--train_batch_size', '-tbs', type=int, default=None,
            help='[default: %(default)s] Batch size for training.',
            metavar='<BS>'
        )

        self.add_argument(
            '--validation_batch_size', '-vbs', type=int, default=None,
            help='[default: %(default)s] Batch size for evaluation.',
            metavar='<BS>'
        )

        self.add_argument(
            '--validate', action='store_true',
            help='If set, run only validation instead of training.'
        )

        self.add_argument(
            '--gpu_growth', action='store_true',
            help='If set, allocate only minimal GPU memory and reallocate if needed.'
        )

        self.add_argument(
            '--xla', action='store_true',
            help='If set, enable Accelerated Linear Algebra (XLA).'
        )

        self.add_argument(
            '--verbose', action='store_true',
            help='If set, verbose log output is generated.'
        )

        self.add_argument(
            '--random_seed', '-rs', type=int, default=None,
            help='[default: %(default)s] The seed used for random initialization.',
            metavar='<RS>'
        )

        self.add_argument(
            '--max_eval_steps', '-mes', type=int, default=None,
            help='[default: %(default)s] The model will stop evaluating if the '
                 'this many batches were processed or the input is exhausted.',
            metavar='<MTS>'
        )

        self.add_argument(
            '--learning_rate', '-lr', type=float, default=1e-4,
            help='[default: %(default)s] The learning rate.',
            metavar='<LR>'
        )

        self.add_argument(
            '--parallel_interleave_sources', '-pis', type=int, default=2,
            help='[default: %(default)s] The number of input sources that are '
                 'parsed in parallel and interleaved.',
            metavar='<PIS>'
        )

        self.add_argument(
            '--num_parallel_reads', '-npr', type=int, default=1,
            help='[default: %(default)s] The number of input sources that are '
                 'read in parallel.',
            metavar='<NPR>'
        )

        self.add_argument(
            '--prefetch_examples', '-pre', type=int, default=1024,
            help='[default: %(default)s] The number of examples to prefetch.',
            metavar='<PRE>'
        )

        self.add_argument(
            '--prefetch_batches', '-pb', type=int, default=100,
            help='[default: %(default)s] The number of batches to prefetch.',
            metavar='<PRB>'
        )

        self.add_argument(
            '--num_parallel_batches', '-npb', type=int, default=8,
            help='[default: %(default)s] The number of batches to prepare in parallel.',
            metavar='<NPB>'
        )

        self.add_argument(
            '--prefetch_to_device', '-ptd', type=str, default=None,
            help='[default: %(default)s] The device to prefetch to, e.g. \'/gpu:0\'.',
            metavar='<DEVICE>'
        )

        self.set_defaults(
            data_dir='dataset',
            validate=False,
            model_dir='out',
            train_epochs=1000,
            inter_op_parallelism_threads=multiprocessing.cpu_count(),
            intra_op_parallelism_threads=multiprocessing.cpu_count(),
            num_parallel_calls=multiprocessing.cpu_count(),
            epochs_between_evals=1,
            batch_size=256,
            train_batch_size=None,   # default to global batch_size
            validation_batch_size=None  # default to global batch_size
        )
