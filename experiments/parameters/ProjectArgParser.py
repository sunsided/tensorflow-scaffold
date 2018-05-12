import argparse
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

        self.set_defaults(
            data_dir='dataset',
            validate=False,
            model_dir='model',
            train_epochs=1000,
            epochs_between_evals=1,
            batch_size=100,
            train_batch_size=None,   # default to global batch_size
            validation_batch_size=None  # default to global batch_size
        )
