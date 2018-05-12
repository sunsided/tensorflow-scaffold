# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import tempfile
from tensorflow import test


class BaseParser(argparse.ArgumentParser):
    """Parser to contain flags which will be nearly universal across models.
    Args:
        add_help: Create the '--help' flag. False if class instance is a parent.
        data_dir: Create a flag for specifying the input data directory.
        model_dir: Create a flag for specifying the model file directory.
        train_epochs: Create a flag to specify the number of training epochs.
        epochs_between_evals: Create a flag to specify the frequency of testing.
        stop_threshold: Create a flag to specify a threshold accuracy or other
          eval metric which should trigger the end of training.
        batch_size: Create a flag to specify the global batch size.
        multi_gpu: Create a flag to allow the use of all available GPUs.
        num_gpu: Create a flag to specify the number of GPUs used.
    """

    def __init__(self, add_help=False, data_dir=True, model_dir=True,
                 train_epochs=True, epochs_between_evals=True,
                 stop_threshold=True, batch_size=True,
                 multi_gpu=False, num_gpu=True):
        super(BaseParser, self).__init__(add_help=add_help)

        if data_dir:
            self.add_argument(
                '--data_dir', '-dd', default=tempfile.tempdir,
                help='[default: %(default)s] The location of the input data.',
                metavar='<DD>',
            )

        if model_dir:
            self.add_argument(
                '--model_dir', '-md', default=tempfile.tempdir,
                help='[default: %(default)s] The location of the model checkpoint '
                     'files.',
                metavar='<MD>',
            )

        if train_epochs:
            self.add_argument(
                '--train_epochs', '-te', type=int, default=1,
                help='[default: %(default)s] The number of epochs used to train.',
                metavar='<TE>'
            )

        if epochs_between_evals:
            self.add_argument(
                '--epochs_between_evals', '-ebe', type=int, default=1,
                help='[default: %(default)s] The number of training epochs to run '
                     'between evaluations.',
                metavar='<EBE>'
            )

        if stop_threshold:
            self.add_argument(
                '--stop_threshold', '-st', type=float, default=None,
                help='[default: %(default)s] If passed, training will stop at '
                     'the earlier of train_epochs and when the evaluation metric is '
                     'greater than or equal to stop_threshold.',
                metavar='<ST>'
            )

        if batch_size:
            self.add_argument(
                '--batch_size', '-bs', type=int, default=32,
                help='[default: %(default)s] Global batch size for training and '
                     'evaluation.',
                metavar='<BS>'
            )

        assert not (multi_gpu and num_gpu)

        if multi_gpu:
            self.add_argument(
                '--multi_gpu', action='store_true',
                help='If set, run across all available GPUs.'
            )

        if num_gpu:
            self.add_argument(
                '--num_gpus', '-ng',
                type=int,
                default=1 if test.is_built_with_cuda() else 0,
                help='[default: %(default)s] How many GPUs to use with the '
                     'DistributionStrategies API. The default is 1 if TensorFlow was'
                     'built with CUDA, and 0 otherwise.',
                metavar='<NG>'
            )
