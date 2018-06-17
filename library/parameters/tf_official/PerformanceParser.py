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
import tensorflow as tf
import multiprocessing as mp


# Map string to (TensorFlow dtype, default loss scale)
DTYPE_MAP = {
    'fp16': (tf.float16, 128),
    'fp32': (tf.float32, 1),
}


class PerformanceParser(argparse.ArgumentParser):
    """Default parser for specifying performance tuning arguments.
    Args:
      add_help: Create the '--help' flag. False if class instance is a parent.
      num_parallel_calls: Create a flag to specify parallelism of data loading.
      inter_op: Create a flag to allow specification of inter-op threads.
      intra_op: Create a flag to allow specification of intra-op threads.
    """

    def __init__(self, add_help=False, num_parallel_calls=True, inter_op=True,
                 intra_op=True, use_synthetic_data=True, max_train_steps=True,
                 dtype=True):
        super(PerformanceParser, self).__init__(add_help=add_help)

        if num_parallel_calls:
            self.add_argument(
                '--num_parallel_calls', '-npc',
                type=int, default=mp.cpu_count(),
                help='[default: %(default)s] The number of records that are '
                     'processed in parallel  during input processing. This can be '
                     'optimized per data set but for generally homogeneous data '
                     'sets, should be approximately the number of available CPU '
                     'cores.',
                metavar='<NPC>'
            )

        if inter_op:
            self.add_argument(
                '--inter_op_parallelism_threads', '-inter',
                type=int, default=0,
                help='[default: %(default)s Number of inter_op_parallelism_threads '
                     'to use for CPU. See TensorFlow config.proto for details.',
                metavar='<INTER>'
            )

        if intra_op:
            self.add_argument(
                '--intra_op_parallelism_threads', '-intra',
                type=int, default=0,
                help='[default: %(default)s Number of intra_op_parallelism_threads '
                     'to use for CPU. See TensorFlow config.proto for details.',
                metavar='<INTRA>'
            )

        if use_synthetic_data:
            self.add_argument(
                '--use_synthetic_data', '-synth',
                action='store_true',
                help='If set, use fake data (zeroes) instead of a real dataset. '
                     'This mode is useful for performance debugging, as it removes '
                     'input processing steps, but will not learn anything.'
            )

        if max_train_steps:
            self.add_argument(
                '--max_train_steps', '-mts', type=int, default=None,
                help='[default: %(default)s] The model will stop training if the '
                     'global_step reaches this value. If not set, training will run'
                     'until the specified number of epochs have run as usual. It is'
                     'generally recommended to set --train_epochs=1 when using this'
                     'flag.',
                metavar='<MTS>'
            )

        if dtype:
            self.add_argument(
                '--dtype', '-dt',
                default='fp32',
                choices=list(DTYPE_MAP.keys()),
                help='[default: %(default)s] {%(choices)s} The TensorFlow data type '
                     'used for calculations. Variables may be cast to a higher'
                     'precision on a case-by-case basis for numerical stability.',
                metavar='<DT>'
            )

            self.add_argument(
                '--loss_scale', '-ls',
                type=int,
                help='[default: %(default)s] The amount to scale the loss by when '
                     'the model is run. Before gradients are computed, the loss is '
                     'multiplied by the loss scale, making all gradients loss_scale '
                     'times larger. To adjust for this, gradients are divided by the '
                     'loss scale before being applied to variables. This is '
                     'mathematically equivalent to training without a loss scale, '
                     'but the loss scale helps avoid some intermediate gradients '
                     'from underflowing to zero. If not provided the default for '
                     'fp16 is 128 and 1 for all other dtypes.',
            )
