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


class ImageModelParser(argparse.ArgumentParser):
    """Default parser for specification image specific behavior.
    Args:
      add_help: Create the '--help' flag. False if class instance is a parent.
      data_format: Create a flag to specify image axis convention.
    """

    def __init__(self, add_help=False, data_format=True):
        super(ImageModelParser, self).__init__(add_help=add_help)
        if data_format:
            self.add_argument(
                '--data_format', '-df',
                default=None,
                choices=['channels_first', 'channels_last'],
                help='A flag to override the data format used in the model. '
                     'channels_first provides a performance boost on GPU but is not '
                     'always compatible with CPU. If left unspecified, the data '
                     'format will be chosen automatically based on whether TensorFlow'
                     'was built for CPU or GPU.',
                metavar='<CF>'
            )
