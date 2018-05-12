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


class ExportParser(argparse.ArgumentParser):
    """Parsing options for exporting saved models or other graph definitions.
    This is a separate parser for now, but should be made part of BaseParser
    once all models are brought up to speed.
    Args:
      add_help: Create the '--help' flag. False if class instance is a parent.
      export_dir: Create a flag to specify where a SavedModel should be exported.
    """

    def __init__(self, add_help=False, export_dir=True):
        super(ExportParser, self).__init__(add_help=add_help)
        if export_dir:
            self.add_argument(
                '--export_dir', '-ed',
                help='[default: %(default)s] If set, a SavedModel serialization of '
                     'the model will be exported to this directory at the end of '
                     'training. See the README for more details and relevant links.',
                metavar='<ED>'
            )
