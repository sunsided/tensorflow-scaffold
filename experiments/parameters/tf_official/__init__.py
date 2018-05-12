# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Collection of parsers which are shared among the official models.
The parsers in this module are intended to be used as parents to all arg
parsers in official models. For instance, one might define a new class:
class ExampleParser(argparse.ArgumentParser):
  def __init__(self):
    super(ExampleParser, self).__init__(parents=[
      arg_parsers.LocationParser(data_dir=True, model_dir=True),
      arg_parsers.DummyParser(use_synthetic_data=True),
    ])
    self.add_argument(
      "--application_specific_arg", "-asa", type=int, default=123,
      help="[default: %(default)s] This arg is application specific.",
      metavar="<ASA>"
    )
Notes about add_argument():
    Argparse will automatically template in default values in help messages if
  the "%(default)s" string appears in the message. Using the example above:
    parser = ExampleParser()
    parser.set_defaults(application_specific_arg=3141592)
    parser.parse_args(["-h"])
    When the help text is generated, it will display 3141592 to the user. (Even
  though the default was 123 when the flag was created.)
    The metavar variable determines how the flag will appear in help text. If
  not specified, the convention is to use name.upper(). Thus rather than:
    --app_specific_arg APP_SPECIFIC_ARG, -asa APP_SPECIFIC_ARG
  if metavar="<ASA>" is set, the user sees:
    --app_specific_arg <ASA>, -asa <ASA>
"""


from .BaseParser import BaseParser
from .ExportParser import ExportParser
from .ImageModelParser import ImageModelParser
from .PerformanceParser import PerformanceParser
