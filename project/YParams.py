from tensorflow.contrib.training import HParams
from ruamel.yaml import YAML
from .HyperparameterError import HyperparameterFileError


class YParams(HParams):
    """
    Support for YAML formatted hyper-parameter files.
    """

    def __init__(self, yaml_file, config_name: str = 'default'):
        """
        Initializes the YAML-formatted HParams parser.
        :param yaml_file: The YAML hyper-parameter file.
        :param config_name: The configuration to load.
        """
        super().__init__()
        with open(yaml_file) as fp:
            yaml = YAML().load(fp)

        if config_name not in yaml:
            raise HyperparameterFileError(
                f"The configuration '{config_name}' does not exist in parameter file {yaml_file}.", yaml_file)
        for k, v in yaml[config_name].items():
            self.add_hparam(k, v)
