import sys
from .ProjectArgParser import ProjectArgParser
from .YParams import YParams


def _merge_with_config_file(args):
    args_dict = vars(args)

    # Load the application configuration and merge with CLI arguments
    config = YParams(args.config_file, args.config_set)
    config = dict(config.values())
    errors = []
    for k, v in config.items():
        if k not in args_dict:
            errors.append(k)
            continue
        args.__setattr__(k, v)
    if len(errors) > 0:
        print(f'Invalid configuration options found in file {args.config_file}:', file=sys.stderr)
        print(', '.join(errors), file=sys.stderr)
        sys.exit(1)
    return config


def get_project_parameters():
    # We first parse the command-line arguments in order to determine the
    # location of the configuration file (project.yaml, by default).
    parser = ProjectArgParser()
    args = parser.parse_args()

    # We now load the application configuration and merge with CLI arguments.
    # We apply these as new command line defaults and re-parse,
    # thus overwriting the config file with command-line arguments.
    parser.set_defaults(**_merge_with_config_file(args))
    return parser.parse_args()
