from argparse import Namespace
from collections import OrderedDict
from typing import Optional, Callable
import tensorflow as tf
from .simple_model import simple_model
from .hub_model import hub_model


MODELS = OrderedDict([
    ('simple_model', simple_model),
    ('hub_model', hub_model)
])


def model_builder(model_name: Optional[str]) -> Callable[[tf.Tensor, str, Namespace], tf.Tensor]:
    lower_model_name = model_name.lower() if model_name is not None else 'latest'
    if lower_model_name == 'latest':
        lower_model_name = next(reversed(MODELS))

    model = MODELS.get(lower_model_name, None)
    if model is not None:
        return model
    else:
        raise ValueError("Unknown model name {}".format(model_name))
