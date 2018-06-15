from collections import OrderedDict
from typing import Optional
from .Model import HyperParameters
from .SimpleModel import SimpleModel
from .HubModel import HubModel


MODELS = OrderedDict([
    ('simple_model', SimpleModel),
    ('hub_model', HubModel)
])


def model_builder(model_name: Optional[str] = 'latest', params: Optional[HyperParameters] = None) -> Model:
    """
    Builds a model by name, or builds the latest variant.

    :param model_name: The name of the model to build.
    :param params: The hyper-parameters to pass to the model.
    :return: The model.
    """
    lower_model_name = model_name.lower() if model_name is not None else 'latest'
    if lower_model_name == 'latest':
        lower_model_name = next(reversed(MODELS))

    model = MODELS.get(lower_model_name, None)
    if model is not None:
        return model(params)
    else:
        raise ValueError("Unknown model name {}".format(model_name))
