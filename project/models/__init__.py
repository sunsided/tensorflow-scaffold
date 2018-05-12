from project.models.my_model import my_model


def model_builder(model_name):
    if model_name == 'MY_MODEL':
        return my_model
    else:
        raise ValueError("Unknown model name {}".format(model_name))
