import tensorflow as tf


def default_params():
    return tf.contrib.training.HParams(
        # Model options
        model={
            # type of the model to create
            'type': 'MY_MODEL',
        }
    )
