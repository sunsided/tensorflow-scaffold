from argparse import Namespace
from typing import Dict
import tensorflow as tf
import tensorflow_hub as hub

from .visualization import put_kernels_on_grid


def hub_model(features: tf.Tensor, mode: str, params: Namespace) -> Dict[str, tf.Tensor]:
    """ Base model definitions """
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.variable_scope('model'):

        # https://www.tensorflow.org/hub/modules/google/imagenet/mobilenet_v2_035_224/feature_vector/1
        # Note that even though we specify 'train' during training, the variables are not automatically
        # added to the global trainable variables (see tf.trainable_variables()=.
        module_tags = {'train'} if is_training else None
        module = hub.Module('https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/1',
                            trainable=False, name='module',
                            tags=module_tags)

        # We require a specific input image size for this.
        height, width = hub.get_expected_image_size(module)
        assert (height == 224) and (width == 224), 'Module image dimension mismatch.'

        # The feature_vector variant of the model already provides the standard signature for
        # extracting image feature vectors; it does not need to be specified explicitly. However,
        # in order to show how this changes the network output with as_dict=True, it is used here.
        module_out = module(features, signature='image_feature_vector', as_dict=True)
        net = module_out['default']

        with tf.variable_scope('infer'):
            # To stay 2D convolutional, we need to re-add the missing dimensions.
            net = tf.reshape(net, shape=(-1, 1, 1, net.shape[1].value), name='reshape')
            net = tf.layers.conv2d(net, filters=256, kernel_size=1, strides=1, padding='valid',
                                   data_format='channels_last',
                                   activation=tf.nn.selu,
                                   kernel_initializer=tf.initializers.orthogonal,
                                   name='conv_1')

            # TODO: Hyperparameter: Dropout
            net = tf.layers.dropout(net, rate=0.5, name='dropout', training=is_training)

            net = tf.layers.conv2d(net, filters=1, kernel_size=1, strides=1, padding='valid',
                                   data_format='channels_last',
                                   activation=None,
                                   kernel_initializer=tf.initializers.orthogonal,
                                   name='conv_2')

            logits = tf.squeeze(tf.layers.flatten(net, name='flatten'), axis=1, name='logits')
            prediction = tf.nn.sigmoid(logits, name='prediction')

        grid = tf.get_default_graph().get_tensor_by_name('model/infer/conv_1/kernel:0')
        tf.summary.histogram('infer/conv1/weights', grid)

        grid = tf.get_default_graph().get_tensor_by_name('model/infer/conv_2/kernel:0')
        tf.summary.histogram('infer/conv2/weights', grid)

    return {
        'logits': logits,
        'predictions': prediction
    }
