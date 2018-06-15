from typing import Optional
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib.training import HParams
from .Model import Model, HyperParameters


class HubModel(Model):
    def __init__(self, params: Optional[HyperParameters] = None):
        super().__init__(params)

    def default_hparams(self):
        return HParams(learning_rate=1e-5,
                       dropout_rate=0.5,
                       l2_regularization=1e-5,
                       fine_tuning=False)

    def build(self, features: tf.Tensor, mode: str):
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        graph = tf.get_default_graph()

        # Hub image models are expected to pass their input as 0..1 (i.e. unnormalized).
        # See https://www.tensorflow.org/hub/common_signatures/images for more information.

        # Note that even though we specify 'train' during training, the variables are not automatically
        # added to the global trainable variables (see tf.trainable_variables() unless trainable
        # is set to True.
        # See https://github.com/tensorflow/hub/blob/master/docs/fine_tuning.md
        module_tags = {'train'} if is_training else None
        fine_tuning = is_training and self.params.fine_tuning

        # https://www.tensorflow.org/hub/modules/google/imagenet/mobilenet_v2_035_224/feature_vector/1
        module = hub.Module('https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/1',
                            name='module', tags=module_tags, trainable=fine_tuning)

        with tf.variable_scope('model'):
            # We require a specific input image size for this.
            # We should actually use these values directly in the input pipeline.
            height, width = hub.get_expected_image_size(module)
            assert (height == 224) and (width == 224), 'Module image dimension mismatch.'

            # The feature_vector variant of the model already provides the standard signature for
            # extracting image feature vectors; it does not need to be specified explicitly. However,
            # in order to show how this changes the network output with as_dict=True, it is used here.
            module_out = module(features, signature='image_feature_vector', as_dict=True)
            net = module_out['default']

            with tf.variable_scope('infer'):
                name_scope = graph.get_name_scope()

                # The MobileNet feature vector is the output of an average pooling operation,
                # so we activate it for further processing.
                net = tf.nn.selu(net, 'activate_features')
                net = tf.layers.dropout(net, rate=self.params.dropout_rate, name='dropout', training=is_training)

                regularizers = tf.contrib.layers.l2_regularizer(scale=self.params.l2_regularization)

                # To stay 2D convolutional, we need to re-add the missing dimensions.
                net = tf.reshape(net, shape=(-1, 1, 1, net.shape[1].value), name='reshape')
                net = tf.layers.conv2d(net, filters=256, kernel_size=1, strides=1, padding='valid',
                                       data_format='channels_last',
                                       activation=tf.nn.selu,
                                       kernel_initializer=tf.initializers.orthogonal,
                                       kernel_regularizer=regularizers,
                                       name='conv_1')

                # Add weights to TensorBoard
                grid = tf.get_default_graph().get_tensor_by_name(f'{name_scope}/conv_1/kernel:0')
                tf.summary.histogram('infer/conv1/weights', grid)

                net = tf.layers.dropout(net, rate=self.params.dropout_rate, name='dropout', training=is_training)
                net = tf.layers.conv2d(net, filters=1, kernel_size=1, strides=1, padding='valid',
                                       data_format='channels_last',
                                       activation=None,
                                       kernel_initializer=tf.initializers.orthogonal,
                                       kernel_regularizer=regularizers,
                                       name='conv_2')

                # Add weights to TensorBoard
                grid = tf.get_default_graph().get_tensor_by_name(f'{name_scope}/conv_2/kernel:0')
                tf.summary.histogram('infer/conv2/weights', grid)

                logits = tf.squeeze(tf.layers.flatten(net, name='flatten'), axis=1, name='logits')
                prediction = tf.nn.sigmoid(logits, name='prediction')

        return {'logits': logits,
                'predictions': prediction}
