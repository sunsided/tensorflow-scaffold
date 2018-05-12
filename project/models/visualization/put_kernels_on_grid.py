import tensorflow as tf


# https://gist.github.com/kukuruza/03731dc494603ceab0c5
def put_kernels_on_grid(kernel: tf.Tensor, grid_Y: int, grid_X: int, pad=1, scope: str='kernels_to_grid'):
    """Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    """

    with tf.variable_scope(scope):
        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)

        kernel1 = (kernel - x_min) / (x_max - x_min)

        # pad X and Y
        x1 = tf.pad(kernel1, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

        # X and Y dimensions, w.r.t. padding
        Y = kernel1.get_shape()[0] + 2 * pad
        X = kernel1.get_shape()[1] + 2 * pad

        channels = kernel1.get_shape()[2]

        # put NumKernels to the 1st dimension
        x2 = tf.transpose(x1, (3, 0, 1, 2))
        # organize grid on Y axis
        x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))  # 3

        # switch X and Y axes
        x4 = tf.transpose(x3, (0, 2, 1, 3))
        # organize grid on X axis
        x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))  # 3

        # back to normal order (not combining with the next step for clarity)
        x6 = tf.transpose(x5, (2, 1, 3, 0))

        # to tf.image_summary order [batch_size, height, width, channels],
        #   where in this case batch_size == 1
        x7 = tf.transpose(x6, (3, 0, 1, 2))

        # scale to [0, 255] and convert to uint8
        return x7  # tf.image.convert_image_dtype(x7, dtype=tf.uint8)
