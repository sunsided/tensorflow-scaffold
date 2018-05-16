"""Estimator inputs definitions. """
import glob
import os
import random
from argparse import Namespace
from typing import Tuple, List
import tensorflow as tf


def _resample_class(items: List[str], n: int) -> List[str]:
    """
    Resamples the items in the list to have exactly N items.
    :param items: The items to resample.
    :param n: The number of items after resampling.
    :return: The resampled list.
    """
    random.shuffle(items)
    difference = n - len(items)
    if difference == 0:
        return items
    elif difference < 0:
        return items[:n]
    else:
        extra = random.sample(items, difference)
        return items + extra


# The following function is meant to be used with Dataset.from_generator()
def get_images(data_dir: str) -> Tuple[str, float]:
    """
    Uniformly samples images from the dataset directories.

    :return: A tuple of file content, image orientation and image quality. An orientation of 4 encodes "undecidable".
    """
    positive = glob.glob(os.path.join(data_dir, 'hot_dog', '*.jpg'))
    negative = glob.glob(os.path.join(data_dir, 'not_hot_dog', '*.jpg'))
    assert len(positive) > 0
    assert len(negative) > 0

    # Get the maximum number of items over all classes
    max_items = max(len(positive), len(negative))

    # Resample the data to have the same number of items.
    positive = _resample_class(positive, max_items)
    negative = _resample_class(negative, max_items)

    # We're assuming there are much more images in the regular_images list
    # than in the others, which is why we are iterating over that one.
    while True:
        if len(positive) == 0:
            assert len(negative) == 0
            break

        yield os.path.abspath(positive.pop()), 1
        yield os.path.abspath(negative.pop()), 0


def read_image(image_data: tf.Tensor, label: tf.Tensor, augment: bool, image_size: Tuple[int, int]=(224, 224),
               data_is_path: bool=False) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Decodes images from file contents, random flips, resizes and normalizes them.

    :param image_data: The JPEG bytes to decode.
    :param data_is_path: If set, the data is treated as a path to an image file. If unset, data is considered raw bytes.
    :param augment: Is set, images are augmented by random cropping.
    :param image_size: The image size to resize the images to.
    :param label: The labels to pass through.
    :return: A tuple of the decoded image, the orientation and quality.
    """

    if data_is_path:
        image_data = tf.gfile.FastGFile(image_data, 'rb').read()

    if augment:
        with tf.variable_scope('read_image_cropped'):
            # Extract image shape from raw JPEG image buffer.
            image_shape = tf.image.extract_jpeg_shape(image_data)

            # Get a crop window with distorted bounding box.
            full_image = tf.constant([[0, 0, 1., 1.]], dtype=tf.float32, shape=[1, 1, 4])
            sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(image_shape,
                                                                                   bounding_boxes=full_image,
                                                                                   use_image_if_no_bounding_boxes=True,
                                                                                   aspect_ratio_range=[.5, 1.5],
                                                                                   area_range=[.5, 1.])  # TODO: Extract magic numbers
            bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

            # Decode and crop image.
            offset_y, offset_x, _ = tf.unstack(bbox_begin)
            target_height, target_width, _ = tf.unstack(bbox_size)
            crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
            image = tf.image.decode_and_crop_jpeg(image_data, crop_window, channels=3,
                                                  try_recover_truncated=True, acceptable_fraction=.8,
                                                  name='decode_image')
    else:
        with tf.variable_scope('read_image'):
            # NOTE: decode_image not return the image's shape, however decode_jpeg also decodes PNG files.
            #       https://github.com/tensorflow/tensorflow/issues/9356
            image = tf.image.decode_jpeg(image_data, channels=3,
                                         try_recover_truncated=True, acceptable_fraction=.8,
                                         name='decode_image')

    with tf.variable_scope('resize_and_normalize'):
        # We need to get the image to a known size.
        images = tf.expand_dims(image, axis=0, name='expand_image_dims')
        images = tf.image.resize_images(images, size=image_size, method=tf.image.ResizeMethod.BICUBIC)  # TODO: Do in batch?

        # Note: tf.squeeze implicitly converts to tf.float32. For convert_to_image_type we
        # expect uint8 as otherwise the values won't be scaled.
        image = tf.cast(tf.squeeze(images, axis=0, name='squeeze_image_dims'), dtype=tf.uint8)

        # Convert to floating-point.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32, name='convert_image_dtype')

    return image, label


def parse_tfrecords(ex: tf.train.Example):
    # This function parses a TFRecord file for Examples. In order to use
    # SequenceExamples, see https://www.tensorflow.org/api_docs/python/tf/parse_single_sequence_example
    features = {
        'image/class/label': tf.FixedLenFeature([], dtype=tf.int64),
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string),
        # 'image/format': tf.FixedLenFeature([], dtype=tf.string),
        # 'image/height': tf.FixedLenFeature([], dtype=tf.int64),
        # 'image/width': tf.FixedLenFeature([], dtype=tf.int64),
    }
    parsed = tf.parse_single_example(ex, features)
    return parsed['image/encoded'], parsed['image/class/label']


def input_fn(flags: Namespace, is_training: bool):
    """Load data here and return features and labels tensors."""

    data_prefix = 'train' if is_training else 'test'
    data_dir = os.path.join(flags.data_dir, data_prefix)

    # dataset = tf.data.Dataset().from_generator(lambda: get_images(data_dir),
    #                                            output_types=(tf.string, tf.int32),
    #                                            output_shapes=(None, None))

    tfrecords = glob.glob(os.path.join(data_dir, '*.tfrecord'))
    dataset = tf.data.TFRecordDataset(tfrecords)
    dataset = dataset.map(parse_tfrecords, num_parallel_calls=flags.num_parallel_calls)

    dataset = dataset.map(lambda x, y: read_image(x, y, augment=True), flags.num_parallel_calls)  # TODO: Support data_format (channels_first, ...), extract image_size
    dataset = dataset.prefetch(1000)  # TODO: Extract magic number
    dataset = dataset.batch(flags.batch_size)
    dataset = dataset.prefetch(100)   # TODO: Extract magic number

    dataset = dataset.repeat(flags.epochs_between_evals)

    # TODO: Add random rotation and flipping

    if tf.test.is_built_with_cuda():
        print('Prefetching to GPU ...')
        prefetch_op = tf.contrib.data.prefetch_to_device(device="/gpu:0")
        dataset = dataset.apply(prefetch_op)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels
