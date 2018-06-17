"""Estimator inputs definitions. """
import glob
import os
import random
from argparse import Namespace
from typing import Tuple, List
import tensorflow as tf
from .data.image import parse_tfexample, read_image, augment_image


def input_fn(params: Namespace, mode: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """Load data here and return features and labels tensors."""

    # TODO: Add support for flags.use_synthetic_data

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    data_prefix = 'train' if is_training else 'test'
    data_dir = os.path.join(params.data_dir, data_prefix)

    # dataset = tf.data.Dataset().from_generator(lambda: get_images(data_dir),
    #                                            output_types=(tf.string, tf.int32),
    #                                            output_shapes=(None, None))

    # https://www.tensorflow.org/api_docs/python/tf/contrib/data/parallel_interleave
    # https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset
    file_pattern = os.path.join(data_dir, '*.tfrecord')

    if params.parallel_interleave_sources > 1:
        tfrecords = tf.data.Dataset.list_files(file_pattern)
        tfrecords = tfrecords.cache()
        dataset = tfrecords.apply(
            tf.contrib.data.parallel_interleave(
                map_func=tf.data.TFRecordDataset,
                sloppy=True,
                cycle_length=params.num_parallel_calls))
    else:
        dataset = tf.data.TFRecordDataset(filenames=glob.glob(file_pattern),
                                          num_parallel_reads=max(1, params.num_parallel_reads))

    def decode_example(example: tf.train.Example) -> Tuple[tf.Tensor, tf.Tensor]:
        feature, label = parse_tfexample(example)
        feature = read_image(feature, augment=is_training)
        return feature, label

    dataset = dataset.map(decode_example, num_parallel_calls=max(1, params.num_parallel_calls))

    if params.prefetch_examples > 0:
        dataset = dataset.prefetch(params.prefetch_examples)

    # Fused shuffle / repeat.
    dataset = dataset.apply(
        tf.contrib.data.shuffle_and_repeat(
            buffer_size=params.batch_size,
            count=params.epochs_between_evals))

    if is_training:
        # During training runs, we augment the input data.
        def map_data(feature: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            feature = augment_image(feature, fast_mode=True)
            return feature, label
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                map_func=map_data,
                batch_size=params.train_batch_size,
                num_parallel_batches=max(1, params.num_parallel_batches),
                drop_remainder=False))
    else:
        # During validation runs, we don't need to augment the input data.
        dataset = dataset.batch(batch_size=params.eval_batch_size)

    if params.prefetch_batches >= 1:
        dataset = dataset.prefetch(params.prefetch_batches)

    if params.prefetch_to_device:
        print(f'Prefetching to {params.prefetch_to_device} ...')
        dataset = dataset.apply(
            tf.contrib.data.prefetch_to_device(device=params.prefetch_to_device))

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


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
