from typing import List, Union
import tensorflow as tf


def int64_feature(values: Union[int, List[int]]) -> tf.train.Feature:
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def int64_list_feature(values: List[int]) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(value: Union[bytes, str]) -> tf.train.Feature:
    if isinstance(value, str):
        value = value.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# https://kwotsin.github.io/tech/2017/01/29/tfrecords.html
# https://github.com/kwotsin/create_tfrecords/blob/master/dataset_utils.py
def image_to_tfexample(image_data: bytes, image_format: str, height: int, width: int, label_id: int) -> tf.train.Example:
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(label_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }))
