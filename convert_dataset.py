import glob
import os
import random
import sys
import math
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from experiments.preparation.tfrecord import image_to_tfexample
from experiments.preparation.image import get_image_bytes


flags = tf.app.flags
flags.DEFINE_string('dataset_dir', None, 'Your dataset directory.')
flags.DEFINE_string('tfrecord_filename', None, 'The output file to create.')
flags.DEFINE_string('tfrecord_dir', '.', 'The output directory to use.')
flags.DEFINE_integer('num_shards', 2, 'Number of shards to split the TFRecord files into')
flags.DEFINE_integer('random_seed', None, 'Random seed to use for repeatability.')
flags.DEFINE_integer('max_edge', 512, 'The maximum edge length for an image.')

FLAGS = flags.FLAGS


def get_dataset_filename(dataset_dir: str, tfrecord_filename: str, shard_id: int, num_shards: int) -> str:
    output_filename = '%s_%05d-of-%05d.tfrecord' % (tfrecord_filename, shard_id, num_shards)
    return os.path.join(dataset_dir, output_filename)


def main(_):
    if not FLAGS.dataset_dir:
        sys.exit('No dataset directory was specified.')

    if not FLAGS.tfrecord_filename:
        sys.exit('No TFRecord file name was specified.')

    image_filenames = glob.glob(os.path.join(FLAGS.dataset_dir, '**', '*.jpg'), recursive=True)
    num_per_shard = int(math.ceil(len(image_filenames) / float(FLAGS.num_shards)))

    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    random.shuffle(image_filenames)

    progress = tqdm(desc='Processing', total=len(image_filenames))
    for shard_id in range(FLAGS.num_shards):
        output_filename = get_dataset_filename(dataset_dir=FLAGS.tfrecord_dir,
                                               tfrecord_filename=FLAGS.tfrecord_filename,
                                               shard_id=shard_id, num_shards=FLAGS.num_shards)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, len(image_filenames))
            for i in range(start_ndx, end_ndx):
                progress.update(1)
                progress.set_postfix(shard=f'{shard_id+1}/{FLAGS.num_shards}')

                image = get_image_bytes(image_filenames[i], max_width=FLAGS.max_edge, max_height=FLAGS.max_edge)

                # TODO: Create an index file that maps inputs (or directories) to classes.
                label_id = 0 if 'not_hot_dog' in image_filenames[i] else 1

                example = image_to_tfexample(image_data=image.bytes,
                                             image_format=image.format,
                                             height=image.height,
                                             width=image.width,
                                             label_id=label_id)

                tfrecord_writer.write(example.SerializeToString())


if __name__ == '__main__':
    tf.app.run()
