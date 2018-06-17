# TensorFlow Project Scaffold

This project is meant to provide a starting point for new
TensorFlow projects. It showcases

- [`tf.estimator.Estimator`]-based training using custom
  `input_fn` and `model_fn` functions, using 
  standard [`tf.estimator.EstimatorSpec`] definitions.
  - Image files are read using [`tf.gfile.FastGFile`] for source-agnostic, lock-free file loading.
  - Efficient JPEG decoding using [`tf.image.decode_and_crop_jpeg`].
  - Usage of pretrained models using [`tensorflow_hub.Module`].
- [`tf.data.Dataset`] with `.list_files()` and `.from_generator()`
   examples.
  - Interleaved `TFRecord` input streams using [`tf.data.TFRecordDataset`] and 
    [`tf.contrib.data.parallel_interleave`].
  - GPU prefetching using [`tf.contrib.data.prefetch_to_device`].
- Automatic snapshotting of parameters with the best
  validation loss into a separate directory.

[`tf.estimator.Estimator`]: https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator
[`tf.estimator.EstimatorSpec`]: https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec
[`tf.gfile.FastGFile`]: https://www.tensorflow.org/api_docs/python/tf/gfile/FastGFile
[`tf.image.decode_and_crop_jpeg`]: https://www.tensorflow.org/api_docs/python/tf/image/decode_and_crop_jpeg
[`tensorflow_hub.Module`]: https://www.tensorflow.org/hub/
[`tf.data.TFRecordDataset`]: https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset
[`tf.data.Dataset`]: https://www.tensorflow.org/api_docs/python/tf/data/Dataset
[`tf.contrib.data.parallel_interleave`]: https://www.tensorflow.org/api_docs/python/tf/contrib/data/parallel_interleave
[`tf.contrib.data.prefetch_to_device`]: https://www.tensorflow.org/api_docs/python/tf/contrib/data/prefetch_to_device

Inspirations and sources:

- [Importing Data](https://www.tensorflow.org/programmers_guide/datasets)
- [Input Pipeline Performance Guide](https://www.tensorflow.org/versions/master/performance/datasets_performance    )
- [Preparing a large-scale image dataset with TensorFlow's TFRecord files](https://kwotsin.github.io/tech/2017/01/29/tfrecords.html)
- [Getting Text into Tensorflow with the Dataset API](https://medium.com/@TalPerry/getting-text-into-tensorflow-with-the-dataset-api-ffb832c8bec6)
- [How to write into and read from a TFRecords file in TensorFlow](http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html)
- [Use HParams and YAML to Better Manage Hyperparameters in Tensorflow](https://hanxiao.github.io/2017/12/21/Use-HParams-and-YAML-to-Better-Manage-Hyperparameters-in-Tensorflow/)
- [generator-tf](https://github.com/jrabary/generator-tf/)

## Structure of the project

- `project`: project modules such as networks, input pipelines, etc.
- `library`: scripts and boilerplate code

Two configuration files exist:

- `project.yaml`: Serialized command-line options
- `hyperparameters.yaml`: Model hyperparameters

Here's an example `hyperparameters.yaml`, with a default hyper-parameter
set (conveniently called `default`), and an additional set named `mobilenet`.
Here, the `mobilenet` set inherits from `default` and overwrites
only the default parameters with the newly defined ones.

```yaml
default: &DEFAULT
  # batch_size: 100
  # num_epoch: 1000
  # optimizer: Adam
  learning_rate: 1e-4
  dropout_rate: 0.5
  l2_regularization: 1e-8
  xentropy_label_smoothing: 0.
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1e-8

mobilenet:
  <<: *DEFAULT
  learning_rate: 1e-5
  fine_tuning: True
```

Likewise, the `project.yaml` contains serialized command-line
parameters:

```yaml
default: &DEFAULT
  train_batch_size: 32
  train_epochs: 1000
  epochs_between_evals: 100
  hyperparameter_file: hyperparameters.yaml
  hyperparameter_set: default
  model: latest
  model_dir: out/current/checkpoints
  best_model_dir: out/current/best

gtx1080ti:
  <<: *DEFAULT
  train_batch_size: 512

thinkpadx201t:
  <<: *DEFAULT
  train_batch_size: 10
  train_epochs: 10
  epochs_between_evals: 1
  random_seed: 0
```

By selecting a configuration set on startup using the `--config_set` command-line
option, best configurations can be stored and versioned easily.
Configuration provided on the command-line overrides values defined
in `project.yaml`, allowing for quick iteration.

## Run training

In order to run a training session (manually overriding configuration
from `project.yaml`), try

```bash
python run.py \
    --xla \
    --epochs_between_evals 1000 \
    --train_epochs 10000 \
    --learning_rate 0.0001 
```

## Prepare the dataset

In order to improve processing speed later on, the image files are
converted to `TFRecord` format first. For this, run

```bash
python convert_dataset.py \
    --dataset_dir dataset/train \
    --tfrecord_filename train \
    --tfrecord_dir dataset/train \
    --max_edge 384
python convert_dataset.py \
    --dataset_dir dataset/test \
    --tfrecord_filename test \
    --tfrecord_dir dataset/test \
    --max_edge 384
```

This example stores image data as JPEG encoded raw bytes and decodes
them on the fly in the input pipeline. While this leads to much smaller
TFRecord files compared to storing raw pixel values, it also creates
a (noticeable) latency. There's a tradeoff here.

## TensorFlow Hub

In order to use [TensorFlow Hub](https://github.com/tensorflow/hub), install it using e.g.

```bash
pip install tensorflow-hub
```

When initializing a Conda environment from `environment.yaml`, this is
already taken care of.

