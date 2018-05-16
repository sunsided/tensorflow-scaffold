# TensorFlow Project Scaffold

This project is meant to provide a starting point for new
TensorFlow projects.

Inspirations and sources:

- [Preparing a large-scale image dataset with TensorFlow's TFRecord files](https://kwotsin.github.io/tech/2017/01/29/tfrecords.html)
- [Getting Text into Tensorflow with the Dataset API](https://medium.com/@TalPerry/getting-text-into-tensorflow-with-the-dataset-api-ffb832c8bec6)
- [How to write into and read from a TFRecords file in TensorFlow](http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html)
- [generator-tf](https://github.com/jrabary/generator-tf/)

## Structure of the project

- `project`: project modules
- `experiments`: experiment scripts
- `configs`: experiments configuration files

## Prepare the dataset

In order to improve processing speed later on, the image files are
converted to `TFRecord` format first. For this, run

```bash
python convert_dataset.py --dataset_dir dataset/train --tfrecord_filename train --tfrecord_dir dataset/train
python convert_dataset.py --dataset_dir dataset/test --tfrecord_filename test --tfrecord_dir dataset/test
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

