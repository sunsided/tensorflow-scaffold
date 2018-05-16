# TensorFlow Project Scaffold

This project is meant to provide a starting point for new
TensorFlow projects.

Inspirations and sources:

- [Preparing a large-scale image dataset with TensorFlow's TFRecord files](https://kwotsin.github.io/tech/2017/01/29/tfrecords.html)
- [generator-tf](https://github.com/jrabary/generator-tf/)

## Structure of the project

- `project`: project modules
- `experiments`: experiment scripts
- `configs`: experiments configuration files

## Prepare the dataset

In order to improve processing speed later on, the image files are
converted to `TFRecord` format first. For this, run

```bash
python convert_dataset.py --dataset_dir dataset/train --tfrecord_filename train --tfrecord_dir dataset
python convert_dataset.py --dataset_dir dataset/test --tfrecord_filename test --tfrecord_dir dataset
```

## TensorFlow Hub

In order to use [TensorFlow Hub](https://github.com/tensorflow/hub), install it using e.g.

```bash
pip install tensorflow-hub
```

When initializing a Conda environment from `environment.yaml`, this is
already taken care of.

