import tensorflow as tf
from argparse import Namespace
from tensorflow.contrib.training import HParams
from typing import Union, Dict, Tuple


HyperParameters = Union[dict, HParams, Namespace]
Output = Dict[str, tf.Tensor]
Metrics = Dict[str, tf.Tensor]

SingleLoss = tf.Tensor                 # A single training loss tensor
LossesToReport = Dict[str, tf.Tensor]  # Losses that should be reported during training progress
Losses = Union[SingleLoss, Tuple[SingleLoss, LossesToReport]]