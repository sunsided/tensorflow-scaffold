import argparse
from abc import ABCMeta, abstractmethod
from typing import Dict, Union, Optional, Tuple, Any
import tensorflow as tf
from tensorflow.contrib.training import HParams


HyperParameters = Union[dict, HParams, argparse.Namespace]
Output = Dict[str, tf.Tensor]
Metrics = Dict[str, tf.Tensor]

SingleLoss = tf.Tensor                 # A single training loss tensor
LossesToReport = Dict[str, tf.Tensor]  # Losses that should be reported during training progress
Losses = Union[SingleLoss, Tuple[SingleLoss, LossesToReport]]


class Model(metaclass=ABCMeta):
    def __init__(self, params: Optional[HyperParameters] = None):
        self.__params = self.default_hparams()
        if params is not None:
            self.apply_hparams(params)

    @property
    def params(self) -> Union[HParams, Any]:
        """
        Returns the hyperparameters.
        :return:
        """
        return self.__params

    @abstractmethod
    def default_hparams(self) -> HParams:
        """
        Defines the default hyperparameters.
        :return: The default hyperparameters.
        """
        # e.g. return tf.contrib.training.HParams(learning_rate=1e-5)
        pass

    @abstractmethod
    def build(self, features: tf.Tensor, mode: str) -> Output:
        """
        Builds the model graph.
        :param features: The input features.
        :param mode: The operation mode, one of tf.estimator.ModeKeys.
        :return: A dictionary of endpoint names to tensors.
        """
        pass

    @abstractmethod
    def loss(self, labels: tf.Tensor, net: Output) -> Losses:
        """
        Determines the losses and returns them as a dictionary.
        :param labels: The ground truth labels.
        :param net: The network output.
        :return: The losses.
        """
        pass

    def eval_metrics(self, labels: tf.Tensor, net: Output) -> Metrics:
        """
        Defines the metrics used during testing/evaluation.
        :param labels: The ground truth labels.
        :param net: The network output.
        :return: A dictionary of metric name to metric operation (e.g. tf.metrics.accuracy).
        """
        return {}

    def __call__(self, features: tf.Tensor, mode: str) -> Output:
        """
        Builds the model graph.
        :param features: The input features.
        :param mode: The operation mode, one of tf.estimator.ModeKeys.
        :return: A dictionary of endpoint names to tensors.
        """
        return self.build(features, mode)

    def apply_hparams(self, params: HyperParameters) -> None:
        """
        Applies hyperparameters to the current configuration.
        :param params: The parameters to apply.
        """
        if isinstance(params, HParams):
            params = params.values()
        elif isinstance(params, argparse.Namespace):
            params = vars(params)
        elif not isinstance(params, dict):
            raise RuntimeError("Invalid argument type for hyperparameters.")

        updated, new = {}, {}
        for k, v in params.items():
            if k in self.__params:
                updated[k] = v
            else:
                new[k] = v

        self.__params.override_from_dict(updated)
        for k, v in new.items():
            self.__params.add_hparam(k, v)

        if len(updated) > 0 and len(new) == 0:
            tf.logging.info("Updated %d hyper-parameters.", len(updated))
        elif len(updated) > 0 and len(new) > 0:
            # If this happens, something is probably missing in self.default_hparams()
            tf.logging.warn("Updated %d and added %d hyper-parameters.", len(updated), len(new))
