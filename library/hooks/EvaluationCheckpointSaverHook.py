import os
from typing import Dict, Optional
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util
from tensorflow.python.framework import ops
from tensorflow.python.training.session_run_hook import SessionRunArgs


class EvaluationCheckpointSaverHook(tf.train.SessionRunHook):
    """Saves checkpoints every N steps or seconds."""

    def __init__(self,
                 checkpoint_dir,
                 tensors_to_minimize=Dict[str, Optional[float]],
                 saver=None,
                 checkpoint_basename="eval.ckpt",
                 scaffold=None,
                 listeners=None):
        """Initializes a `CheckpointSaverHook`.
    Args:
      checkpoint_dir: `str`, base directory for the checkpoint files.
      tensors_to_minimize: `Dict[str, Optional[float]]`, dictionary of tensor names to their current values to minimize
      saver: `Saver` object, used for saving.
      checkpoint_basename: `str`, base name for the checkpoint files.
      scaffold: `Scaffold`, use to get saver object.
      listeners: List of `CheckpointSaverListener` subclass instances.
        Used for callbacks that run immediately before or after this hook saves
        the checkpoint.
    Raises:
      ValueError: One of `save_steps` or `save_secs` should be set.
      ValueError: At most one of saver or scaffold should be set.
    """
        logging.info("Create EvaluationCheckpointSaverHook.")
        if saver is not None and scaffold is not None:
            raise ValueError("You cannot provide both saver and scaffold.")
        self._saver = saver
        self._global_step_tensor = None
        self._metrics_to_minimize = {p[0]: p[1] for p in tensors_to_minimize.items()}
        self._tensors = None
        self._accumulated_values = None
        self._accumulation_count = 0
        self._checkpoint_dir = checkpoint_dir
        self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)
        self._scaffold = scaffold
        self._listeners = listeners or []
        self._graph_saved = False

    def begin(self):
        # noinspection PyProtectedMember
        self._global_step_tensor = training_util._get_or_create_global_step_read()

        graph = ops.get_default_graph()
        self._tensors = [(n, graph.get_tensor_by_name(n + ':0')) for n in self._metrics_to_minimize]
        self._accumulated_values = {tensor[0]: 0.0 for tensor in self._tensors}
        self._accumulation_count = 0
        self._saver = self._saver if self._saver is not None else tf.train.Saver(max_to_keep=5)

        for l in self._listeners:
            l.begin()

    def before_run(self, run_context):
        if not self._graph_saved:
            self._graph_saved = True
            # We do write graph and saver_def at the first call of before_run.
            # We cannot do this in begin, since we let other hooks to change graph and
            # add variables in begin. Graph is finalized after all begin calls.
            training_util.write_graph(
                ops.get_default_graph().as_graph_def(add_shapes=True),
                self._checkpoint_dir,
                "graph.pbtxt")

        fetches = [pair[1] for pair in self._tensors]
        return SessionRunArgs(fetches=fetches)

    def after_run(self, run_context, run_values):
        self._accumulation_count += 1

        values = {tensor[0]: value for tensor, value in zip(self._tensors, run_values.results)}
        for name, new_value in values.items():
            self._accumulated_values[name] += new_value

    def end(self, session):
        logging.info("Evaluation session ended. Testing for improvements ...")
        last_step = session.run(self._global_step_tensor)
        for l in self._listeners:
            l.end(session, last_step)

        take_snapshot = False
        for name, new_value in self._accumulated_values.items():
            # Obtain the average value over the batches.
            new_value /= self._accumulation_count
            old_value = self._metrics_to_minimize[name]
            if old_value is None:
                self._metrics_to_minimize[name] = new_value
                continue
            if old_value > new_value:
                take_snapshot = True
                self._metrics_to_minimize[name] = new_value
            else:
                logging.info("Evaluation metric \"%s\" did not improve (%f >= %f).", name, new_value, old_value)

        if take_snapshot:
            global_step = session.run(self._global_step_tensor)
            self._save(session, global_step)

    def _save(self, session, step):
        """Saves the latest checkpoint."""
        logging.info("Evaluation metrics have improved. Saving checkpoints for %d into %s.", step, self._save_path)

        for l in self._listeners:
            l.before_save(session, step)

        self._get_saver().save(session, self._save_path, global_step=step)

        for l in self._listeners:
            l.after_save(session, step)

    def _get_saver(self):
        if self._saver is not None:
            return self._saver
        elif self._scaffold is not None:
            return self._scaffold.saver

        # Get saver from the SAVERS collection if present.
        collection_key = ops.GraphKeys.SAVERS
        savers = ops.get_collection(collection_key)
        if not savers:
            raise RuntimeError(
                "No items in collection {}. Please add a saver to the collection "
                "or provide a saver or scaffold.".format(collection_key))
        elif len(savers) > 1:
            raise RuntimeError(
                "More than one item in collection {}. "
                "Please indicate which one to use by passing it to the constructor.".format(collection_key))

        self._saver = savers[0]
        return savers[0]
