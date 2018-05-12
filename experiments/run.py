import argparse
import tensorflow as tf

from project.data.inputs import input_fn
from project.estimator import model_fn
from project.parameters import default_params

# FLAGS containing CLI arguments
FLAGS = None


def define_cli_args():
    """Define the experiment CLI arguments. """
    parser = argparse.ArgumentParser()

    parser.add_argument('--params', type=str, default="",
                        help=""" 
                    Comma separated list of "name=value" pairs. Possible names are:
                        train_data, gallery_data, test_data, train_steps 
                    """)

    parser.add_argument(
        '--model_dir', type=str, default='/tmp/model',
        help='The directory where the model will be stored.')

    parser.add_argument(
        '--params_file', type=str, default='',
        help='The name of the .json parameters file to load.')

    parser.add_argument(
        '--mode', type=str, default='train',
        help='run mode. One of train or eval')

    FLAGS = parser.parse_args()


def main(_):
    # Training and evaluation parameters.
    # First read the parameters form the static json file
    # and supersede them with the ones provided through the CLI.
    params = default_params()
    if FLAGS.params_file != '':
        with open(FLAGS.params_file, 'r') as f:
            params.parse_json(f.read())
    params.parse(FLAGS.params)

    # run configuration
    run_config = tf.estimator.RunConfig(save_summary_steps=200)

    # Create estimator that trains and evaluates the model
    ml_estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir,
        config=run_config
    )

    if FLAGS.mode == 'train':
        ml_estimator.train(input_fn=lambda: input_fn(params, True),
                           steps=params.train_steps)
    elif FLAGS.mode == 'eval':
        eval_name = params.eval_name if params.eval_name != '' else None
        eval_steps = params.eval_steps if params.eval_steps > 0 else None
        eval_checkpoint = FLAGS.eval_checkpoint or None
        ml_estimator.evaluate(input_fn=lambda: input_fn(params, False),
                              steps=eval_steps,
                              checkpoint_path=eval_checkpoint,
                              name=eval_name)
    else:
        print('Unknown mode. Please use one of train or eval')


if __name__ == '__main__':
    # Define CLI arguments
    define_cli_args()

    # Set tensorflow verbosity
    tf.logging.set_verbosity(tf.logging.INFO)

    # Run the experiment
    tf.app.run()
