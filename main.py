from train import *
from test import *
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('total_epoch', 10000,
                            """Number of epoches to run.""")
tf.app.flags.DEFINE_string('dataset_name', 'mias',
                           """Name of dataset to run""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of examples in a batch""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001,
                          """Initial_learning_rate""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.9,
                          """Parameter for learning rate decay""")
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 100,
                            """Period of decaying learning rate""")
tf.app.flags.DEFINE_integer('image_size', 640,
                            """Image size""")
tf.app.flags.DEFINE_integer('channel_dim', 1,
                            """Color channel dimension""")
tf.app.flags.DEFINE_integer('num_class', 2,
                            """Nuber of classes""")
tf.app.flags.DEFINE_integer('num_gpu', 0,
                            """Number of GPU""")
tf.app.flags.DEFINE_string('phase', 'train',
                           """train or test""")
tf.app.flags.DEFINE_boolean('model_log', False,
                            """Enable/disable log of model""")
tf.app.flags.DEFINE_string('ckpt_name', 'mias2_1',
                           """'dataset_name'+'_'+'batch_size'""")
tf.app.flags.DEFINE_string('ckpt_dir', './checkpoint',
                           """./checkpoint""")

def main(_):
    if FLAGS.phase == 'train':
        if FLAGS.num_gpu == 0:
            train_with_cpu(FLAGS)
        else:
            train_with_gpu(FLAGS)
    else:
        test(FLAGS)


if __name__ == '__main__':
    tf.app.run()
