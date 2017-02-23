import tensorflow as tf
from FusionNet import FusionNet
import numpy as np
from ops import losses
from ops import utils
from glob import glob
import cv2
import time
from Unet import Unet

def train_with_cpu(flag):
    with tf.Graph().as_default():
        data = glob('./dataset/{}/train/*.pgm'.format(flag.dataset_name))

        num_samples_per_epoch = len(data)
        num_batches_per_epoch = num_samples_per_epoch // flag.batch_size

        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0), trainable=False)

        decay_steps = int(num_batches_per_epoch * flag.num_epochs_per_decay)

        lr = tf.train.exponential_decay(flag.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        flag.learning_rate_decay_factor,
                                        staircase=True)

        input_placeholder = tf.placeholder(tf.float32,
                                           [flag.batch_size, flag.image_size, flag.image_size*2, flag.channel_dim])

        input_ = input_placeholder[:, :, :flag.image_size, :]
        target_ = input_placeholder[:, :, flag.image_size:flag.image_size*2, :]

        bool_target_mask = tf.equal(target_, 255)
        bool_target_background = tf.not_equal(target_, 255)

        float_target_mask = tf.to_float(bool_target_mask)
        float_target_background = tf.to_float(bool_target_background)
        target_ = tf.concat(concat_dim=3, values=[float_target_mask, float_target_background])

        print("target", target_.get_shape())
        target_flat = tf.reshape(target_, [1, -1, 2])
        print("flat", target_flat)

        model = FusionNet()
        #model = Unet()
        output_ = model.inference(input_)

        loss = losses.pixel_wise_cross_entropy(output_, target_)

        var_list = tf.trainable_variables()

        optimizer = tf.train.AdamOptimizer(lr)
        grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

        init = tf.initialize_all_variables()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.log_device_placement = True

        with tf.Session(config=config) as sess:
            sess.run(init)
            print("Learning start!!")
            start = time.time()

            ckpt_cnt = 1
            if utils.load_ckpt(flag.ckpt_dir, sess, flag.ckpt_name):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            for epoch in range(flag.total_epoch):
                np.random.shuffle(data)
                for batch_idx in range(num_batches_per_epoch):
                    batch_files = data[batch_idx * flag.batch_size:(batch_idx + 1) * flag.batch_size]
                    batch_image = [cv2.imread(batch_file) for batch_file in batch_files]
                    if flag.channel_dim == 1:
                        batch_images = np.array(batch_image).astype(np.float32)[:, :, :, :1]
                    else:
                        batch_images = np.array(batch_image).astype(np.float32)

                    feed = {input_placeholder: batch_images}

                    sess.run(train_op, feed_dict=feed)
                    print("(%ds) Epoch: %d[%d/%d], loss: %f" % (time.time() - start, epoch, batch_idx,
                                                                      num_batches_per_epoch-1, sess.run(loss, feed)))

                    ckpt_cnt += 1
                    if np.mod(ckpt_cnt, 100) == 1:
                        pass

                    if np.mod(ckpt_cnt, 500) == 2:
                        utils.save_ckpt(flag.ckpt_dir, ckpt_cnt, sess, flag.ckpt_name)


def train_with_gpu(flag):
    pass