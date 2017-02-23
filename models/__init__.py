import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from models.WideResNet import WideResNet

print("Package loaded!!")

_data_dir = "./mnist/"

mnist = input_data.read_data_sets(_data_dir, one_hot=True)
print("Data loaded!!")

img_test = mnist.test.images
label_test = mnist.test.labels

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE  # 784
NUM_HIDDEN1_NODE = 256
NUM_HIDDEN2_NODE = 128
batch_size = 200


stddev = 0.1

x = tf.placeholder("float", [batch_size, 28, 28, 1])
y = tf.placeholder("float", [batch_size, NUM_CLASSES])

#x = tf.reshape(x_, [-1, 28, 28, 1])

model = WideResNet()

def cost(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

def training(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train = optimizer.minimize(loss, global_step=global_step)
    return train

def evaluation(logits, labels):
    pred = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(labels, 1))
    accr = tf.reduce_mean(tf.cast(pred, "float"))
    return accr

logits = model.inference(x)
print(logits.get_shape())
loss = cost(logits, y)
train = training(loss, learning_rate=0.001)
eval_correct = evaluation(logits, y)

init = tf.initialize_all_variables()

n_samples = mnist.train.num_examples
total_batch = int(n_samples / batch_size)
start = time.time()
total_epoch = 50

# SESSION
with tf.Session() as sess:
    sess.run(init)
    # MINI-BATCH LEARNING
    for epoch in range(total_epoch + 1):
        avg_cost = 0.
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed = {x: batch_xs, y: batch_ys}
            sess.run(train, feed)
            avg_cost += sess.run(loss, feed) / batch_size

        # DISPLAY
        if epoch % 5 == 0:
            feeds_train = {x: batch_xs, y: batch_ys}
            feeds_test = {x: img_test, y: label_test}
            train_acc = sess.run(eval_correct, feeds_train)
            test_acc = sess.run(eval_correct, feeds_test)
            print ("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f"
                   % (epoch, total_epoch, avg_cost, train_acc, test_acc))
print("DONE")
print(time.time() - start)