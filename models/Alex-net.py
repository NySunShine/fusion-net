import tensorflow as tf
input_ = []

with tf.variable_scope('layer1'):
    w = tf.get_variable(name='w', shape=[11, 11, 3, 96], initializer=tf.contrib.layers.xavier_initializer())
    conv1 = tf.nn.conv2d(input=input_, filter=[11, 11, 3, 96], strides=[1,1,1,1], padding='VALID', name='conv1')
    b = tf.get_variable(name='b', shape=[])
    pool1 =