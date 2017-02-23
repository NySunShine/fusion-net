from ops import acts, layers, losses, utils
import tensorflow as tf

class VGG(object):
    def __init__(self):
        self.activation_fn = acts.pRelu
        self.num_class = 10
        self.num_kernel = 16
        self.pool_kernel = 2

    def inference(self, input_):
        conv1 = layers.conv2d_same_repeat(input_, self.kernel_num, num_repeat=2, name="down1")
        pool1 = layers.max_pool(conv1, name="pool1")

        conv2 = layers.conv2d_same_repeat(pool1, self.kernel_num * 2, num_repeat=2, name="down2")
        pool2 = layers.max_pool(conv2, name="pool2")

        conv3 = layers.conv2d_same_repeat(pool2, self.kernel_num * 4, num_repeat=3, name="down3")
        pool3 = layers.max_pool(conv3, name="pool3")

        conv4 = layers.conv2d_same_repeat(pool3, self.kernel_num * 8, num_repeat=3, name="down4")
        pool4 = layers.max_pool(conv4, name="pool4")

        conv5 = layers.conv2d_same_repeat(pool4, self.kernel_num * 8, num_repeat=3, name="down5")
        pool5 = layers.max_pool(conv5, name="pool5")

        flat = layers.flatten(pool5, 'flat')

        linear = layers.linear(flat, flat.get_shape().as_list()[-1], name='linear')

        logits = layers.linear(linear, self.num_class, name='logits')

        return logits
