from ops import acts, layers, losses, utils
import tensorflow as tf

class WideResNet(object):
    def __init__(self):
        self.activation_fn = acts.pRelu
        self.num_class = 10
        self.num_kernel = 16
        self.pool_kernel = 4

    def res_act(self, input_, name='res_act'):
        bn = layers.batch_norm(input_, name=name + '_bn')
        act = self.activation_fn(bn, name=name + '_act')
        return act

    def res_block(self, input_, output_dim, is_downsizing=True, name="res_block"):
        with tf.variable_scope(name):
            act1 = self.res_act(input_, name='act1')

            if is_downsizing:
                skip2 = layers.bottleneck_layer(act1, output_dim, d_h=2, d_w=2, name='skip1')
                _, conv1 = layers.conv2d_same_act(act1, output_dim, d_h=2, d_w=2,
                                                  activation_fn=self.activation_fn, with_logit=True, name='conv1')
            else:
                skip2 = layers.bottleneck_layer(act1, output_dim, d_h=1, d_w=1, name='skip1')
                _, conv1 = layers.conv2d_same_act(act1, output_dim, d_h=1, d_w=1,
                                                  activation_fn=self.activation_fn, with_logit=True, name='conv1')
            conv2 = layers.conv2d_same(conv1, output_dim, name='conv2')
            res1 = tf.add(skip2, conv2, name='res1')

            act2 = self.res_act(res1, name='act2')
            _, conv3 = layers.conv2d_same_repeat(act2, output_dim, num_repeat=2, d_h=1, d_w=1,
                                                 activation_fn=self.activation_fn, with_logit=True, name='conv3')
            res2 = tf.add(res1, conv3, name='res2')

            return res2

    def inference(self, input_):
        conv1 = layers.conv2d_same(input_, self.num_kernel, name='conv1')

        res_block1 = self.res_block(conv1, self.num_kernel * 2, is_downsizing=False, name='res_block1')
        res_block2 = self.res_block(res_block1, self.num_kernel * 4, is_downsizing=True, name='res_block2')
        res_block3 = self.res_block(res_block2, self.num_kernel * 8, is_downsizing=True, name='res_block3')

        act = self.res_act(res_block3)
        pool = layers.avg_pool(act, k_h=self.pool_kernel, k_w=self.pool_kernel, d_h=1, d_w=1, name='pool')
        flat = layers.flatten(pool, 'flat')

        linear = layers.linear(flat, self.num_class, name='linear')

        return linear

    def _inference(self, input_):
        conv1 = layers.conv2d_same_act(input_, 16, activation_fn=self.activation_fn, name='conv1')
        skip1 = layers.bottleneck_layer(conv1, 32, name='skip1')
        _, conv2 = layers.conv2d_same_repeat(conv1, 32, num_repeat=2,
                                             activation_fn=self.activation_fn, with_logit=True, name='conv2')

        res1 = tf.add(skip1, conv2, name='res1')
        res_act1 = self.res_act(res1)

        _, conv3 = layers.conv2d_same_repeat(res_act1, 32, num_repeat=2,
                                             activation_fn=self.activation_fn, with_logit=True, name='conv3')

        res2 = tf.add(conv3, res1, name='res2')
        res_act2 = self.res_act(res2)

        skip2 = layers.bottleneck_layer(res_act2, 64, d_h=2, d_w=2, name='skip2')
        conv4 = layers.conv2d_same_act(res_act2, 64, d_h=2, d_w=2, activation_fn=self.activation_fn, name='conv4')
        conv5 = layers.conv2d_same(conv4, 64, name='conv5')

        res3 = tf.add(skip2, conv5, name='res3')
        res_act3 = self.res_act(res3)

        _, conv6 = layers.conv2d_same_repeat(res_act1, 64, num_repeat=2,
                                             activation_fn=self.activation_fn, with_logit=True, name='conv3')

        res4 = tf.add(res3, conv6, name='res4')
        res_act4 = self.res_act(res4)

        skip3 = layers.bottleneck_layer(res_act4, 128, d_h=2, d_w=2, name='skip3')
        conv7 = layers.conv2d_same_act(res_act4, 128, d_h=2, d_w=2, activation_fn=self.activation_fn, name='conv7')
        conv8 = layers.conv2d_same(conv7, 128, name='conv8')

        res5 = tf.add(skip3, conv8, name='res5')

        res_act5 = self.res_act(res5)
        _, conv9 = layers.conv2d_same_repeat(res_act5, 128, num_repeat=2,
                                             activation_fn=self.activation_fn, with_logit=True, name='conv9')

        res6 = tf.add(res5, conv9, name='res6')
        res_act6 = self.res_act(res6)

        pool = layers.avg_pool(res_act6, k_h=8, k_w=8, d_h=1, d_w=1, name='pool')
        flat = layers.flatten(pool, 'flat')

        linear = layers.linear(flat, self.num_class, name='linear')

        return linear