from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from ops import layers
from ops import acts


class FusionNet(object):
    def __init__(self):
        self.act_fn = acts.pRelu
        self.kernel_num = 64
        self.output_dim = 2
        self.log = 0
        print("FusionNet Loading"),

    def skip_connection(self, input_, output_):
        return tf.add(input_, output_)

    def res_block_with_n_conv_layers(self, input_, output_dim, num_repeat, name="res_block"):
        output_ = layers.conv2d_same_repeat(input_, output_dim,
                                            num_repeat=num_repeat, activation_fn=self.act_fn, name=name)
        return self.skip_connection(input_, output_)

    def res_block_with_3_conv_layers(self, input_, output_dim, name="res_block"):
        return self.res_block_with_n_conv_layers(input_, output_dim, num_repeat=3, name=name)

    def conv_res_conv_block(self, input_, output_dim, name="conv_res_conv_block"):
        with tf.variable_scope(name):
            conv1 = layers.conv2d_same_act(input_, output_dim, activation_fn=self.act_fn,
                                           with_logit=False, name="conv1")
            res = self.res_block_with_3_conv_layers(conv1, output_dim, name="res_block")
            conv2 = layers.conv2d_same_act(res, output_dim, activation_fn=self.act_fn,
                                           with_logit=False, name="conv2")
            return conv2

    def encoder(self, input_):
        self.down1 = self.conv_res_conv_block(input_, self.kernel_num, name="down1")
        pool1 = layers.max_pool(self.down1, name="pool1")

        self.down2 = self.conv_res_conv_block(pool1, self.kernel_num * 2, name="down2")
        pool2 = layers.max_pool(self.down2, name="pool2")

        self.down3 = self.conv_res_conv_block(pool2, self.kernel_num * 4, name="down3")
        pool3 = layers.max_pool(self.down3, name="pool3")

        self.down4 = self.conv_res_conv_block(pool3, self.kernel_num * 8, name="down4")
        pool4 = layers.max_pool(self.down4, name="pool4")

        if self.log == 1:
            print("encoder input : ", input_.get_shape())
            print("conv1 : ", self.down1.get_shape())
            print("pool1 : ", pool1.get_shape())
            print("conv2 : ", self.down2.get_shape())
            print("pool2 : ", pool2.get_shape())
            print("conv3 : ", self.down3.get_shape())
            print("pool3 : ", pool3.get_shape())
            print("conv4 : ", self.down4.get_shape())
            print("pool4 : ", pool4.get_shape())

        return pool4

    def decoder(self, input_):
        conv_trans4 = layers.conv2dTrans_same_act(input_, self.down4.get_shape(),
                                                  activation_fn=self.act_fn, with_logit=False, name="unpool4")
        res4 = self.skip_connection(conv_trans4, self.down4)
        up4 = self.conv_res_conv_block(res4, self.kernel_num * 8, name="up4")

        conv_trans3 = layers.conv2dTrans_same_act(up4, self.down3.get_shape(),
                                                  activation_fn=self.act_fn, with_logit=False, name="unpool3")
        res3 = self.skip_connection(conv_trans3, self.down3)
        up3 = self.conv_res_conv_block(res3, self.kernel_num * 4, name="up3")

        conv_trans2 = layers.conv2dTrans_same_act(up3, self.down2.get_shape(),
                                                  activation_fn=self.act_fn, with_logit=False, name="unpool2")
        res2 = self.skip_connection(conv_trans2, self.down2)
        up2 = self.conv_res_conv_block(res2, self.kernel_num * 2, name="up2")

        conv_trans1 = layers.conv2dTrans_same_act(up2, self.down1.get_shape(),
                                                  activation_fn=self.act_fn, with_logit=False, name="unpool1")
        res1 = self.skip_connection(conv_trans1, self.down1)
        up1 = self.conv_res_conv_block(res1, self.kernel_num, name="up1")

        if self.log == 1:
            print("dncoder input : ", input_.get_shape())
            print("convT1 : ", conv_trans4.get_shape())
            print("res1 : ", res4.get_shape())
            print("up1 : ", up4.get_shape())
            print("convT2 : ", conv_trans3.get_shape())
            print("res2 : ", res3.get_shape())
            print("up2 : ", up3.get_shape())
            print("convT3 : ", conv_trans2.get_shape())
            print("res3 : ", res2.get_shape())
            print("up3 : ", up2.get_shape())
            print("convT4 : ", conv_trans1.get_shape())
            print("res4 : ", res1.get_shape())
            print("up4 : ", up1.get_shape())

        return up1

    def inference(self, input_):
        encode_vec = self.encoder(input_)
        bridge = self.conv_res_conv_block(encode_vec, self.kernel_num * 16, name="bridge")
        decode_vec = self.decoder(bridge)
        output = layers.bottleneck_layer(decode_vec, self.output_dim, name="output")

        if self.log == 1:
            print("output : ", output.get_shape())

        print("Complete!!")

        return output
