# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

from keras.models import Model
from keras import layers
from keras.layers import Dense, Input, BatchNormalization, Activation, Conv2D, Concatenate
from keras.layers import SeparableConv2D, MaxPooling2D, DepthwiseConv2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, Dropout
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine import Layer, InputSpec
from keras.utils import conv_utils

""" 
DeepLabv3 based on modified version of Xception.
Model architecture is from original TF graph provided by authors
"""


class BilinearUpsampling(Layer):
    '''Just a simple bilinear upsampling layer. Works only with TF.
    # Arguments
        upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w.
        name: the name of the layer
    '''

    def __init__(self, upsampling=(2, 2), data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
            input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
            input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        return K.tf.image.resize_bilinear(inputs, (int(inputs.shape[1]*self.upsampling[0]),
                                                   int(inputs.shape[2]*self.upsampling[1])))

    def get_config(self):
        config = {'size': self.upsampling,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def SepConv_BN(x, filters, prefix, strides=1, activation=None, end_act=True):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation
    Args: 
        x: input tensors
        filters: num of filters in pointwise convolution
        prefix: prefix before name
        strides: stride at depthwise conv
        activation: activation to use"""

    x = DepthwiseConv2D((3, 3), padding='same',
                        use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN')(x)
    if activation:
        x = Activation(activation)(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN')(x)
    x = Activation(activation)(x)
    return x


def Deeplabv3(input_shape=(512, 512, 3), num_classes=21, last_activation=None, weights='pascalvoc', output_stride=16):
    """ Requires input to be scaled between [-1,1]. OS == Output stride (inp_shape/decoder output)
         Args: 
             input_shape: model input shape. originaly xception had 299x299, bit deeplab used 512
             num_classes: number of classes. output has shape (input_shape[0],input_shape[1],num_classes)
             last_activation: activation on logits
             pretrain: one of {"imagenet","pascalvoc",None}
             output_stride: currently only 16 is supported."""

    # Entry Flow

    img_input = Input(shape=input_shape)

    # CONV1_1 (notation like in TF graph)
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same',
               use_bias=False, name='entry_flow_conv1_1')(img_input)  # OS = 2
    x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    x = Activation('relu')(x)
    # CONV1_2
    x = Conv2D(64, (3, 3), padding='same', use_bias=False,
               name='entry_flow_conv1_2')(x)
    x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
    x = Activation('relu')(x)

    # BLOCK 1
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False, name='entry_flow_block1_shortcut')(x)
    residual = BatchNormalization(
        name='entry_flow_block1_shortcut_BN')(residual)

    # SEPARABLE_CONV_1
    x = DepthwiseConv2D((3, 3), padding='same', use_bias=False,
                        name='entry_flow_block1_separable_conv1_depthwise')(x)
    x = BatchNormalization(
        name='entry_flow_block1_separable_conv1_depthwise_BN')(x)
    x = Conv2D(128, (1, 1), padding='same', use_bias=False,
               name='entry_flow_block1_separable_conv1_pointwise')(x)
    x = BatchNormalization(
        name='entry_flow_block1_separable_conv1_pointwise_BN')(x)
    x = Activation('relu')(x)

    # SEPARABLE_CONV_2
    x = DepthwiseConv2D((3, 3), padding='same', use_bias=False,
                        name='entry_flow_block1_separable_conv2_depthwise')(x)
    x = BatchNormalization(
        name='entry_flow_block1_separable_conv2_depthwise_BN')(x)
    x = Conv2D(128, (1, 1), padding='same', use_bias=False,
               name='entry_flow_block1_separable_conv2_pointwise')(x)
    x = BatchNormalization(
        name='entry_flow_block1_separable_conv2_pointwise_BN')(x)
    x = Activation('relu')(x)

    # SEPARABLE_CONV_3
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False,
                        name='entry_flow_block1_separable_conv3_depthwise')(x)
    x = BatchNormalization(
        name='entry_flow_block1_separable_conv3_depthwise_BN')(x)
    x = Conv2D(128, (1, 1), padding='same', use_bias=False,
               name='entry_flow_block1_separable_conv3_pointwise')(x)
    x = BatchNormalization(
        name='entry_flow_block1_separable_conv3_pointwise_BN')(x)
    x = layers.add([x, residual])  # OS = 4

    # BLOCK 2
    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False, name='entry_flow_block2_shortcut')(x)
    residual = BatchNormalization(
        name='entry_flow_block2_shortcut_BN')(residual)

    # SEPARABLE_CONV_1
    x = Activation('relu')(x)
    x = DepthwiseConv2D((3, 3), padding='same', use_bias=False,
                        name='entry_flow_block2_separable_conv1_depthwise')(x)
    x = BatchNormalization(
        name='entry_flow_block2_separable_conv1_depthwise_BN')(x)
    x = Conv2D(256, (1, 1), padding='same', use_bias=False,
               name='entry_flow_block2_separable_conv1_pointwise')(x)
    x = BatchNormalization(
        name='entry_flow_block2_separable_conv1_pointwise_BN')(x)
    x = Activation('relu')(x)

    # SEPARABLE_CONV_2
    x = DepthwiseConv2D((3, 3), padding='same', use_bias=False,
                        name='entry_flow_block2_separable_conv2_depthwise')(x)
    x = BatchNormalization(
        name='entry_flow_block2_separable_conv2_depthwise_BN')(x)
    x = Conv2D(256, (1, 1), padding='same', use_bias=False,
               name='entry_flow_block2_separable_conv2_pointwise')(x)
    skip1 = BatchNormalization(
        name='entry_flow_block2_separable_conv2_pointwise_BN')(x)  # skip мой
    x = Activation('relu')(skip1)

    # SEPARABLE_CONV_3
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False,
                        name='entry_flow_block2_separable_conv3_depthwise')(x)
    x = BatchNormalization(
        name='entry_flow_block2_separable_conv3_depthwise_BN')(x)
    x = Conv2D(256, (1, 1), padding='same', use_bias=False,
               name='entry_flow_block2_separable_conv3_pointwise')(x)
    x = BatchNormalization(
        name='entry_flow_block2_separable_conv3_pointwise_BN')(x)
    x = layers.add([x, residual])  # OS = 8

    # BLOCK 3
    residual = Conv2D(728, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False, name='entry_flow_block3_shortcut')(x)
    residual = BatchNormalization(
        name='entry_flow_block3_shortcut_BN')(residual)

    # SEPARABLE_CONV_1
    x = Activation('relu')(x)
    x = DepthwiseConv2D((3, 3), padding='same', use_bias=False,
                        name='entry_flow_block3_separable_conv1_depthwise')(x)
    x = BatchNormalization(
        name='entry_flow_block3_separable_conv1_depthwise_BN')(x)
    x = Conv2D(728, (1, 1), padding='same', use_bias=False,
               name='entry_flow_block3_separable_conv1_pointwise')(x)
    x = BatchNormalization(
        name='entry_flow_block3_separable_conv1_pointwise_BN')(x)
    x = Activation('relu')(x)

    # SEPARABLE_CONV_2
    x = DepthwiseConv2D((3, 3), padding='same', use_bias=False,
                        name='entry_flow_block3_separable_conv2_depthwise')(x)
    x = BatchNormalization(
        name='entry_flow_block3_separable_conv2_depthwise_BN')(x)
    x = Conv2D(728, (1, 1), padding='same', use_bias=False,
               name='entry_flow_block3_separable_conv2_pointwise')(x)
    x = BatchNormalization(
        name='entry_flow_block3_separable_conv2_pointwise_BN')(x)  # skip мой
    x = Activation('relu')(x)

    # SEPARABLE_CONV_3
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False,
                        name='entry_flow_block3_separable_conv3_depthwise')(x)
    x = BatchNormalization(
        name='entry_flow_block3_separable_conv3_depthwise_BN')(x)
    x = Conv2D(728, (1, 1), padding='same', use_bias=False,
               name='entry_flow_block3_separable_conv3_pointwise')(x)
    x = BatchNormalization(
        name='entry_flow_block3_separable_conv3_pointwise_BN')(x)
    x = layers.add([x, residual])  # OS = 16

    # Middle Flow
    for i in range(16):
        residual = x
        prefix = 'middle_flow_unit_' + str(i+1)

        x = Activation('relu')(x)
        x = DepthwiseConv2D((3, 3), padding='same', use_bias=False,
                            name=prefix + '_separable_conv1_depthwise')(x)
        x = BatchNormalization(
            name=prefix + '_separable_conv1_depthwise_BN')(x)
        x = Conv2D(728, (1, 1), padding='same', use_bias=False,
                   name=prefix + '_separable_conv1_pointwise')(x)
        x = BatchNormalization(
            name=prefix + '_separable_conv1_pointwise_BN')(x)

        x = Activation('relu')(x)
        x = DepthwiseConv2D((3, 3), padding='same', use_bias=False,
                            name=prefix + '_separable_conv2_depthwise')(x)
        x = BatchNormalization(
            name=prefix + '_separable_conv2_depthwise_BN')(x)
        x = Conv2D(728, (1, 1), padding='same', use_bias=False,
                   name=prefix + '_separable_conv2_pointwise')(x)
        x = BatchNormalization(
            name=prefix + '_separable_conv2_pointwise_BN')(x)

        x = Activation('relu')(x)
        x = DepthwiseConv2D((3, 3), padding='same', use_bias=False,
                            name=prefix + '_separable_conv3_depthwise')(x)
        x = BatchNormalization(
            name=prefix + '_separable_conv3_depthwise_BN')(x)
        x = Conv2D(728, (1, 1), padding='same', use_bias=False,
                   name=prefix + '_separable_conv3_pointwise')(x)
        x = BatchNormalization(
            name=prefix + '_separable_conv3_pointwise_BN')(x)

        x = layers.add([x, residual])

    # Exit flow

    # BLOCK 1
    residual = Conv2D(1024, (1, 1), strides=(1, 1),
                      padding='same', use_bias=False, name='exit_flow_block1_shortcut')(x)
    residual = BatchNormalization(
        name='exit_flow_block1_shortcut_BN')(residual)

    # SEPARABLE_CONV_1
    x = Activation('relu')(x)
    x = DepthwiseConv2D((3, 3), padding='same', use_bias=False,
                        name='exit_flow_block1_separable_conv1_depthwise')(x)
    x = BatchNormalization(
        name='exit_flow_block1_separable_conv1_depthwise_BN')(x)
    x = Conv2D(728, (1, 1), padding='same', use_bias=False,
               name='exit_flow_block1_separable_conv1_pointwise')(x)
    x = BatchNormalization(
        name='exit_flow_block1_separable_conv1_pointwise_BN')(x)
    x = Activation('relu')(x)

    # SEPARABLE_CONV_2
    x = DepthwiseConv2D((3, 3), padding='same', use_bias=False,
                        name='exit_flow_block1_separable_conv2_depthwise')(x)
    x = BatchNormalization(
        name='exit_flow_block1_separable_conv2_depthwise_BN')(x)
    x = Conv2D(1024, (1, 1), padding='same', use_bias=False,
               name='exit_flow_block1_separable_conv2_pointwise')(x)
    x = BatchNormalization(
        name='exit_flow_block1_separable_conv2_pointwise_BN')(x)  # skip мой
    x = Activation('relu')(x)

    # SEPARABLE_CONV_3
    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False,
                        name='exit_flow_block1_separable_conv3_depthwise')(x)
    x = BatchNormalization(
        name='exit_flow_block1_separable_conv3_depthwise_BN')(x)
    x = Conv2D(1024, (1, 1), padding='same', use_bias=False,
               name='exit_flow_block1_separable_conv3_pointwise')(x)
    x = BatchNormalization(
        name='exit_flow_block1_separable_conv3_pointwise_BN')(x)
    x = layers.add([x, residual])  # OS = 16 !!!

    # BLOCK 2

    # SEPARABLE_CONV_1
    # Only this encoder block has activation after depthwise convolution!
    x = DepthwiseConv2D((3, 3), padding='same', use_bias=False,
                        name='exit_flow_block2_separable_conv1_depthwise')(x)
    x = BatchNormalization(
        name='exit_flow_block2_separable_conv1_depthwise_BN')(x)
    x = Activation('relu')(x)
    x = Conv2D(1536, (1, 1), padding='same', use_bias=False,
               name='exit_flow_block2_separable_conv1_pointwise')(x)
    x = BatchNormalization(
        name='exit_flow_block2_separable_conv1_pointwise_BN')(x)
    x = Activation('relu')(x)

    # SEPARABLE_CONV_2
    x = DepthwiseConv2D((3, 3), padding='same', use_bias=False,
                        name='exit_flow_block2_separable_conv2_depthwise')(x)
    x = BatchNormalization(
        name='exit_flow_block2_separable_conv2_depthwise_BN')(x)
    x = Activation('relu')(x)
    x = Conv2D(1536, (1, 1), padding='same', use_bias=False,
               name='exit_flow_block2_separable_conv2_pointwise')(x)
    x = BatchNormalization(
        name='exit_flow_block2_separable_conv2_pointwise_BN')(x)
    x = Activation('relu')(x)

    # SEPARABLE_CONV_3
    x = DepthwiseConv2D((3, 3), padding='same', use_bias=False,
                        name='exit_flow_block2_separable_conv3_depthwise')(x)
    x = BatchNormalization(
        name='exit_flow_block2_separable_conv3_depthwise_BN')(x)
    x = Activation('relu')(x)
    x = Conv2D(2048, (1, 1), padding='same', use_bias=False,
               name='exit_flow_block2_separable_conv3_pointwise')(x)
    x = BatchNormalization(
        name='exit_flow_block2_separable_conv3_pointwise_BN')(x)
    x = Activation('relu')(x)

    # end of feature extractor
    # branching for Atrous Spatial Pyramid Pooling
    # How to use BN properly: freeze all layers up to 'exit_flow_block2_separable_conv3_pointwise_BN'
    # And use the biggest possible batch_size
    # In the article they said many times that it is very important

    # simple 1x1 conv
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN')(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    # rate = 6
    b1 = DepthwiseConv2D((3, 3), dilation_rate=(
        6, 6), padding='same', use_bias=False, name='aspp1_depthwise')(x)
    b1 = BatchNormalization(name='aspp1_depthwise_BN')(b1)
    b1 = Activation('relu')(b1)
    b1 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='aspp1_pointwise')(b1)
    b1 = BatchNormalization(name='aspp1_pointwise_BN')(b1)
    b1 = Activation('relu', name='aspp1_activation')(b1)

    # rate = 12
    b2 = DepthwiseConv2D((3, 3), dilation_rate=(
        12, 12), padding='same', use_bias=False, name='aspp2_depthwise')(x)
    b2 = BatchNormalization(name='aspp2_depthwise_BN')(b2)
    b2 = Activation('relu')(b2)
    b2 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='aspp2_pointwise')(b2)
    b2 = BatchNormalization(name='aspp2_pointwise_BN')(b2)
    b2 = Activation('relu', name='aspp2_activation')(b2)

    # rate = 18
    b3 = DepthwiseConv2D((3, 3), dilation_rate=(
        18, 18), padding='same', use_bias=False, name='aspp3_depthwise')(x)
    b3 = BatchNormalization(name='aspp3_depthwise_BN')(b3)
    b3 = Activation('relu')(b3)
    b3 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='aspp3_pointwise')(b3)
    b3 = BatchNormalization(name='aspp3_pointwise_BN')(b3)
    b3 = Activation('relu', name='aspp3_activation')(b3)

    # Image feature branch.
    out_shape = int(input_shape[0] / output_stride)
    b4 = AveragePooling2D(pool_size=(out_shape, out_shape))(x)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN')(b4)
    b4 = Activation('relu')(b4)
    b4 = BilinearUpsampling((out_shape, out_shape))(b4)

    # concatenate & project ASPP branches
    x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN')(x)
    x = Activation('relu')(x)
    x = Dropout(0.9)(x)  # not sure if this drop rate was used

    # DeepLab v.3+ decoder

    # Feature projection
    x = BilinearUpsampling((4, 4))(x)
    dec_skip1 = Conv2D(48, (1, 1), padding='same', use_bias=False,
                       activation='relu', name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(name='feature_projection0_BN')(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)
    x = Concatenate()([x, dec_skip1])

    x = DepthwiseConv2D((3, 3), padding='same', use_bias=False,
                        name='decoder_conv0_depthwise')(x)
    x = BatchNormalization(name='decoder_conv0_depthwise_BN')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (1, 1), padding='same', use_bias=False,
               name='decoder_conv0_pointwise')(x)
    x = BatchNormalization(name='decoder_conv0_pointwise_BN')(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), padding='same', use_bias=False,
                        name='decoder_conv1_depthwise')(x)
    x = BatchNormalization(name='decoder_conv1_depthwise_BN')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (1, 1), padding='same', use_bias=False,
               name='decoder_conv1_pointwise')(x)
    x = BatchNormalization(name='decoder_conv1_pointwise_BN')(x)
    x = Activation('relu', name='prelogits_activation')(x)

    # Final projection. The only place where there is bias term
    if num_classes == 21:
        logits_name = 'logits_semantic'
    else:
        logits_name = 'custom_logits_semantic'
    x = Conv2D(num_classes, (1, 1), padding='same',
               name=logits_name)(x)  # ,activation = 'sigmoid'

    x = BilinearUpsampling((4, 4))(x)

    model = Model(img_input, x, name='deeplab')

    if weights == 'pascalvoc':
        model.load_weights('models/deeplabv3_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)

    return model


def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf')
