###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
import configparser
import tensorflow as tf
import keras
import os
import tensorboard
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.layers.core import Layer
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout,\
    Conv2DTranspose,add, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, Dense, multiply, Lambda
from keras import optimizers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from help_functions import *
from extract_patches import get_data_training
import sys
tfe = tf.contrib.eager

sys.path.insert(0, './lib/')

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# function to obtain data for training/testing (validation)


class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.gamma = tfe.Variable(0., trainable=True, name="gamma")
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x, training=None, mask=None):
        # f = self.f(x)
        number_of_filters = x.shape[1].value
        print('use attention module')
        # print(number_of_filters)

        f = Conv2D(number_of_filters//8, (1, 1), activation='relu', padding='same', data_format='channels_first')(x)
        g = Conv2D(number_of_filters//8, (1, 1), activation='relu', padding='same', data_format='channels_first')(x)
        h = Conv2D(number_of_filters, (1, 1), activation='relu', padding='same', data_format='channels_first')(x)
        # print(f.shape, 'f')
        f_flatten = Reshape((f.shape[1].value, f.shape[2].value*f.shape[3].value))(f)
        g_flatten = Reshape((g.shape[1].value, g.shape[2].value*g.shape[3].value))(g)
        h_flatten = Reshape((h.shape[1].value, h.shape[2].value*h.shape[3].value))(h)
        s = tf.matmul(g_flatten, f_flatten, transpose_a=True)  # [B,N,C] * [B, C, N] = [B, N, N]
        # print(s.shape, 'sss')
        b = tf.nn.softmax(s, axis=-1)
        o = tf.matmul(h_flatten, b)
        y = self.gamma * tf.reshape(o, tf.shape(x)) + x
        return y


def calayer(inputs):
    x = GlobalAveragePooling2D(data_format='channels_first')(inputs)
    print('use calayer module')
    # x = Reshape((inputs.shape[1].value, 1, 1))(x)    # 用卷积的话用这行
    x1 = Reshape((1, 1, inputs.shape[1].value))(x)  # 用dense的话用这行
    # print(x.shape, 'gloshape')
    i = inputs.shape[1].value
    i /= 4
    i = int(i)
    # print(i)
    x2 = Dense(i, activation='relu')(x1)
    # print(x.shape, 'qwer')
    x3 = Dense(inputs.shape[1].value, activation='sigmoid')(x2)
    # print(x.shape)
    # x = Conv2D(i, (1, 1), activation='relu', padding='same', data_format='channels_first')(x)
    # x = Conv2D(inputs.shape[1].value, (1, 1), activation='sigmoid', padding='same', data_format='channels_first')(x)
    x4 = Reshape((inputs.shape[1].value, 1, 1))(x3)
    out = multiply([inputs, x4])
    # output = add([out, inputs])
    return out

def polayer(inputs):
    # 已经解决不同尺寸张量乘法的问题，tf.matmul
    from keras.layers import Conv2D, add, Permute
    import tensorflow as tf

    i = inputs.shape[1].value
    o = inputs.shape[2].value*inputs.shape[3].value
    a1 = Conv2D(i, (1, 1), activation='relu', padding='same', data_format='channels_first')(inputs)
    b1 = Conv2D(i, (1, 1), activation='relu', padding='same', data_format='channels_first')(inputs)
    c1 = Conv2D(i, (1, 1), activation='relu', padding='same', data_format='channels_first')(inputs)
    a2 = Reshape((i, o))(a1)
    a2 = Permute((2, 1))(a2)
    # print(a2.shape, 'a2shape')
    b2 = Reshape((i, o))(b1)
    # print(b2.shape, 'b2shape')
    c2 = Reshape((i, o))(c1)
    # print(c2.shape, 'c2shape')

    s = tf.matmul(a2, b2)
    ss = tf.nn.softmax(s)
    # s = multiply([a2, b2])
    # print(s.shape)
    # x = np.dot(c2,s)
    x = tf.matmul(c2, ss)
    # print(x.shape,'xshape')
    # x = multiply([c2, s])
    conv0 = Reshape((i, inputs.shape[2].value, inputs.shape[3].value))(x)
    conv1 = add([conv0, inputs])
    print(conv1.shape, 'conv1')
    # print(conv1.shape, 'conv1shape')
    return conv1

#
def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # y_true_f = y_true
    # print(y_true_f.shape,'ytrue')

    # y_pred_f = y_pred
    # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # print(y_pred.shape,'ypred')

    # epsilon = K.epsilon()
    # clip to prevent NaN's and Inf's
    # y_pred_f = K.clip(y_pred, epsilon, 1. - epsilon)

    intersection = K.sum(y_true_f * y_pred_f)
    # return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    # loss = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    # print(loss.shape,'lossshape')
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    # return K.sum(loss, axis=-1)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# def dice_coef(y_true, y_pred, smooth, thresh):
#     # y_pred =K.cast((K.greater(y_pred,thresh)), dtype='float32')#转换为float型
#     # y_pred = y_pred[y_pred > thresh]=1.0
#     y_true_f = y_true  # K.flatten(y_true)
#     y_pred_f = y_pred  # K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f, axis=(0, 1, 2))
#     denom = K.sum(y_true_f, axis=(0, 1, 2)) + K.sum(y_pred_f, axis=(0, 1, 2))
#     return K.mean((2. * intersection + smooth) / (denom + smooth))
#
#
# def dice_loss(smooth, thresh):
#     def dice(y_true, y_pred):
#         return 1 - dice_coef(y_true, y_pred, smooth, thresh)
#
#     return dice


# model_dice=dice_loss(smooth=1e-5,thresh=0.5)


def focal_loss(gamma=0., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


def categorical_focal_loss(gamma=0., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)**gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    References:
        https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.

        """
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        # Cross_entropy
        cross_entropy = y_true * K.log(y_pred)
        # Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        # loss = alpha * 1/y_pred * cross_entropy
        return -K.sum(loss, axis=-1)
        # return 1-loss
    return focal_loss_fixed

# Define the neural network
def get_unet(n_ch, patch_height, patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.3)(conv1)
    # conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    conv1 = Dropout(0.3)(conv1)
    # conv1 = BatchNormalization()(conv1)
    print(conv1.shape)
    # conv1 = calayer(conv1)
    # conv1 = SelfAttention()(conv1)
    # conv1 = Lambda(polayer)(conv1)
    pool1 = Conv2D(64, (2, 2), activation='relu', strides=(2, 2), padding='same', data_format='channels_first')(conv1)
    # pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    # conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.3)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    conv2 = Dropout(0.3)(conv2)
    # conv2 = BatchNormalization()(conv2)
    # conv2 = calayer(conv2)
    # conv2 = polayer(conv2)
    # conv2 = Lambda(polayer)(conv2)
    pool2 = Conv2D(128, (2, 2), activation='relu', strides=(2, 2), padding='same', data_format='channels_first')(conv2)
    # pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.3)(conv3)
    # conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    conv3 = Dropout(0.3)(conv3)
    # conv3 = BatchNormalization()(conv3)
    # conv3 = calayer(conv3)
    # conv3 = Lambda(polayer)(conv3)
    up1 = Conv2DTranspose(64, (2, 2), activation='relu', strides=(2, 2), padding='same', data_format='channels_first')(conv3)
    # up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv3)  # 上采样，行列都扩大2倍
    up1 = concatenate([conv2, up1], axis=1)  # 把这两个水平相接
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    # conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    conv4 = Dropout(0.3)(conv4)
    # conv4 = BatchNormalization()(conv4)
    # conv4 = calayer(conv4)
    # conv4 = Lambda(polayer)(conv4)
    #
    up2 = Conv2DTranspose(32, (2, 2),  activation='relu', strides=(2, 2), padding='same', data_format='channels_first')(conv4)
    # up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.3)(conv5)
    # conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    conv5 = Dropout(0.3)(conv5)
    # conv5 = BatchNormalization()(conv5)
    # conv5 = calayer(conv5)
    # conv5 = Lambda(polayer)(conv5)
    #
    conv6 = Conv2D(2, (1, 1),  padding='same', activation='relu', data_format='channels_first')(conv5)
    # conv6 = BatchNormalization()(conv6)
    conv6 = core.Reshape((2, patch_height*patch_width))(conv6)
    conv6 = core.Permute((2, 1))(conv6)   # 第一维度和第二维度互换 kernel_initializer='he_normal',
    ############
    # conv7 = Conv2D(1, 1, activation='softmax', padding='same', data_format='channels_first')(conv6)
    conv7 = core.Activation('softmax')(conv6)  # softmax激活函数

    model = Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=False)
    # model.compile(optimizer='adam', loss=binary_focal_loss(alpha=.25, gamma=2), metrics=["accuracy"])
    # model.compile(optimizer=Adam(lr=1e-4), loss=categorical_focal_loss(gamma=0, alpha=.9), metrics=["accuracy"])
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=["accuracy"])
    # sgd:随机梯度下降 categorical_crossentropy：多分类的对数损失函数,与softmax分类器相对应的损失函数 binary_crossentropy与sigmoid
    return model


def get_sumnet(n_ch, patch_height, patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    sub1 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(sub1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    add1 = add([pool1, conv2])
    sub2 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(add1)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(sub2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    add2 = add([pool2, conv3])
    sub3 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(add2)

    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(sub3)  # 上采样，行列都扩大2倍
    con1 = concatenate([add1, up1], axis=1)  # 把这两个水平相接
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(con1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    add3 = add([up1, conv4])
    sub4 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(add3)
    #
    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(sub4)
    con2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(con2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    add4 = add([up2, conv5])
    #
    conv6 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(add4)
    conv6 = core.Reshape((2, patch_height*patch_width))(conv6)
    conv6 = core.Permute((2, 1))(conv6)   # 第一维度和第二维度互换
    ############
    conv7 = core.Activation('softmax')(conv6)  # softmax激活函数

    model = Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
# sgd:随机梯度下降 categorical_crossentropy：多分类的对数损失函数,与softmax分类器相对应的损失函数
    return model

# define wnet
def get_wunet(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.5)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    # 2.......................
    pool3 = MaxPooling2D((2, 2), data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool3)
    conv6 = Dropout(0.5)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv6)
    pool4 = MaxPooling2D((2, 2), data_format='channels_first')(conv6)
    #
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool4)
    conv7 = Dropout(0.5)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv7)

    up3 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv7)
    up3 = concatenate([conv6, up3], axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up3)
    conv8 = Dropout(0.5)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv8)
    #
    up4 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv8)
    up4 = concatenate([conv5, up4], axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up4)
    conv9 = Dropout(0.5)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv9)
    # 3....................................................
    pool5 = MaxPooling2D((2, 2), data_format='channels_first')(conv9)
    #
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool5)
    conv10 = Dropout(0.5)(conv10)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv10)
    pool6 = MaxPooling2D((2, 2), data_format='channels_first')(conv10)
    #
    conv11 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool6)
    conv11 = Dropout(0.5)(conv11)
    conv11 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv11)

    up5 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv11)
    up5 = concatenate([conv10, up5], axis=1)
    conv12 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up5)
    conv12 = Dropout(0.5)(conv12)
    conv12 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv12)
    #
    up6 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv12)
    up6 = concatenate([conv9, up6], axis=1)
    conv13 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up6)
    conv13 = Dropout(0.5)(conv13)
    conv13 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv13)

    conv14 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv13)
    conv14 = core.Reshape((2, patch_height*patch_width))(conv14)
    conv14 = core.Permute((2, 1))(conv14)
    ############
    conv15 = core.Activation('softmax')(conv14)

    model = Model(inputs=inputs, outputs=conv15)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
#Define the neural network gnet
#you need change function call "get_unet" to "get_gnet" in line 166 before use this network
def get_gnet(n_ch,patch_height,patch_width):
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    up1 = UpSampling2D(size=(2, 2))(conv1)
    #
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool1 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    pool2 = MaxPooling2D((2, 2))(conv3)
    #
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    pool3 = MaxPooling2D((2, 2))(conv4)
    #
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool3)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    #
    up2 = UpSampling2D(size=(2, 2))(conv5)
    up2 = concatenate([conv4, up2], axis=1)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv6)
    #
    up3 = UpSampling2D(size=(2, 2))(conv6)
    up3 = concatenate([conv3, up3], axis=1)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up3)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv7)
    #
    up4 = UpSampling2D(size=(2, 2))(conv7)
    up4 = concatenate([conv2, up4], axis=1)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(up4)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv8)
    #
    pool4 = MaxPooling2D((2, 2))(conv8)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv9)
    #
    conv10 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv9)
    conv10 = core.Reshape((2, patch_height * patch_width))(conv10)
    conv10 = core.Permute((2, 1))(conv10)
    ############
    conv10 = core.Activation('softmax')(conv10)

    model = Model(input=inputs, output=conv10)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_uunet(n_ch,patch_height,patch_width):
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    #
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    #
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool4)
    conv5 = Dropout(0.2)(conv5)
    #
    up1 = UpSampling2D(size=(2, 2))(conv5)
    up1 = concatenate([conv4, up1], axis=1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv6)
    #
    up2 = UpSampling2D(size=(2, 2))(conv6)
    up2 = concatenate([conv3, up2], axis=1)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv7)
    #
    up3 = UpSampling2D(size=(2, 2))(conv7)
    up3 = concatenate([conv2, up3], axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up3)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv8)
    #
    up4 = UpSampling2D(size=(2, 2))(conv8)
    up4 = concatenate([conv1, up4], axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv9)
    #
    conv10 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv9)
    conv10 = core.Reshape((2, patch_height * patch_width))(conv10)
    conv10 = core.Permute((2, 1))(conv10)
    ############
    conv10 = core.Activation('softmax')(conv10)

    model = Model(input=inputs, output=conv10)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def get_testn2et(n_ch, patch_height, patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    # input2 = AveragePooling2D((2, 2), data_format='channels_first')(inputs)
    # convnv1 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(input2)
    # input3 = AveragePooling2D((2, 2), data_format='channels_first')(input2)
    # convnv2 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(input3)
    conv0 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    # conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.3)(conv1)
    # conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    conv1 = Dropout(0.3)(conv1)
    # conv1 = SelfAttention()(conv1)

    conv1 = calayer(conv1)
    add1 = add([conv1, conv0])
    conv100 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(add1)
    conv100 = Dropout(0.3)(conv100)
    # conv1 = BatchNormalization()(conv1)
    conv100 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv100)
    conv100 = Dropout(0.3)(conv100)
    conv100 = calayer(conv100)
    add100 = add([add1, conv100])
    pool1 = Conv2D(64, (2, 2), strides=(2, 2), activation='relu', padding='same')(add100)
    # print(pool1.shape, 'www')
    # pool1 = MaxPooling2D((2, 2), data_format='channels_first')(add1)
    # sub1 = Conv2D(64, (1, 1), activation='relu', padding='same', data_format='channels_first')(pool1)
    # poolol1 = concatenate([convnv1, pool1], axis=1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.3)(conv2)
    # conv2 = BatchNormalization()(conv2)
    # conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    conv2 = Dropout(0.3)(conv2)
    # conv2 = SelfAttention()(conv2)
    conv2 = calayer(conv2)
    add2 = add([pool1, conv2])
    conv200 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(add2)
    conv200 = Dropout(0.3)(conv200)
    # conv200 = BatchNormalization()(conv200)
    conv200 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv200)
    conv200 = Dropout(0.3)(conv200)
    # conv200 = SelfAttention()(conv200)
    conv200 = calayer(conv200)
    add200 = add([add2, conv200])
    pool2 = Conv2D(128, (2, 2), strides=(2, 2), activation='relu', padding='same', data_format='channels_first')(add200)
    # pool2 = MaxPooling2D((2, 2), data_format='channels_first')(add200)
    # sub2 = Conv2D(128, (1, 1), activation='relu', padding='same', data_format='channels_first')(pool2)
    # poolol2 = concatenate([convnv2, pool2], axis=1)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.3)(conv3)
    # conv3 = BatchNormalization()(conv3)
    # conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    conv3 = Dropout(0.3)(conv3)
    # conv3 = Lambda(polayer)(conv3)
    # conv3 = SelfAttention()(conv3)
    conv3 = calayer(conv3)
    add3 = add([pool2, conv3])
    conv300 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(add3)
    conv300 = Dropout(0.3)(conv300)
    # conv300 = BatchNormalization()(conv300)
    # conv300 = BatchNormalization()(conv300)
    conv300 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv300)
    conv300 = Dropout(0.3)(conv300)
    # conv300 = Lambda(polayer)(conv300)
    # conv300 = SelfAttention()(conv300)
    conv300 = calayer(conv300)
    add300 = add([conv300, add3])
    up1 = Conv2DTranspose(64, (2, 2), activation='relu', strides=(2, 2), padding='same', data_format='channels_first')(add300)
    # up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(add300)  # 上采样，行列都扩大2倍
    con1 = concatenate([add200, up1], axis=1)  # 把这两个水平相接
    sub3 = Conv2D(64, (1, 1), activation='relu', padding='same', data_format='channels_first')(con1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(sub3)
    conv4 = Dropout(0.3)(conv4)
    # conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    conv4 = Dropout(0.3)(conv4)
    # conv4 = Lambda(polayer)(conv4)
    # conv4 = SelfAttention()(conv4)
    conv4 = calayer(conv4)
    add4 = add([sub3, conv4])
    conv400 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(add4)
    conv400 = Dropout(0.3)(conv400)
    # conv400 = BatchNormalization()(conv400)
    conv400 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv400)
    conv400 = Dropout(0.3)(conv400)
    conv400 = calayer(conv400)
    # conv400 = SelfAttention()(conv400)
    add400 = add([add4, conv400])

    #
    up2 = Conv2DTranspose(32, (2, 2), activation='relu', strides=(2, 2), padding='same', data_format='channels_first')(add400)
    # up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(add400)
    con2 = concatenate([add100, up2], axis=1)
    sub4 = Conv2D(32, (1, 1), activation='relu', padding='same', data_format='channels_first')(con2)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(sub4)
    conv5 = Dropout(0.3)(conv5)
    # conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    conv5 = Dropout(0.3)(conv5)
    # conv5 = Lambda(polayer)(conv5)
    # conv5 = SelfAttention()(conv5)
    conv5 = calayer(conv5)
    add5 = add([sub4, conv5])
    conv500 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(add5)
    conv500 = Dropout(0.3)(conv500)
    # conv500 = BatchNormalization()(conv500)
    conv500 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv500)
    conv500 = Dropout(0.3)(conv500)
    conv500 = calayer(conv500)
    # conv500 = SelfAttention()(conv500)
    add500 = add([add5, conv500])
    conv110 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(add500)
    conv111 = core.Reshape((2, patch_height * patch_width))(conv110)
    conv112 = core.Permute((2, 1))(conv111)  # 第一维度和第二维度互换
    ############
    conv113 = core.Activation('softmax')(conv112)  # softmax激活函数
    # 第二轮...............................................................................................
    pool3 = Conv2D(64, (2, 2), strides=(2, 2), activation='relu', padding='same', data_format='channels_first')(add500)
    # pool3 = MaxPooling2D((2, 2), data_format='channels_first')(add500)
    # sub5 = Conv2D(64, (1, 1), activation='relu', padding='same', data_format='channels_first')(pool3)
    #
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool3)
    conv6 = Dropout(0.3)(conv6)
    # conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv6)
    conv6 = Dropout(0.3)(conv6)
    # conv6 = Lambda(polayer)(conv6)
    conv6 = calayer(conv6)
    # conv6 = SelfAttention()(conv6)
    add6 = add([pool3, conv6])
    conv600 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(add6)
    conv600 = Dropout(0.3)(conv600)
    # conv600 = BatchNormalization()(conv600)
    conv600 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv600)
    conv600 = Dropout(0.3)(conv600)
    conv600 = calayer(conv600)
    # conv600 = SelfAttention()(conv600)
    add600 = add([add6, conv600])

    pool4 = Conv2D(128, (2, 2), strides=(2, 2), activation='relu', padding='same', data_format='channels_first')(add600)
    # pool4 = MaxPooling2D((2, 2), data_format='channels_first')(add600)
    # sub6 = Conv2D(128, (1, 1), activation='relu', padding='same', data_format='channels_first')(pool4)
    #
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool4)
    conv7 = Dropout(0.3)(conv7)
    # conv7 = BatchNormalization()(conv7)
    # conv3 = BatchNormalization()(conv3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv7)
    conv7 = Dropout(0.3)(conv7)
    # conv7 = Lambda(polayer)(conv7)
    conv7 = calayer(conv7)
    # conv7 = SelfAttention()(conv7)
    add7 = add([pool4, conv7])
    conv700 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(add7)
    conv700 = Dropout(0.3)(conv700)
    # conv700 = BatchNormalization()(conv700)
    conv700 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv700)
    conv700 = Dropout(0.3)(conv700)
    conv700 = calayer(conv700)
    # conv700 = SelfAttention()(conv700)
    add700 = add([conv700, add7])
#
    up3 = Conv2DTranspose(64, (2, 2), activation='relu', strides=(2, 2), padding='same', data_format='channels_first')(add700)
    # up3 = UpSampling2D(size=(2, 2), data_format='channels_first')(add700)  # 上采样，行列都扩大2倍
    con3 = concatenate([add600, up3], axis=1)  # 把这两个水平相接
    sub7 = Conv2D(64, (1, 1), activation='relu', padding='same', data_format='channels_first')(con3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(sub7)
    conv8 = Dropout(0.3)(conv8)
    # conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv8)
    conv8 = Dropout(0.3)(conv8)
    # conv8 = Lambda(polayer)(conv8)
    # conv8 = SelfAttention()(conv8)
    conv8 = calayer(conv8)
    add8 = add([sub7, conv8])
    conv800 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(add8)
    conv800 = Dropout(0.3)(conv800)
    # conv800 = BatchNormalization()(conv800)
    conv800 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv800)
    conv800 = Dropout(0.3)(conv800)
    # conv800 = SelfAttention()(conv800)
    conv800 = calayer(conv800)
    add800 = add([add8, conv800])

    #
    up4 = Conv2DTranspose(32, (2, 2), activation='relu', strides=(2, 2), padding='same', data_format='channels_first')(add800)
    # up4 = UpSampling2D(size=(2, 2), data_format='channels_first')(add800)
    con4 = concatenate([add500, up4], axis=1)
    sub8 = Conv2D(32, (1, 1), activation='relu', padding='same', data_format='channels_first')(con4)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(sub8)
    conv9 = Dropout(0.3)(conv9)
    # conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv9)
    conv9 = Dropout(0.3)(conv9)
    # conv9 = Lambda(polayer)(conv9)
    # conv9 = SelfAttention()(conv9)
    conv9 = calayer(conv9)
    add9 = add([sub8, conv9])
    conv900 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(add9)
    conv900 = Dropout(0.3)(conv900)
    # conv900 = BatchNormalization()(conv900)
    conv900 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv900)
    conv900 = Dropout(0.3)(conv900)
    conv900 = calayer(conv900)
    # conv900 = SelfAttention()(conv900)
    add900 = add([add9, conv900])

    # 最后部分....................................................................................................
    conv10 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(add900)
    conv10 = core.Reshape((2, patch_height * patch_width))(conv10)
    conv10 = core.Permute((2, 1))(conv10)  # 第一维度和第二维度互换

    ############
    conv11 = core.Activation('softmax')(conv10)  # softmax激活函数

    model = Model(inputs=inputs, outputs=[conv11, conv113])
    # model = Model(inputs=inputs, outputs=conv113)
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(optimizer=Adam(1e-4), loss=categorical_focal_loss(gamma=0, alpha=0.8), metrics=['accuracy'], loss_weights=[0.7, 0.3])
    # model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'], loss_weights=[0.5, 0.5])
    # sgd:随机梯度下降 categorical_crossentropy：多分类的对数损失函数,与softmax分类器相对应的损失函数
    return model

def get_testn3et(n_ch, patch_height, patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    # input2 = AveragePooling2D((2, 2), data_format='channels_first')(inputs)
    # convnv1 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(input2)
    # input3 = AveragePooling2D((2, 2), data_format='channels_first')(input2)
    # convnv2 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(input3)
    conv0 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    # conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.3)(conv1)
    # conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    conv1 = Dropout(0.3)(conv1)
    # conv1 = SelfAttention()(conv1)

    conv1 = calayer(conv1)
    add1 = add([conv1, conv0])
    # conv100 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(add1)
    # conv100 = Dropout(0.3)(conv100)
    # # conv1 = BatchNormalization()(conv1)
    # conv100 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv100)
    # conv100 = Dropout(0.3)(conv100)
    # conv100 = calayer(conv100)
    # add100 = add([add1, conv100])
    pool1 = Conv2D(64, (2, 2), strides=(2, 2), activation='relu', padding='same')(add1)
    # print(pool1.shape, 'www')
    # pool1 = MaxPooling2D((2, 2), data_format='channels_first')(add1)
    # sub1 = Conv2D(64, (1, 1), activation='relu', padding='same', data_format='channels_first')(pool1)
    # poolol1 = concatenate([convnv1, pool1], axis=1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.3)(conv2)
    # conv2 = BatchNormalization()(conv2)
    # conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    conv2 = Dropout(0.3)(conv2)
    # conv2 = SelfAttention()(conv2)
    conv2 = calayer(conv2)
    add2 = add([pool1, conv2])
    # conv200 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(add2)
    # conv200 = Dropout(0.3)(conv200)
    # # conv200 = BatchNormalization()(conv200)
    # conv200 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv200)
    # conv200 = Dropout(0.3)(conv200)
    # # conv200 = SelfAttention()(conv200)
    # conv200 = calayer(conv200)
    # add200 = add([add2, conv200])
    pool2 = Conv2D(128, (2, 2), strides=(2, 2), activation='relu', padding='same', data_format='channels_first')(add2)
    # pool2 = MaxPooling2D((2, 2), data_format='channels_first')(add200)
    # sub2 = Conv2D(128, (1, 1), activation='relu', padding='same', data_format='channels_first')(pool2)
    # poolol2 = concatenate([convnv2, pool2], axis=1)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.3)(conv3)
    # conv3 = BatchNormalization()(conv3)
    # conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    conv3 = Dropout(0.3)(conv3)
    # conv3 = Lambda(polayer)(conv3)
    # conv3 = SelfAttention()(conv3)
    conv3 = calayer(conv3)
    add3 = add([pool2, conv3])
    # conv300 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(add3)
    # conv300 = Dropout(0.3)(conv300)
    # # conv300 = BatchNormalization()(conv300)
    # # conv300 = BatchNormalization()(conv300)
    # conv300 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv300)
    # conv300 = Dropout(0.3)(conv300)
    # # conv300 = Lambda(polayer)(conv300)
    # # conv300 = SelfAttention()(conv300)
    # conv300 = calayer(conv300)
    # add300 = add([conv300, add3])
    up1 = Conv2DTranspose(64, (2, 2), activation='relu', strides=(2, 2), padding='same', data_format='channels_first')(add3)
    # up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(add300)  # 上采样，行列都扩大2倍
    con1 = concatenate([add2, up1], axis=1)  # 把这两个水平相接
    sub3 = Conv2D(64, (1, 1), activation='relu', padding='same', data_format='channels_first')(con1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(sub3)
    conv4 = Dropout(0.3)(conv4)
    # conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    conv4 = Dropout(0.3)(conv4)
    # conv4 = Lambda(polayer)(conv4)
    # conv4 = SelfAttention()(conv4)
    conv4 = calayer(conv4)
    add4 = add([sub3, conv4])
    # conv400 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(add4)
    # conv400 = Dropout(0.3)(conv400)
    # # conv400 = BatchNormalization()(conv400)
    # conv400 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv400)
    # conv400 = Dropout(0.3)(conv400)
    # conv400 = calayer(conv400)
    # # conv400 = SelfAttention()(conv400)
    # add400 = add([add4, conv400])

    #
    up2 = Conv2DTranspose(32, (2, 2), activation='relu', strides=(2, 2), padding='same', data_format='channels_first')(add4)
    # up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(add400)
    con2 = concatenate([add1, up2], axis=1)
    sub4 = Conv2D(32, (1, 1), activation='relu', padding='same', data_format='channels_first')(con2)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(sub4)
    conv5 = Dropout(0.3)(conv5)
    # conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    conv5 = Dropout(0.3)(conv5)
    # conv5 = Lambda(polayer)(conv5)
    # conv5 = SelfAttention()(conv5)
    conv5 = calayer(conv5)
    add5 = add([sub4, conv5])
    # conv500 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(add5)
    # conv500 = Dropout(0.3)(conv500)
    # # conv500 = BatchNormalization()(conv500)
    # conv500 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv500)
    # conv500 = Dropout(0.3)(conv500)
    # conv500 = calayer(conv500)
    # # conv500 = SelfAttention()(conv500)
    # add500 = add([add5, conv500])
    conv110 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(add5)
    conv111 = core.Reshape((2, patch_height * patch_width))(conv110)
    conv112 = core.Permute((2, 1))(conv111)  # 第一维度和第二维度互换
    ############
    conv113 = core.Activation('softmax')(conv112)  # softmax激活函数
    # 第二轮...............................................................................................
    pool3 = Conv2D(64, (2, 2), strides=(2, 2), activation='relu', padding='same', data_format='channels_first')(add5)
    # pool3 = MaxPooling2D((2, 2), data_format='channels_first')(add500)
    # sub5 = Conv2D(64, (1, 1), activation='relu', padding='same', data_format='channels_first')(pool3)
    #
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool3)
    conv6 = Dropout(0.3)(conv6)
    # conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv6)
    conv6 = Dropout(0.3)(conv6)
    # conv6 = Lambda(polayer)(conv6)
    conv6 = calayer(conv6)
    # conv6 = SelfAttention()(conv6)
    add6 = add([pool3, conv6])
    # conv600 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(add6)
    # conv600 = Dropout(0.3)(conv600)
    # # conv600 = BatchNormalization()(conv600)
    # conv600 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv600)
    # conv600 = Dropout(0.3)(conv600)
    # conv600 = calayer(conv600)
    # # conv600 = SelfAttention()(conv600)
    # add600 = add([add6, conv600])

    pool4 = Conv2D(128, (2, 2), strides=(2, 2), activation='relu', padding='same', data_format='channels_first')(add6)
    # pool4 = MaxPooling2D((2, 2), data_format='channels_first')(add600)
    # sub6 = Conv2D(128, (1, 1), activation='relu', padding='same', data_format='channels_first')(pool4)
    #
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool4)
    conv7 = Dropout(0.3)(conv7)
    # conv7 = BatchNormalization()(conv7)
    # conv3 = BatchNormalization()(conv3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv7)
    conv7 = Dropout(0.3)(conv7)
    # conv7 = Lambda(polayer)(conv7)
    conv7 = calayer(conv7)
    # conv7 = SelfAttention()(conv7)
    add7 = add([pool4, conv7])
    # conv700 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(add7)
    # conv700 = Dropout(0.3)(conv700)
    # # conv700 = BatchNormalization()(conv700)
    # conv700 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv700)
    # conv700 = Dropout(0.3)(conv700)
    # conv700 = calayer(conv700)
    # # conv700 = SelfAttention()(conv700)
    # add700 = add([conv700, add7])
#
    up3 = Conv2DTranspose(64, (2, 2), activation='relu', strides=(2, 2), padding='same', data_format='channels_first')(add7)
    # up3 = UpSampling2D(size=(2, 2), data_format='channels_first')(add700)  # 上采样，行列都扩大2倍
    con3 = concatenate([add6, up3], axis=1)  # 把这两个水平相接
    sub7 = Conv2D(64, (1, 1), activation='relu', padding='same', data_format='channels_first')(con3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(sub7)
    conv8 = Dropout(0.3)(conv8)
    # conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv8)
    conv8 = Dropout(0.3)(conv8)
    # conv8 = Lambda(polayer)(conv8)
    # conv8 = SelfAttention()(conv8)
    conv8 = calayer(conv8)
    add8 = add([sub7, conv8])
    # conv800 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(add8)
    # conv800 = Dropout(0.3)(conv800)
    # # conv800 = BatchNormalization()(conv800)
    # conv800 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv800)
    # conv800 = Dropout(0.3)(conv800)
    # # conv800 = SelfAttention()(conv800)
    # conv800 = calayer(conv800)
    # add800 = add([add8, conv800])

    #
    up4 = Conv2DTranspose(32, (2, 2), activation='relu', strides=(2, 2), padding='same', data_format='channels_first')(add8)
    # up4 = UpSampling2D(size=(2, 2), data_format='channels_first')(add800)
    con4 = concatenate([add5, up4], axis=1)
    sub8 = Conv2D(32, (1, 1), activation='relu', padding='same', data_format='channels_first')(con4)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(sub8)
    conv9 = Dropout(0.3)(conv9)
    # conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv9)
    conv9 = Dropout(0.3)(conv9)
    # conv9 = Lambda(polayer)(conv9)
    # conv9 = SelfAttention()(conv9)
    conv9 = calayer(conv9)
    add9 = add([sub8, conv9])
    # conv900 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(add9)
    # conv900 = Dropout(0.3)(conv900)
    # # conv900 = BatchNormalization()(conv900)
    # conv900 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv900)
    # conv900 = Dropout(0.3)(conv900)
    # conv900 = calayer(conv900)
    # # conv900 = SelfAttention()(conv900)
    # add900 = add([add9, conv900])

    # 最后部分....................................................................................................
    conv10 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(add9)
    conv10 = core.Reshape((2, patch_height * patch_width))(conv10)
    conv10 = core.Permute((2, 1))(conv10)  # 第一维度和第二维度互换

    ############
    conv11 = core.Activation('softmax')(conv10)  # softmax激活函数

    model = Model(inputs=inputs, outputs=[conv11, conv113])
    # model = Model(inputs=inputs, outputs=conv113)
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(optimizer=Adam(1e-4), loss=categorical_focal_loss(gamma=0, alpha=0.8), metrics=['accuracy'], loss_weights=[0.5, 0.5])
    # model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'], loss_weights=[0.5, 0.5])
    # sgd:随机梯度下降 categorical_crossentropy：多分类的对数损失函数,与softmax分类器相对应的损失函数
    return model
#========= Load settings from Config file
config = configparser.RawConfigParser()
config.read('configuration.txt')
#patch to the datasets
path_data = config.get('data paths', 'path_local')
#Experiment name
name_experiment = config.get('experiment name', 'name')
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))



#============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original=path_data + config.get('data paths', 'dataset') + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth=path_data + config.get('data paths', 'dataset') +config.get('data paths', 'train_groundTruth'),  #masks
    patch_height=int(config.get('data attributes', 'patch_height')),
    patch_width=int(config.get('data attributes', 'patch_width')),
    N_subimgs=int(config.get('training settings', 'N_subimgs')),
    inside_FOV=config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
)


#========= Save a sample of what you're feeding to the neural network ==========
N_sample = min(patches_imgs_train.shape[0], 40)
visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_imgs")#.show()
visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_masks")#.show()


#=========== Construct and save the model arcitecture =====
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]
model = get_testn2et(n_ch, patch_height, patch_width)  #the U-net model
print("Check: final output of the network:")
print(model.output_shape)
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)



#============  Training ==================================
checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment + '_best_weights-{epoch:02d}.h5', verbose=1, monitor='val_acc', mode='max', save_best_only=False)  # save at each epoch if the validation decreased


# def step_decay(epoch):
#     lrate = 0.01 #the initial learning rate (by default in keras)
#     if epoch==100:
#         return 0.005
#     else:
#         return lrate
#
# lrate_drop = LearningRateScheduler(step_decay)

patches_masks_train = masks_Unet(patches_masks_train)  # reduce memory consumption
# history = LossHistory()
# tb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)  # 在当前目录新建logs文件夹，记录 evens.out
his = model.fit(patches_imgs_train, [patches_masks_train,patches_masks_train] , epochs=N_epochs, batch_size=batch_size, verbose=1, shuffle=True, validation_split=0.1, callbacks=[checkpointer])
# his = model.fit(patches_imgs_train, patches_masks_train, epochs=N_epochs, batch_size=batch_size, verbose=1, shuffle=True, validation_split=0.1, callbacks=[checkpointer])


# lossy = his.history['loss']
# accy = his.history['acc']
# accv = his.history['val_acc']
# lossv = his.history['val_loss']
# np_loss = np.array(lossy)
# np_acc = np.array(accy)
# np_accv = np.array(accv)
# np_lossv = np.array(lossv)
# np.savetxt('test/lossy.txt', np_lossv)
# np.savetxt('test/loss.txt', np_loss)
# np.savetxt('test/acc.txt', np_acc)
# np.savetxt('test/accv.txt', np_accv)
# ========== Save and test the last model ===================
model.save_weights('./'+name_experiment+'/'+name_experiment + '_last_weights.h5', overwrite=True)
#


# plt.figure()
# N = N_epochs
# plt.plot(np.arange(0, N), his.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), his.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), his.history["acc"], label="train_acc")
# plt.plot(np.arange(0, N), his.history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy on SegNet Satellite Seg")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig('plot.png')