# coding=utf-8
import tensorflow as tf
import cv2
import numpy as np


def batchnorm(x, scope, is_training, epsilon=0.001, decay=0.95):
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()
        mean, var = tf.nn.moments(x,
                                  axes=range(len(shape) - 1)  # 想要 normalize 的维度, [0] 代表 batch 维度
                                  #  如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width] 的均值/方差, 注意不要加入 channel 维度
                                  )

        scale = tf.Variable(tf.ones(shape[-1]))
        shift = tf.Variable(tf.zeros(shape[-1]))

        ema = tf.train.ExponentialMovingAverage(decay=decay)  # exponential moving average 的 decay 度

        def mean_var_with_update():
            ema_apply_op = ema.apply([mean, var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(mean), tf.identity(var)

        # mean, var = mean_var_with_update()  # 根据新的 batch 数据, 记录并稍微修改之前的 mean/var
        mean, var = tf.cond(is_training,  # is_training 的值是 True/False
                            mean_var_with_update,  # 如果是 True, 更新 mean/var
                            lambda: (  # 如果是 False, 返回之前 fc_mean/fc_var 的Moving Average
                                ema.average(mean),
                                ema.average(var)
                            )
                            )

    output = tf.nn.batch_normalization(x, mean, var, shift, scale, epsilon)
    return output
