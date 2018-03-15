# coding=utf-8
from __future__ import print_function

"""
include some images and some varable visibel about lenet

"""

from tensorflow.examples.tutorials.mnist import input_data
import argparse
import sys
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot  as plt
from batchnorm import batchnorm
from units import conv2d
from units import max_pool_2x2
from units import avg_pool_2x2
from units import weight_variable
from units import bias_variable
from units import run_in_batch_avg
from none_local_blcok_2D import non_local_block

import time

decay = 0.95
batch_size = 32

# ***********************************#
# build model
# ***********************************#


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

x_input = tf.placeholder("float", shape=[None, 784], name='x_input')
y_ = tf.placeholder("float", shape=[None, 10], name='label')
is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

x = tf.reshape(x_input, [-1, 28, 28, 1])

with tf.variable_scope('conv'):
    x = batchnorm(x, scope='conv1', is_training=is_training, decay=decay)
    x = tf.nn.relu(x)
    x = conv2d(x, 1, 32, 3)

    x = batchnorm(x, scope='conv2', is_training=is_training, decay=decay)
    x = tf.nn.relu(x)
    x = conv2d(x, 32, 64, 3)
    x = avg_pool_2x2(x)

    x = batchnorm(x, scope='conv3', is_training=is_training, decay=decay)
    x = tf.nn.relu(x)
    x = conv2d(x, 64, 64, 3)
    x = avg_pool_2x2(x)

    x = batchnorm(x, scope='conv4', is_training=is_training, decay=decay)
    x = tf.nn.relu(x)

    x = conv2d(x, 64, 64, 3)

    x = batchnorm(x, scope='conv5', is_training=is_training, decay=decay)
    x = tf.nn.relu(x)

    x = conv2d(x, 64, 64, 3)

with tf.variable_scope('none_block'):
    # ['gaussian', 'embedded', 'dot','concatenate']
    x = non_local_block(x, mode='concatenate')

with tf.variable_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(x, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

with tf.variable_scope('dropout'):
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.variable_scope('fc12'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# ***********************************#
# define metrics and optimier
# ***********************************#

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
tf.summary.scalar("cross_entropy", cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
tf.summary.scalar("accuracy", accuracy)

# ***********************************#
# train
# ***********************************#

sess.run(tf.global_variables_initializer())

merged_summary_op = tf.summary.merge_all()

log_dir = "/home/dsz/PycharmProjects/log1"
if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
tf.gfile.MakeDirs(log_dir)
test_summary_writer = tf.summary.FileWriter(log_dir + "/test")
tain_summary_writer = tf.summary.FileWriter(log_dir + "/train")

train_accuracys = []
test_accuracys = []

iters = 800
for i in range(iters):
    batch = mnist.train.next_batch(batch_size)

    t0 = time.time()
    # train
    train_summary, _, train_accuracy = sess.run([merged_summary_op, train_step, accuracy],
                                                feed_dict={x_input: batch[0], y_: batch[1], keep_prob: 0.8,
                                                           is_training: True})
    train_time = time.time() - t0
    tain_summary_writer.add_summary(train_summary, i + 1)

    print("step %d, training_accuracy %g ,train_step_time_cost %f" % (i, train_accuracy, train_time), end='')

    train_accuracys.append(train_accuracy)
    # using the same train for val to make sure btchnorm layer work corectlly
    t1 = time.time()
    test_summary, test_accuracy = sess.run([merged_summary_op, accuracy],
                                           feed_dict={x_input: batch[0], y_: batch[1], keep_prob: 1.0,
                                                      is_training: False})
    test_time = time.time() - t1
    print(" test_accuracy %g,test_step_time_cost %f" % (test_accuracy, test_time))
    test_summary_writer.add_summary(test_summary, i + 1)
    test_accuracys.append(test_accuracy)

# ***********************************#
# test
# ***********************************#

t2 = time.time()

# this operation cost too much memory
# ac = accuracy.eval(feed_dict={x_input: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0,is_training:False})

test_results = run_in_batch_avg(sess,
                                [accuracy],
                                [x_input, y_],
                                feed_dict={x_input: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0,
                                           is_training: False},
                                batch_size=batch_size
                                )
ev_time = time.time() - t2

print("test_results", test_results)

print("test accuracy %g,ev_step_time_cost %f" % (test_results[0], ev_time))

plt.plot(range(iters), train_accuracys, range(iters), test_accuracys)
plt.show()
