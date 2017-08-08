# coding: utf-8

import tensorflow as tf
slim = tf.contrib.slim


def lrelu(inputs, leak=0.2, scope="lrelu"):
    """
    https://github.com/tensorflow/tensorflow/issues/4079
    """
    with tf.variable_scope(scope):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * inputs + f2 * abs(inputs)
