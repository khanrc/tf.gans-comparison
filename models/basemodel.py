# coding: utf-8

'''BaseModel for Generative Adversarial Netowrks.
'''

import tensorflow as tf
slim = tf.contrib.slim


class BaseModel(object):
    FAKE_MAX_OUTPUT = 6

    def __init__(self, name, training, D_lr, G_lr, image_shape=[64, 64, 3], z_dim=100):
        self.name = name
        self.shape = image_shape
        self.bn_params = {
            "decay": 0.99,
            "epsilon": 1e-5,
            "scale": True,
            "is_training": training
        }
        self.z_dim = z_dim
        self.D_lr = D_lr
        self.G_lr = G_lr
        self.args = vars(self).copy() # dict

        if training == True:
            self._build_train_graph()
        else:
            self._build_gen_graph()


    def _build_gen_graph(self):
        '''build computational graph for generation (evaluation)'''
        with tf.variable_scope(self.name):
            self.z = tf.placeholder(tf.float32, [None, self.z_dim])
            self.fake_sample = tf.clip_by_value(self._generator(self.z), -1., 1.)


    def _build_train_graph(self, X):
        '''build computational graph for training'''
        pass