# coding: utf-8

'''
BaseModel for Generative Adversarial Netowrks.
이 모델을 상속받아서 _build_train_graph, _discriminator, _generator 세가지만 구현해주면 된다.
'''

import tensorflow as tf
slim = tf.contrib.slim


class BaseModel(object):
    def __init__(self, input_pipe, z_dim=100, name='basemodel'):
        '''
        training mode: input_pipe = input pipeline
        generation mode: input_pipe = None
        '''
        self.name = name
        training = input_pipe is not None
        # check: DCGAN specified BN-params?
        self.bn_params = {
            "decay": 0.99,
            "epsilon": 1e-5,
            "scale": True,
            "is_training": training
        }
        self.z_dim = z_dim
        if training == True:
            self._build_train_graph(input_pipe)
        else:
            self._build_gen_graph()


    def _build_gen_graph(self):
        '''build computational graph for generation (evaluation)
        '''
        with tf.variable_scope(self.name):
            self.z = tf.placeholder(tf.float32, [None, self.z_dim])
            self.fake_sample = self._generator(self.z)


    def _build_train_graph(self, X):
        '''build computational graph for training
        '''
        pass