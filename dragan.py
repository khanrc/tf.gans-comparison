# coding: utf-8
import tensorflow as tf
slim = tf.contrib.slim
from utils import expected_shape
import ops
from basemodel import BaseModel

'''
DRAGAN 은 DCGAN + GP 느낌임.
'''

class DRAGAN(BaseModel):
    def _build_train_graph(self):
        '''build computational graph for training
        '''
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)

            G = self._generator(z)
            D_real_prob, D_real_logits = self._discriminator(X)
            D_fake_prob, D_fake_logits = self._discriminator(G, reuse=True)

            G_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_fake_logits), logits=D_fake_logits)
            D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_real_logits), logits=D_real_logits)
            D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(D_fake_logits), logits=D_fake_logits)
            D_loss = D_loss_real + D_loss_fake

            # Gradient Penalty (GP)
            # perturbed minibatch: x_noise = x_i + noise_i
            # x_hat = alpha*x + (1-alpha)*x_noise = x_i + (1-alpha)*noise_i
            ld = 10.
            C = 0.5
            
            shape = tf.shape(X)
            eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(X, axes=[0,1,2,3])
            x_std = tf.sqrt(x_var) # magnitude of noise decides the size of local region
            noise = C*x_std*eps # delta in paper
            # perturbed minibatch: Xp = X + noise

            alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=0., maxval=1.)
            xhat = X + alpha*noise

            D_xhat_prob, D_xhat_logits = self._discriminator(xhat, reuse=True)
            D_xhat_grad = tf.gradients(D_xhat_prob, xhat)[0] # gradient of D(x_hat)
            # tf.norm 함수가 좀 이상해서, axis 가 reduce_mean 처럼 작동하긴 하는데 3차원 이상 줄 수 없음. 따라서 아래처럼 flatten 을 활용함
            D_xhat_grad_norm = tf.norm(slim.flatten(D_xhat_grad), axis=1)  # l2 norm
            # GP = ld * tf.reduce_mean(tf.square(tf.reduce_sum(tf.square(D_xhat_prob), axis=[1,2,3])**0.5 - 1.)) # 이것도 맞음
            GP = ld * tf.reduce_mean(tf.square(D_xhat_grad_norm - 1.))
            D_loss += GP

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/discriminator/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/generator/')

            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/discriminator/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/generator/')

            lr = 1e-4
            beta1 = 0.5
            beta2 = 0.9
            with tf.control_dependencies(D_update_ops):
                D_train_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2).minimize(D_loss, var_list=D_vars)
            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2).minimize(G_loss, var_list=G_vars, global_step=global_step)

            # summaries
            # per-step summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G_loss', G_loss),
                tf.summary.scalar('D_loss', D_loss),
                tf.summary.scalar('GP', GP)
            ])

            # sparse-step summary
            tf.summary.image('fake_sample', G, max_outputs=6)
            tf.summary.histogram('real_probs', D_real_prob)
            tf.summary.histogram('fake_probs', D_fake_prob)
            self.all_summary_op = tf.summary.merge_all()

            # accesible points
            self.X = X
            self.z = z
            self.D_train_op = D_train_op 
            self.G_train_op = G_train_op
            self.fake_sample = G
            self.global_step = global_step

    # DRAGAN does not use BN
    def _discriminator(self, X, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            net = X
            
            with slim.arg_scope([slim.conv2d], kernel_size=[5,5], stride=2, padding='SAME', activation_fn=ops.lrelu):
                net = slim.conv2d(net, 64)
                expected_shape(net, [32, 32, 64])
                net = slim.conv2d(net, 128)
                expected_shape(net, [16, 16, 128])
                net = slim.conv2d(net, 256)
                expected_shape(net, [8, 8, 256])
                net = slim.conv2d(net, 512)
                expected_shape(net, [4, 4, 512])

            net = slim.flatten(net)
            logits = slim.fully_connected(net, 1, activation_fn=None)
            prob = tf.nn.sigmoid(logits)

            return prob, logits

    def _generator(self, z, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            net = z
            net = slim.fully_connected(net, 4*4*1024, activation_fn=tf.nn.relu)
            net = tf.reshape(net, [-1, 4, 4, 1024])

            with slim.arg_scope([slim.conv2d_transpose], kernel_size=[5,5], stride=2, padding='SAME', activation_fn=tf.nn.relu):
                net = slim.conv2d_transpose(net, 512)
                expected_shape(net, [8, 8, 512])
                net = slim.conv2d_transpose(net, 256)
                expected_shape(net, [16, 16, 256])
                net = slim.conv2d_transpose(net, 128)
                expected_shape(net, [32, 32, 128])
                net = slim.conv2d_transpose(net, 3, activation_fn=tf.nn.tanh, normalizer_fn=None)
                expected_shape(net, [64, 64, 3])

                return net
