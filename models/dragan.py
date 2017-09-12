# coding: utf-8
import tensorflow as tf
slim = tf.contrib.slim
from utils import expected_shape
import ops
from basemodel import BaseModel

'''
DRAGAN has similar gradient penalty to WGAN-GP, although different motivation.
It is also similar to DCGAN except for gradient penalty.
'''

class DRAGAN(BaseModel):
    def __init__(self, name, training, D_lr=1e-4, G_lr=1e-4, image_shape=[64, 64, 3], z_dim=100):
        self.beta1 = 0.5
        self.beta2 = 0.9
        self.ld = 10. # lambda
        self.C = 0.5
        super(DRAGAN, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr, 
            image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
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

            shape = tf.shape(X)
            eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(X, axes=[0,1,2,3])
            x_std = tf.sqrt(x_var) # magnitude of noise decides the size of local region
            noise = self.C*x_std*eps # delta in paper
            # Author suggested U[0,1] in original paper, but he admitted it is bug in github
            # (https://github.com/kodalinaveen3/DRAGAN). It should be two-sided.
            alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.) 
            xhat = tf.clip_by_value(X + alpha*noise, -1., 1.) # x_hat should be in the space of X

            D_xhat_prob, D_xhat_logits = self._discriminator(xhat, reuse=True)
            # Originally, the paper suggested D_xhat_prob instead of D_xhat_logits.
            # But D_xhat_prob (D with sigmoid) causes numerical problem (NaN in gradient).
            D_xhat_grad = tf.gradients(D_xhat_logits, xhat)[0] # gradient of D(x_hat)
            D_xhat_grad_norm = tf.norm(D_xhat_grad, axis=1)  # l2 norm
            # D_xhat_grad_norm = tf.sqrt(tf.reduce_sum(tf.square(D_xhat_grad), axis=[1]))
            GP = self.ld * tf.reduce_mean(tf.square(D_xhat_grad_norm - 1.))
            D_loss += GP

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/discriminator/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/generator/')

            # DRAGAN does not use BN, so you don't need to set control dependencies for update ops.
            D_train_op = tf.train.AdamOptimizer(learning_rate=self.D_lr, beta1=self.beta1, beta2=self.beta2).\
                minimize(D_loss, var_list=D_vars)
            G_train_op = tf.train.AdamOptimizer(learning_rate=self.G_lr, beta1=self.beta1, beta2=self.beta2).\
                minimize(G_loss, var_list=G_vars, global_step=global_step)

            # summaries
            # per-step summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G_loss', G_loss),
                tf.summary.scalar('D_loss', D_loss),
                tf.summary.scalar('GP', GP)
            ])

            # sparse-step summary
            tf.summary.image('fake_sample', G, max_outputs=self.FAKE_MAX_OUTPUT)
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
    # DCGAN architecture
    def _discriminator(self, X, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            net = X
            
            with slim.arg_scope([slim.conv2d], kernel_size=[5,5], stride=2, activation_fn=ops.lrelu):
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

            with slim.arg_scope([slim.conv2d_transpose], kernel_size=[5,5], stride=2, activation_fn=tf.nn.relu):
                net = slim.conv2d_transpose(net, 512)
                expected_shape(net, [8, 8, 512])
                net = slim.conv2d_transpose(net, 256)
                expected_shape(net, [16, 16, 256])
                net = slim.conv2d_transpose(net, 128)
                expected_shape(net, [32, 32, 128])
                net = slim.conv2d_transpose(net, 3, activation_fn=tf.nn.tanh, normalizer_fn=None)
                expected_shape(net, [64, 64, 3])

                return net
