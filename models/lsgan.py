# coding: utf-8
import tensorflow as tf
slim = tf.contrib.slim
from utils import expected_shape
import ops
from basemodel import BaseModel


class LSGAN(BaseModel):
    def __init__(self, name, training, D_lr=1e-3, G_lr=1e-3, image_shape=[64, 64, 3], z_dim=1024, a=0., b=1., c=1.):
        '''
        a: fake label
        b: real label
        c: real label for G (The value that G wants to deceive D - intuitively same as real label b) 

        Pearson chi-square divergence: a=-1, b=1, c=0.
        Intuitive (real label 1, fake label 0): a=0, b=c=1.
        '''
        self.a = a
        self.b = b
        self.c = c
        self.beta1 = 0.5
        super(LSGAN, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr, 
            image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)

            G = self._generator(z)
            D_real = self._discriminator(X)
            D_fake = self._discriminator(G, reuse=True)

            D_loss_real = 0.5 * tf.reduce_mean(tf.square(D_real - self.b)) # self.b
            D_loss_fake = 0.5 * tf.reduce_mean(tf.square(D_fake - self.a)) # self.a
            D_loss = D_loss_real + D_loss_fake
            G_loss = 0.5 * tf.reduce_mean(tf.square(D_fake - self.c)) # self.c

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/D/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/G/')

            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/D/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/G/')
            
            with tf.control_dependencies(D_update_ops):
                D_train_op = tf.train.AdamOptimizer(learning_rate=self.D_lr, beta1=self.beta1).\
                    minimize(D_loss, var_list=D_vars)

            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.AdamOptimizer(learning_rate=self.G_lr, beta1=self.beta1).\
                    minimize(G_loss, var_list=G_vars, global_step=global_step)

            # summaries
            # per-step summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G/loss', G_loss),
                tf.summary.scalar('D/loss', D_loss),
                tf.summary.scalar('D/loss/real', D_loss_real),
                tf.summary.scalar('D/loss/fake', D_loss_fake)
            ])

            # sparse-step summary
            tf.summary.image('G/fake_sample', G, max_outputs=self.FAKE_MAX_OUTPUT)
            tf.summary.histogram('D/real_value', D_real)
            tf.summary.histogram('D/fake_value', D_fake)

            self.all_summary_op = tf.summary.merge_all()

            # accesible points
            self.X = X
            self.z = z
            self.D_train_op = D_train_op
            self.G_train_op = G_train_op
            self.fake_sample = G
            self.global_step = global_step

    def _discriminator(self, X, reuse=False):
        with tf.variable_scope('D', reuse=reuse):
            net = X
            
            with slim.arg_scope([slim.conv2d], kernel_size=[5,5], stride=2, padding='SAME', activation_fn=ops.lrelu, 
                normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
            
                net = slim.conv2d(net, 64, normalizer_fn=None)
                expected_shape(net, [32, 32, 64])
                net = slim.conv2d(net, 128)
                expected_shape(net, [16, 16, 128])
                net = slim.conv2d(net, 256)
                expected_shape(net, [8, 8, 256])
                net = slim.conv2d(net, 512)
                expected_shape(net, [4, 4, 512])

            net = slim.flatten(net)
            d_value = slim.fully_connected(net, 1, activation_fn=None)

            return d_value

    # Originally, LSGAN used 112x112 LSUN images
    # We used 64x64 CelebA images
    def _generator(self, z, reuse=False):
        with tf.variable_scope('G', reuse=reuse):
            net = z
            net = slim.fully_connected(net, 4*4*256, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, 
                normalizer_params=self.bn_params)
            net = tf.reshape(net, [-1, 4, 4, 256])

            with slim.arg_scope([slim.conv2d_transpose], kernel_size=[3,3], padding='SAME', activation_fn=tf.nn.relu, 
                normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):

                net = slim.conv2d_transpose(net, 256, stride=2)
                net = slim.conv2d_transpose(net, 256, stride=1)
                expected_shape(net, [8, 8, 256])
                net = slim.conv2d_transpose(net, 256, stride=2)
                net = slim.conv2d_transpose(net, 256, stride=1)
                expected_shape(net, [16, 16, 256])
                net = slim.conv2d_transpose(net, 128, stride=2)
                expected_shape(net, [32, 32, 128])
                net = slim.conv2d_transpose(net, 64, stride=2)
                expected_shape(net, [64, 64, 64])
                net = slim.conv2d_transpose(net, 3, stride=1, activation_fn=tf.nn.tanh, normalizer_fn=None)
                expected_shape(net, [64, 64, 3])

                return net
