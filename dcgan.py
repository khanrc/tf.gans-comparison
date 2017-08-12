# coding: utf-8
import tensorflow as tf
slim = tf.contrib.slim
from utils import expected_shape
import ops
from basemodel import BaseModel

'''
일단 MNIST 는 무시하자... 귀찮다.
MNIST [28, 28, 1] - not sure
D: [28, 28, 1] => [14, 14, 64] => [7, 7, 128] => [1]
G: [100] => [4*4*512] => [14, 14, 64] => [28, 28, 1]

CelebA, LSUN [64, 64, 3]
D: [64, 64, 3] => [32, 32, 64] => [16, 16, 128] => [8, 8, 256] => [4, 4, 512] => [1]
G: [100] => [4*4*1024] => [8, 8, 512] => [16, 16, 256] => [32, 32, 128] => [64, 64, 3]

--- hyperparams
adam => SGD
batch size 128
init - normal dist + stddev 0.02
'''

class DCGAN(BaseModel):
    def _build_train_graph(self):
        '''build computational graph for training
        '''
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            batch_size = tf.shape(X)[0] # tensor. tf.shape 의 return 이 tf.Dimension 이 아니라 그냥 int32네.
            z = tf.random_normal([batch_size, self.z_dim]) # tensor, constant 조합이라도 상관없이 잘 된다. 
            global_step = tf.Variable(0, name='global_step', trainable=False)

            G = self._generator(z)
            D_real_prob, D_real_logits = self._discriminator(X)
            D_fake_prob, D_fake_logits = self._discriminator(G, reuse=True)

            G_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_fake_logits), logits=D_fake_logits)
            D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_real_logits), logits=D_real_logits)
            D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(D_fake_logits), logits=D_fake_logits)
            D_loss = D_loss_real + D_loss_fake

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/D/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/G/')

            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/D/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/G/')

            with tf.control_dependencies(D_update_ops):
                D_train_op = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(D_loss, var_list=D_vars)
            with tf.control_dependencies(G_update_ops):
                # learning rate 0.001 => InfoGAN style
                G_train_op = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(G_loss, var_list=G_vars, global_step=global_step)
                # minimize 에서 자동으로 global_step 을 업데이트해줌

            # summaries
            # per-step summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G_loss', G_loss),
                tf.summary.scalar('D_loss', D_loss),
                tf.summary.scalar('D_loss/real', D_loss_real),
                tf.summary.scalar('D_loss/fake', D_loss_fake)
            ])

            # sparse-step summary
            tf.summary.image('fake_sample', G, max_outputs=8)
            tf.summary.histogram('real_probs', D_real_prob)
            tf.summary.histogram('fake_probs', D_fake_prob)
            self.all_summary_op = tf.summary.merge_all()

            # accesible points
            self.X = X
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
            logits = slim.fully_connected(net, 1, activation_fn=None)
            prob = tf.sigmoid(logits)

            return prob, logits


    def _generator(self, z, reuse=False):
        with tf.variable_scope('G', reuse=reuse):
            net = z
            net = slim.fully_connected(net, 4*4*1024, activation_fn=tf.nn.relu)
            net = tf.reshape(net, [-1, 4, 4, 1024])

            with slim.arg_scope([slim.conv2d_transpose], kernel_size=[5,5], stride=2, padding='SAME', activation_fn=tf.nn.relu, 
                normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                net = slim.conv2d_transpose(net, 512)
                expected_shape(net, [8, 8, 512])
                net = slim.conv2d_transpose(net, 256)
                expected_shape(net, [16, 16, 256])
                net = slim.conv2d_transpose(net, 128)
                expected_shape(net, [32, 32, 128])
                net = slim.conv2d_transpose(net, 3, activation_fn=tf.nn.tanh, normalizer_fn=None)
                expected_shape(net, [64, 64, 3])

                return net
