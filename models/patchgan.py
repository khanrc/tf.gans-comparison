# coding: utf-8
import tensorflow as tf
slim = tf.contrib.slim
from utils import expected_shape
import ops
from basemodel import BaseModel

'''
PatchGAN is not from published paper, but from `CycleGAN` and 
`Learning from Simulated and Unsupervised Images through Adversarial Training`.
This method use discriminator for local patches and final D_loss is calculated by mean of losses.
In actual implementation, just use logits from last conv layer in discriminator. 
The each logit has spatial information.

But this method cannot be applied directly in the GAN problem. In the above two cases, generator has not only 
discriminator loss, but also pixel-wise loss which maintains overall structure. So, local discriminator loss 
is sufficient for generator in those cases.

It means that we need one more loss for maintaining overall structure addition to local discriminator loss.
In this project, I experimented original discriminator loss for maintaining overall structure.

loss is weighted mean from overall loss and local loss:
loss = (overall_loss + alpha*local_loss) / (1. + alpha) 
'''

class PatchGAN(BaseModel):
    def __init__(self, name, training, D_lr=2e-4, G_lr=1e-3, image_shape=[64, 64, 3], z_dim=100, alpha=1.0):
        self.alpha = alpha 
        self.beta1 = 0.5
        super(PatchGAN, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr, 
            image_shape=image_shape, z_dim=z_dim)

    def _loss_weighted_mean(self, overall, local):
        return (overall + self.alpha*local) / (1. + self.alpha)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)

            G = self._generator(z)
            D_real_overall_logits, D_real_local_logits = self._discriminator(X)
            D_fake_overall_logits, D_fake_local_logits = self._discriminator(G, reuse=True)

            G_loss_overall = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_fake_overall_logits), logits=D_fake_overall_logits)
            G_loss_local = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_fake_local_logits), logits=D_fake_local_logits)
            G_loss = self._loss_weighted_mean(G_loss_overall, G_loss_local)

            D_loss_real_overall = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_real_overall_logits), logits=D_real_overall_logits)
            D_loss_real_local = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_real_local_logits), logits=D_real_local_logits)
            D_loss_real = self._loss_weighted_mean(D_loss_real_overall, D_loss_real_local)

            D_loss_fake_overall = tf.losses.sigmoid_cross_entropy(tf.zeros_like(D_fake_overall_logits), logits=D_fake_overall_logits)
            D_loss_fake_local = tf.losses.sigmoid_cross_entropy(tf.zeros_like(D_fake_local_logits), logits=D_fake_local_logits)
            D_loss_fake = self._loss_weighted_mean(D_loss_fake_overall, D_loss_fake_local)
            D_loss = D_loss_real + D_loss_fake

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/D/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/G/')

            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/D/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/G/')

            with tf.control_dependencies(D_update_ops):
                D_train_op = tf.train.AdamOptimizer(learning_rate=self.D_lr, beta1=self.beta1).\
                    minimize(D_loss, var_list=D_vars)
            with tf.control_dependencies(G_update_ops):
                # 1e-3 lr for G shows better results
                G_train_op = tf.train.AdamOptimizer(learning_rate=self.G_lr, beta1=self.beta1).\
                    minimize(G_loss, var_list=G_vars, global_step=global_step)

            # summaries
            # per-step summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G_loss', G_loss),
                tf.summary.scalar('G_loss/overall', G_loss_overall),
                tf.summary.scalar('G_loss/local', G_loss_local),
                tf.summary.scalar('D_loss', D_loss),
                tf.summary.scalar('D_loss/real', D_loss_real),
                tf.summary.scalar('D_loss/real/overall', D_loss_real_overall),
                tf.summary.scalar('D_loss/real/local', D_loss_real_local),
                tf.summary.scalar('D_loss/fake', D_loss_fake),
                tf.summary.scalar('D_loss/fake/overall', D_loss_fake_overall),
                tf.summary.scalar('D_loss/fake/local', D_loss_fake_local)
            ])

            # sparse-step summary
            tf.summary.image('fake_sample', G, max_outputs=self.FAKE_MAX_OUTPUT)
            # tf.summary.histogram('real_probs', D_real_prob)
            # tf.summary.histogram('fake_probs', D_fake_prob)
            self.all_summary_op = tf.summary.merge_all()

            # accesible points
            self.X = X
            self.z = z
            self.D_train_op = D_train_op
            self.G_train_op = G_train_op
            self.fake_sample = G
            self.global_step = global_step


    # DCGAN based
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

                overall = slim.conv2d(net, 512)
                expected_shape(overall, [4, 4, 512])

            # local logits
            local_logits = slim.conv2d(net, 1, kernel_size=[3,3], stride=1, activation_fn=None, normalizer_fn=None)
            # prob = tf.sigmoid(logits)

            # overall logits - same as DCGAN
            overall = slim.flatten(overall)
            overall_logits = slim.fully_connected(overall, 1, activation_fn=None)

            return overall_logits, local_logits


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
