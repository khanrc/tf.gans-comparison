# coding: utf-8
import tensorflow as tf
slim = tf.contrib.slim
from utils import expected_shape
import ops
from basemodel import BaseModel

'''
WGAN:
WD = max_f [ Ex[f(x)] - Ez[f(g(z))] ] where f has K-Lipschitz constraint
J = min WD (G_loss)

+ GP:
Instead of weight clipping, WGAN-GP proposed gradient penalty.
'''

class WGAN_GP(BaseModel):
    def __init__(self, name, training, D_lr=1e-4, G_lr=1e-4, image_shape=[64, 64, 3], z_dim=100):
        self.beta1 = 0.0
        self.beta2 = 0.9
        self.ld = 10. # lambda
        self.n_critic = 5
        super(WGAN_GP, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr, 
            image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)

            # `critic` named from wgan (wgan-gp use the term `discriminator` rather than `critic`)
            G = self._generator(z)
            C_real = self._critic(X)
            C_fake = self._critic(G, reuse=True)

            W_dist = tf.reduce_mean(C_real - C_fake)
            C_loss = -W_dist
            G_loss = tf.reduce_mean(-C_fake)

            # Gradient Penalty (GP)
            eps = tf.random_uniform(shape=[tf.shape(X)[0], 1, 1, 1], minval=0., maxval=1.)
            x_hat = eps*X + (1.-eps)*G 
            C_xhat = self._critic(x_hat, reuse=True)
            C_xhat_grad = tf.gradients(C_xhat, x_hat)[0] # gradient of D(x_hat)
            C_xhat_grad_norm = tf.norm(slim.flatten(C_xhat_grad), axis=1)  # l2 norm
            GP = self.ld * tf.reduce_mean(tf.square(C_xhat_grad_norm - 1.))
            C_loss += GP

            C_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/critic/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/generator/')

            C_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/critic/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/generator/')

            n_critic = 5
            lr = 1e-4
            with tf.control_dependencies(C_update_ops):
                C_train_op = tf.train.AdamOptimizer(learning_rate=self.D_lr*n_critic, beta1=self.beta1, beta2=self.beta2).\
                    minimize(C_loss, var_list=C_vars)
            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.AdamOptimizer(learning_rate=self.G_lr, beta1=self.beta1, beta2=self.beta2).\
                    minimize(G_loss, var_list=G_vars, global_step=global_step)

            # summaries
            # per-step summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G_loss', G_loss),
                tf.summary.scalar('C_loss', C_loss),
                tf.summary.scalar('W_dist', W_dist),
                tf.summary.scalar('GP', GP)
            ])

            # sparse-step summary
            tf.summary.image('fake_sample', G, max_outputs=self.FAKE_MAX_OUTPUT)
            # tf.summary.histogram('real_probs', D_real_prob)
            # tf.summary.histogram('fake_probs', D_fake_prob)
            self.all_summary_op = tf.summary.merge_all()

            # accesible points
            self.X = X
            self.z = z
            self.D_train_op = C_train_op # train.py 와의 accesibility 를 위해... 흠... 구린데...
            self.G_train_op = G_train_op
            self.fake_sample = G
            self.global_step = global_step

    def _critic(self, X, reuse=False):
        return self._good_critic(X, reuse)

    def _generator(self, z, reuse=False):
        return self._good_generator(z, reuse)

    def _dcgan_critic(self, X, reuse=False):
        '''
        K-Lipschitz function.
        WGAN-GP does not use critic in batch norm.
        '''
        with tf.variable_scope('critic', reuse=reuse):
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
            net = slim.fully_connected(net, 1, activation_fn=None)

            return net

    def _dcgan_generator(self, z, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            net = z
            net = slim.fully_connected(net, 4*4*1024, activation_fn=tf.nn.relu)
            net = tf.reshape(net, [-1, 4, 4, 1024])

            with slim.arg_scope([slim.conv2d_transpose], kernel_size=[5,5], stride=2, activation_fn=tf.nn.relu, 
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

    '''
    ResNet architecture from appendix C in the paper.
    https://github.com/igul222/improved_wgan_training/blob/master/gan_64x64.py - GoodGenerator / GoodDiscriminator
    layer norm in D, batch norm in G.
    some details are ignored in this implemenation.
    '''
    def _residual_block(self, X, nf_output, resample, kernel_size=[3,3], name='res_block'):
        with tf.variable_scope(name):
            input_shape = X.shape
            nf_input = input_shape[-1]
            if resample == 'down': # Downsample
                shortcut = slim.avg_pool2d(X, [2,2])
                shortcut = slim.conv2d(shortcut, nf_output, kernel_size=[1,1], activation_fn=None) # init xavier

                net = slim.layer_norm(X, activation_fn=tf.nn.relu)
                net = slim.conv2d(net, nf_input, kernel_size=kernel_size, biases_initializer=None) # skip bias
                net = slim.layer_norm(net, activation_fn=tf.nn.relu)
                net = slim.conv2d(net, nf_output, kernel_size=kernel_size)
                net = slim.avg_pool2d(net, [2,2])

                return net + shortcut
            elif resample == 'up': # Upsample
                upsample_shape = map(lambda x: int(x)*2, input_shape[1:3])
                shortcut = tf.image.resize_nearest_neighbor(X, upsample_shape) 
                shortcut = slim.conv2d(shortcut, nf_output, kernel_size=[1,1], activation_fn=None)

                net = slim.batch_norm(X, activation_fn=tf.nn.relu, **self.bn_params)
                net = tf.image.resize_nearest_neighbor(net, upsample_shape) 
                net = slim.conv2d(net, nf_output, kernel_size=kernel_size, biases_initializer=None) # skip bias
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, **self.bn_params)
                net = slim.conv2d(net, nf_output, kernel_size=kernel_size)

                return net + shortcut
            else:
                raise Exception('invalid resample value')

    def _good_generator(self, z, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            nf = 64
            net = slim.fully_connected(z, 4*4*8*nf, activation_fn=None) # 4x4x512
            net = tf.reshape(net, [-1, 4, 4, 8*nf])
            net = self._residual_block(net, 8*nf, resample='up', name='res_block1') # 8x8x512
            net = self._residual_block(net, 4*nf, resample='up', name='res_block2') # 16x16x256
            net = self._residual_block(net, 2*nf, resample='up', name='res_block3') # 32x32x128
            net = self._residual_block(net, 1*nf, resample='up', name='res_block4') # 64x64x64
            expected_shape(net, [64, 64, 64])
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, **self.bn_params)
            net = slim.conv2d(net, 3, kernel_size=[3,3], activation_fn=tf.nn.tanh)
            expected_shape(net, [64, 64, 3])

            return net

    def _good_critic(self, X, reuse=False):
        with tf.variable_scope('critic', reuse=reuse):
            nf = 64
            net = slim.conv2d(X, nf, [3,3], activation_fn=None) # 64x64x64
            net = self._residual_block(net, 2*nf, resample='down', name='res_block1') # 32x32x128
            net = self._residual_block(net, 4*nf, resample='down', name='res_block2') # 16x16x256
            net = self._residual_block(net, 8*nf, resample='down', name='res_block3') # 8x8x512
            net = self._residual_block(net, 8*nf, resample='down', name='res_block4') # 4x4x512
            expected_shape(net, [4, 4, 512])
            net = slim.flatten(net)
            net = slim.fully_connected(net, 1, activation_fn=None)

            return net
