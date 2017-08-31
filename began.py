# coding: utf-8
import tensorflow as tf
slim = tf.contrib.slim
from utils import expected_shape
import ops
from basemodel import BaseModel


class BEGAN(BaseModel):
    def __init__(self, name, training, image_shape=[64, 64, 3], z_dim=64, gamma=0.5):
        self.gamma = gamma
        super(BEGAN, self).__init__(name=name, training=training, image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        '''build computational graph for training
        '''
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)

            G = self._generator(z)
            # BEGAN 에서는 이 값을 energy 라고 부르지는 않음. EBGAN-concept.
            D_real_energy = self._discriminator(X)
            D_fake_energy = self._discriminator(G, reuse=True)

            k = tf.Variable(0., name='k', trainable=False)
            with tf.variable_scope('D_loss'):
                D_loss = D_real_energy - k * D_fake_energy
            with tf.variable_scope('G_loss'):
                G_loss = D_fake_energy
            with tf.variable_scope('balance'):
                balance = self.gamma*D_real_energy - D_fake_energy
            with tf.variable_scope('M'):
                M = D_real_energy + tf.abs(balance)
            
            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/D/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/G/')

            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/D/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/G/')

            with tf.variable_scope('D_train_op'):
                with tf.control_dependencies(D_update_ops):
                    D_train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5).minimize(D_loss, var_list=D_vars)
            with tf.variable_scope('G_train_op'):
                with tf.control_dependencies(G_update_ops):
                    G_train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5).minimize(G_loss, var_list=G_vars, global_step=global_step)

            # k 는 D_train 에만 관여
            # G_train 할 때에는 D_real 을 계산하지 않아도 됨
            # [!] control_dependencies 아래에는 ops 정의가 들어가야 하는 듯 함.
            # 아래에서 update_k = tf.assign 부분을 밖으로 빼고 그냥 update_k 만 놓으면 control_dependency 가 제대로 연결이 안 됨.
            lambd_k = 0.001
            with tf.control_dependencies([D_train_op]): # should be iterable
                with tf.variable_scope('update_k'):
                    update_k = tf.assign(k, tf.clip_by_value(k + lambd_k * balance, 0., 1.))
            D_train_op = update_k # run op

            # summaries
            # per-step summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G_loss', G_loss),
                tf.summary.scalar('D_loss', D_loss),
                tf.summary.scalar('D_energy/real', D_real_energy),
                tf.summary.scalar('D_energy/fake', D_fake_energy),
                tf.summary.scalar('convergence_measure', M),
                tf.summary.scalar('balance', balance),
                tf.summary.scalar('k', k)
            ])

            # sparse-step summary
            tf.summary.image('fake_sample', G, max_outputs=6)
            # histogram all varibles
            # gradients 도 보면 좋긴 한데 그러면 위에도 수정해야되서 귀찮으니 일단 이렇게
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

            self.all_summary_op = tf.summary.merge_all()

            # accesible points
            self.X = X
            self.z = z
            self.D_train_op = D_train_op
            self.G_train_op = G_train_op
            self.fake_sample = G
            self.global_step = global_step

    # BEGAN 에서는 기본적으론 decoder 를 generator 로 쓰기 때문에 nh (h_dim) 와 z_dim 이 같아야 한다.
    # 단, 꼭 decoder 를 generator 로 쓸 필요는 없고 그 경우 nh 와 z_dim 이 달라도 된다.
    def _encoder(self, X, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):
            nf = 64
            nh = self.z_dim

            with slim.arg_scope([slim.conv2d], kernel_size=[3,3], padding='SAME', activation_fn=tf.nn.elu):
                net = slim.conv2d(X, nf)

                net = slim.conv2d(net, nf)
                net = slim.conv2d(net, nf)
                net = slim.conv2d(net, nf*2, stride=2) # 32x32

                net = slim.conv2d(net, nf*2)
                net = slim.conv2d(net, nf*2)
                net = slim.conv2d(net, nf*3, stride=2) # 16x16

                net = slim.conv2d(net, nf*3)
                net = slim.conv2d(net, nf*3)
                net = slim.conv2d(net, nf*4, stride=2) # 8x8

                net = slim.conv2d(net, nf*4)
                net = slim.conv2d(net, nf*4)
                net = slim.conv2d(net, nf*4)

            net = slim.flatten(net)
            h = slim.fully_connected(net, nh, activation_fn=None)

            return h

    def _decoder(self, h, reuse=False):
        with tf.variable_scope('decoder', reuse=reuse):
            nf = 64
            nh = self.z_dim

            h0 = slim.fully_connected(h, 8*8*nf, activation_fn=None) # h0
            net = tf.reshape(h0, [-1, 8, 8, nf])

            with slim.arg_scope([slim.conv2d], kernel_size=[3,3], padding='SAME', activation_fn=tf.nn.elu):
                net = slim.conv2d(net, nf)
                net = slim.conv2d(net, nf)
                net = tf.image.resize_nearest_neighbor(net, [16, 16]) # upsampling

                net = slim.conv2d(net, nf)
                net = slim.conv2d(net, nf)
                net = tf.image.resize_nearest_neighbor(net, [32, 32])

                net = slim.conv2d(net, nf)
                net = slim.conv2d(net, nf)
                net = tf.image.resize_nearest_neighbor(net, [64, 64])

                net = slim.conv2d(net, nf)
                net = slim.conv2d(net, nf)

                net = slim.conv2d(net, 3, activation_fn=None)

            return net

    def _discriminator(self, X, reuse=False):
        with tf.variable_scope('D', reuse=reuse):
            h = self._encoder(X, reuse=reuse)
            x_recon = self._decoder(h, reuse=reuse)

            energy = tf.abs(X-x_recon)
            energy = tf.reduce_mean(energy)

            return energy

    def _generator(self, z, reuse=False):
        with tf.variable_scope('G', reuse=reuse):
            x_fake = self._decoder(z, reuse=reuse)

            return x_fake
