# coding: utf-8
'''
EBGAN 논문에는 mse 를 쓰라고 되어 있으나, 실제로 그렇게 써서 그대로 파라메터를 적용해 보면 작동하지 않음
구현체들을 찾아보면 특이한 로스를 쓰는데, 이건 rmse 도 아니고 l2 norm 도 아니고 뭥미...
실제로 mse 를 쓰려면 lr 을 낮춰줘야함: 1e-3 => 1e-4. (참고: pytorch 구현체에서는 2e-4)
* mse: mean(dx**2) = sum(dx**2) / # of dx
* rmse: root(mse)
* l2_norm: root(sum(dx**2))
* energy loss on implementations: root(sum(dx**2)) / # of dx
'''
import tensorflow as tf
slim = tf.contrib.slim
from utils import expected_shape
import ops
from basemodel import BaseModel


class EBGAN(BaseModel):
    def __init__(self, name, training, image_shape=[64, 64, 3], z_dim=100, use_pt_regularizer=False, pt_weight=0.1, margin=20.):
        '''
        pt_weight 만으로도 pt_reg 를 쓰냐 안쓰냐 조정할 수 있지만, 만약 안쓰면서도 pt_loss 를 체크해보고 싶은 경우를 위해
        use_pt_regularizer 를 따로 두었음.

        pt_weight 와 margin 의 default 값은 논문에 나온 celebA 최적값임. pt_weight 은 크게 중요하지 않은 것 같고, margin 은 상당히 데이터에 민감한 듯.
        '''
        self.use_pt_regularizer = use_pt_regularizer
        self.pt_weight = pt_weight
        self.m = margin
        super(EBGAN, self).__init__(name=name, training=training, image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        '''build computational graph for training
        '''
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)

            G = self._generator(z)
            # mse is energy
            D_real_latent, D_real_energy = self._discriminator(X)
            D_fake_latent, D_fake_energy = self._discriminator(G, reuse=True)

            D_fake_hinge = tf.maximum(0., self.m - D_fake_energy) # hinge_loss
            D_loss = D_real_energy + D_fake_hinge
            G_loss = D_fake_energy
            pt_loss = self.pt_weight * self.pt_regularizer(D_fake_latent) # pt_loss
            if self.use_pt_regularizer:
                G_loss += pt_loss

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/D/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/G/')

            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/D/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/G/')

            with tf.control_dependencies(D_update_ops):
                D_train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5).minimize(D_loss, var_list=D_vars)
            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5).minimize(G_loss, var_list=G_vars, global_step=global_step)

            # summaries
            # per-step summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G_loss', G_loss),
                tf.summary.scalar('D_loss', D_loss),
                tf.summary.scalar('pt_loss', pt_loss),
                tf.summary.scalar('D_energy/real', D_real_energy),
                tf.summary.scalar('D_energy/fake', D_fake_energy),
                tf.summary.scalar('D_fake_hinge', D_fake_hinge)
            ])

            # sparse-step summary
            tf.summary.image('fake_sample', G, max_outputs=6)
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
            
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], kernel_size=[4,4], stride=2, padding='SAME', 
                activation_fn=ops.lrelu, normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                # encoder
                net = slim.conv2d(net, 64, normalizer_fn=None) # 32x32
                net = slim.conv2d(net, 128) # 16x16
                net = slim.conv2d(net, 256) # 8x8
                latent = net
                expected_shape(latent, [8, 8, 256])
                # decoder
                net = slim.conv2d_transpose(net, 128) # 16x16
                net = slim.conv2d_transpose(net, 64) # 32x32
                x_recon = slim.conv2d_transpose(net, 3, activation_fn=None, normalizer_fn=None)
                expected_shape(x_recon, [64, 64, 3])
            
            mse = tf.losses.mean_squared_error(X, x_recon) # loss 를 mse 로 계산하므로 sigmoid 를 걸면 안되는 것 같음...

            return latent, mse

    def _generator(self, z, reuse=False):
        with tf.variable_scope('G', reuse=reuse):
            net = z
            net = slim.fully_connected(net, 4*4*1024, activation_fn=tf.nn.relu)
            net = tf.reshape(net, [-1, 4, 4, 1024])

            with slim.arg_scope([slim.conv2d_transpose], kernel_size=[4,4], stride=2, padding='SAME', activation_fn=tf.nn.relu, 
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

    # lf: latent features
    def pt_regularizer(self, lf):
        eps = 1e-8 # epsilon for numerical stability
        lf = slim.flatten(lf)
        # l2_norm = tf.sqrt(tf.reduce_sum(tf.square(lf), axis=1, keep_dims=True))
        l2_norm = tf.norm(lf, axis=1, keep_dims=True)
        expected_shape(l2_norm, [1])
        unit_lf = lf / (l2_norm + eps) # this is unit vector?
        cos_sim = tf.square(tf.matmul(unit_lf, unit_lf, transpose_b=True)) # [N, h_dim] x [h_dim, N] = [N, N]
        N = tf.cast(tf.shape(lf)[0], tf.float32) # batch_size
        pt_loss = (tf.reduce_sum(cos_sim)-N) / (N*(N-1))
        return pt_loss