# coding: utf-8
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
from utils import expected_shape
import ops
from basemodel import BaseModel

# 각 image 간의 square distance
# return: [batch_size, batch_size]
def sd_matrix(a, b):
    batch_size = tf.shape(a)[0]
    a = tf.reshape(a, [batch_size, 1, -1])
    b = tf.reshape(b, [1, batch_size, -1])
    return tf.reduce_sum((b-a)**2, axis=2)


# a, b 간의 square-distance = d 라 하면,
# plummer_kernel = 1/root(d). (element-wise)
# 즉 거리가 가까울수록 값이 커지는 그런 커널임 => 영향력 이라고 해도 될 듯
# 이걸 axis=0 으로 reduce_mean 하면 a 의 모든 b 에 대한 평균 k 값이 나옴.
def plummer_kernel(a, b, kernel_dim, kernel_eps):
    r = sd_matrix(a, b) + kernel_eps**2
    d = kernel_dim-2
    return r**(-d/2.)


def get_potentials(x, y, kernel_dim, kernel_eps):
    '''
    This is alsmost the same `calculate_potential`, but
        px, py = get_potentials(x, y)
    is faster than:
        px = calculate_potential(x, y, x)
        py = calculate_potential(x, y, y)
    because we calculate the cross terms only once.
    '''
    x_fixed = tf.stop_gradient(x)
    y_fixed = tf.stop_gradient(y)
    pk_xx = plummer_kernel(x_fixed, x, kernel_dim, kernel_eps)
    pk_yx = plummer_kernel(y, x, kernel_dim, kernel_eps)
    pk_yy = plummer_kernel(y_fixed, y, kernel_dim, kernel_eps)
    batch_size = tf.shape(x)[0]
    pk_xx = tf.matrix_set_diag(pk_xx, tf.ones(shape=[batch_size], dtype=pk_xx.dtype))
    pk_yy = tf.matrix_set_diag(pk_yy, tf.ones(shape=[batch_size], dtype=pk_yy.dtype))
    kxx = tf.reduce_mean(pk_xx, axis=0)
    kyx = tf.reduce_mean(pk_yx, axis=0)
    kxy = tf.reduce_mean(pk_yx, axis=1)
    kyy = tf.reduce_mean(pk_yy, axis=0)
    pot_x = kxx - kyx
    pot_y = kxy - kyy
    pot_x = tf.reshape(pot_x, [batch_size, -1])
    pot_y = tf.reshape(pot_y, [batch_size, -1])
    return pot_x, pot_y

# x 가 a 에 미치는 영향력 - y 가 a 에 미치는 영향력
# 영향력은 거리가 가까울수록 커짐
#     px, py = get_potentials(x, y)
#     px = calculate_potential(x, y, x) # px: xx 간 영향력 - xy 간 영향력
#     py = calculate_potential(x, y, y) # py: xy 간 영향력 - yy 간 영향력
def calc_potential(x, y, a, kernel_dim, kernel_eps):
    # 이 연산 과정은 backprop 안하려고 stop_grad 하는 듯
    # 왜 a 에는 안해주지?
    x = tf.stop_gradient(x)
    y = tf.stop_gradient(y)
    kxa = tf.reduce_mean(plummer_kernel(x, a, kernel_dim, kernel_eps), axis=0)
    kya = tf.reduce_mean(plummer_kernel(y, a, kernel_dim, kernel_eps), axis=0)
    p = kxa - kya  # 수식이랑 다르게 kxa - kxy 하네? 별 차이는 없을거 같기는 한데...
    p = tf.reshape(p, [-1, 1])
    return p

'''
L2 weight decay, genenrator history 는 사용하지 않음
history 는 논문에는 없느 ㄴ것 같은데...
'''
class CoulombGAN(BaseModel):
    def __init__(self, name, training, D_lr=5e-5, G_lr=1e-4, image_shape=[64, 64, 3], z_dim=32):
        self.beta1 = 0.5
        self.kernel_dim = 3
        self.kernel_eps = 1.
        super(CoulombGAN, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr, 
            image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)

            G = self._generator(z)
            D_real = self._discriminator(X)
            D_fake = self._discriminator(G, reuse=True)

            '''
            D 는 potential 을 예측하는 네트워크. D 와 P 를 맞추려고 한다.
            G 는 D_fake 를 최소화. 즉 fake sample 의 potential 을 최소화하려고 한다.
            fake sample 의 포텐셜을 최소화 한다는 것은 xx간 거리를 최대한 떨어뜨려놓으면서 xy간 거리는 최대한 줄이는 걸 말함.

            P 는 뭘까?
            P(a) = k(a, real) - k(a, fake).
            즉, 
            P(real) = k(real, real) - k(real, fake)
            P(fake) = k(fake, real) - k(fake, fake)
            가 된다.

            그리고 plummer kernel 의 경우 k(a,b) = k(b,a).T 임.
            '''

            # 구현과 논문이 오묘하게 기호가 다름. 잘 다 맞춰보면 맞을거같은데...
            P_fake, P_real = get_potentials(G, X, kernel_dim=self.kernel_dim, kernel_eps=self.kernel_eps)
            D_loss_real = tf.losses.mean_squared_error(D_real, P_real)
            D_loss_fake = tf.losses.mean_squared_error(D_fake, P_fake)
            D_loss = D_loss_real + D_loss_fake
            G_loss = tf.reduce_mean(D_fake) # 정확히는 D_real - D_fake 를 minimize

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
                tf.summary.scalar('G_loss', G_loss),
                tf.summary.scalar('D_loss', D_loss),
                tf.summary.scalar('D_loss/real', D_loss_real),
                tf.summary.scalar('D_loss/fake', D_loss_fake)
            ])

            # sparse-step summary
            tf.summary.image('fake_sample', G, max_outputs=self.FAKE_MAX_OUTPUT)
            self.all_summary_op = tf.summary.merge_all()

            # accesible points
            self.X = X
            self.z = z
            self.D_train_op = D_train_op
            self.G_train_op = G_train_op
            self.fake_sample = G
            self.global_step = global_step

    # Discriminator of CoulombGAN uses double channels of DCGAN
    def _discriminator(self, X, reuse=False):
        with tf.variable_scope('D', reuse=reuse):
            net = X
            
            with slim.arg_scope([slim.conv2d], kernel_size=[5,5], stride=2, padding='SAME', activation_fn=ops.lrelu, 
                normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                net = slim.conv2d(net, 128, normalizer_fn=None)
                net = slim.conv2d(net, 256)
                net = slim.conv2d(net, 512)
                net = slim.conv2d(net, 1024)
                expected_shape(net, [4, 4, 1024])

            net = slim.flatten(net)
            logits = slim.fully_connected(net, 1, activation_fn=None)

            return logits # potential

    def _generator(self, z, reuse=False):
        with tf.variable_scope('G', reuse=reuse):
            net = z
            net = slim.fully_connected(net, 4*4*1024, activation_fn=tf.nn.relu)
            net = tf.reshape(net, [-1, 4, 4, 1024])

            with slim.arg_scope([slim.conv2d_transpose], kernel_size=[5,5], stride=2, padding='SAME', 
                activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                net = slim.conv2d_transpose(net, 512)
                expected_shape(net, [8, 8, 512])
                net = slim.conv2d_transpose(net, 256)
                expected_shape(net, [16, 16, 256])
                net = slim.conv2d_transpose(net, 128)
                expected_shape(net, [32, 32, 128])
                net = slim.conv2d_transpose(net, 3, activation_fn=tf.nn.tanh, normalizer_fn=None)
                expected_shape(net, [64, 64, 3])

                return net
