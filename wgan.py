# coding: utf-8
import tensorflow as tf
slim = tf.contrib.slim
from utils import expected_shape
import ops
from basemodel import BaseModel

'''
based on DCGAN.

WGAN:
WD = max_f [ Ex[f(x)] - Ez[f(g(z))] ] where f has K-Lipschitz constraint
J = min WD (G_loss)
'''

class WGAN(BaseModel):
    def _build_train_graph(self):
        '''build computational graph for training
        '''
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            batch_size = tf.shape(X)[0] # tensor. tf.shape 의 return 이 tf.Dimension 이 아니라 그냥 int32네.
            z = tf.random_normal([batch_size, self.z_dim]) # tensor, constant 조합이라도 상관없이 잘 된다. 
            global_step = tf.Variable(0, name='global_step', trainable=False)

            G = self._generator(z)
            C_real = self._critic(X)
            C_fake = self._critic(G, reuse=True)

            # reduce_mean!
            W_dist = tf.reduce_mean(C_real - C_fake) # maximize
            C_loss = -W_dist # minimize
            G_loss = tf.reduce_mean(-C_fake) # =W_dist. 근데 C_real 이 G 와 상관없으므로 생략해줌. 넣어도 속도는 똑같긴 하겠지?

            C_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/critic/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/generator/')

            C_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/critic/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/generator/')

            # 사실 C 는 n_critic 번 학습시켜줘야 하는데 귀찮아서 그냥 러닝레이트로 때려박음 
            # 학습횟수를 건드리려면 train.py 를 수정해야해서...
            n_critic = 5
            lr = 0.00005
            with tf.control_dependencies(C_update_ops):
                C_train_op = tf.train.RMSPropOptimizer(learning_rate=lr*n_critic).minimize(C_loss, var_list=C_vars)
            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(G_loss, var_list=G_vars, global_step=global_step)

            # weight clipping
            '''
            이 때 웨이트 클리핑은 자동으로 실행이 안 되니 control dependency 를 설정해주거나
            group_op 로 묶어주거나 둘중 하나를 해야 할 듯
            Q. batch_norm parameter 도 clip 해줘야 하나?
            베타는 해 주는게 맞는 것 같은데, 감마는 좀 이상한데...? 
            => 대부분의 구현체들이 감마도 하고 있는 것 같음. 일단 해준다.
            '''
            C_clips = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in C_vars]
            with tf.control_dependencies([C_train_op]): # should be iterable
            	C_train_op = tf.tuple(C_clips)

            # summaries
            # per-step summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G_loss', G_loss),
                tf.summary.scalar('C_loss', C_loss),
                tf.summary.scalar('W_dist', W_dist),
                # tf.summary.scalar('C_loss/real', C_real), # 무슨 의미가 있는지는 모르겠는데 그냥 궁금해서
                # tf.summary.scalar('C_loss/fake', C_fake) # 는 vector 라 안되네
            ])

            # sparse-step summary
            tf.summary.image('fake_sample', G, max_outputs=6)
            # tf.summary.histogram('real_probs', D_real_prob)
            # tf.summary.histogram('fake_probs', D_fake_prob)
            self.all_summary_op = tf.summary.merge_all()

            # accesible points
            self.X = X
            self.D_train_op = C_train_op # train.py 와의 accesibility 를 위해... 흠... 구린데...
            self.G_train_op = G_train_op
            self.fake_sample = G
            self.global_step = global_step

    def _critic(self, X, reuse=False):
    	'''
    	K-Lipschitz function.
    	확인해봐야겠지만 K-Lipschitz function 을 근사하는 함수고, 
    	Lipschitz constraint 는 weight clipping 으로 걸어주니 
    	짐작컨대 그냥 linear 값을 추정하면 될 것 같음.
    	'''
        with tf.variable_scope('critic', reuse=reuse):
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
            net = slim.fully_connected(net, 1, activation_fn=None)

            return net

    def _generator(self, z, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
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
