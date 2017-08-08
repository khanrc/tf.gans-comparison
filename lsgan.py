# coding: utf-8
import tensorflow as tf
slim = tf.contrib.slim
from utils import expected_shape
import ops
from basemodel import BaseModel


class LSGAN(BaseModel):
    def __init__(self, input_pipe, z_dim=100, name='lsgan5', a=0., b=1., c=1.):
        '''
        training mode: input_pipe = input pipeline
        generation mode: input_pipe = None

        --- LSGAN ---
        a: fake label
        b: real label
        c: real label for G (G가 D를 속이고자 하는 값 - 보통은 real label)

        Pearson chi-square divergence: a=-1, b=1, c=0.
        Intuitive (real label 1, fake label 0): a=0, b=c=1.
        check - 현재 D 의 output 이 sigmoid 이므로, a/b/c 값에 맞게 scale 되어야 함
        '''
        self.a = a
        self.b = b
        self.c = c
        super(LSGAN, self).__init__(input_pipe=input_pipe, z_dim=z_dim, name=name)

    def _range_transform(self, value):
        """a <= c <= b 라고 가정
        value in [0,1] => [a, b]
        """
        size = self.b - self.a
        return value * size + self.a

    def _build_train_graph(self, X):
        '''build computational graph for training
        '''
        with tf.variable_scope(self.name):
            batch_size = tf.shape(X)[0] # tensor. tf.shape 의 return 이 tf.Dimension 이 아니라 그냥 int32네.
            z = tf.random_normal([batch_size, self.z_dim]) # tensor, constant 조합이라도 상관없이 잘 된다.
            global_step = tf.Variable(0, name='global_step', trainable=False)

            G = self._generator(z)
            D_real_prob = self._discriminator(X)
            D_fake_prob = self._discriminator(G, reuse=True)

            # l2_loss: 0.5 * reduce_sum(v ** 2)
            D_loss_real = tf.nn.l2_loss(D_real_prob - 1.) # self.b
            D_loss_fake = tf.nn.l2_loss(D_fake_prob) # self.a
            D_loss =  D_loss_real + D_loss_fake
            G_loss = tf.nn.l2_loss(D_fake_prob - 1.) # self.c
            # G_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_fake_logits), logits=D_fake_logits)
            # D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_real_logits), logits=D_real_logits)
            # D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(D_fake_logits), logits=D_fake_logits)
            # D_loss = D_loss_real + D_loss_fake

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/D/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/G/')

            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/D/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/G/')

            '''
            참고 - 요렇게 슬림으로도 가능함 (summarize_gradients, update_ops)
            D_train_op = slim.learning.create_train_op(D_loss, tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5), update_ops=D_update_ops,
                                                       variables_to_train=D_vars, summarize_gradients=True)
            '''

            # 논문에 따르면 scene (LSUN) data 에 대해서 0.001 로 했다고 되어 있음.
            D_optim = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5)
            G_optim = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5)
            # D_optim = tf.train.RMSPropOptimizer(learning_rate=0.0002)
            # G_optim = tf.train.RMSPropOptimizer(learning_rate=0.002)

            with tf.control_dependencies(D_update_ops):
                D_grads = D_optim.compute_gradients(D_loss, var_list=D_vars)

            with tf.control_dependencies(G_update_ops):
                G_grads = G_optim.compute_gradients(G_loss, var_list=G_vars)

            # # histogram all varibles
            # for var in tf.trainable_variables():
            #     tf.summary.histogram(var.op.name, var)

            # # histogram all gradients
            # for grad, var in D_grads + G_grads:
            #     # if grad is not None:
            #     tf.summary.histogram(var.op.name + '/gradients', grad)

            D_train_op = D_optim.apply_gradients(D_grads, global_step=global_step)
            G_train_op = G_optim.apply_gradients(G_grads)

            
            # with tf.control_dependencies(D_update_ops):
            #     D_train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(D_loss, var_list=D_vars)

            # with tf.control_dependencies(G_update_ops):
            #     G_train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(G_loss, var_list=G_vars, global_step=global_step)

            # summaries
            # per-step summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G/loss', G_loss),
                tf.summary.scalar('D/loss', D_loss),
                tf.summary.scalar('D/loss/real', D_loss_real),
                tf.summary.scalar('D/loss/fake', D_loss_fake)
            ])

            # sparse-step summary
            tf.summary.image('G/fake_sample', G, max_outputs=8)
            tf.summary.histogram('D/real_value', D_real_prob)
            tf.summary.histogram('D/fake_value', D_fake_prob)

            self.all_summary_op = tf.summary.merge_all()

            # accesible points
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
            prob = slim.fully_connected(net, 1, activation_fn=tf.sigmoid)

            return prob


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




    # '''
    # 현재 문제: D 가 무조건 1.0 만 뱉는다. 그래서 real 은 싹다 맞추지만 fake 는 싹다 틀림.
    # 왜 그럴까...?
    # 학습이 전혀 안 되는 것 같은데 gradient 를 체크해 봐야 할듯.
    # '''
    # def _discriminator(self, X, reuse=False):
    #     with tf.variable_scope('D', reuse=reuse):
    #         net = X
            
    #         with slim.arg_scope([slim.conv2d], kernel_size=[5,5], stride=2, padding='SAME', activation_fn=ops.lrelu, 
    #                             normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
    #             net = slim.conv2d(net, 64, normalizer_fn=None)
    #             expected_shape(net, [32, 32, 64])
    #             net = slim.conv2d(net, 128)
    #             expected_shape(net, [16, 16, 128])
    #             net = slim.conv2d(net, 256)
    #             expected_shape(net, [8, 8, 256])
    #             net = slim.conv2d(net, 512)
    #             expected_shape(net, [4, 4, 512])

    #         net = slim.flatten(net)
    #         prob = slim.fully_connected(net, 1, activation_fn=tf.nn.sigmoid)
    #         # d_value = self._range_transform(prob)

    #         return prob

    # # LSGAN 에서는 원래는 LSUN 데이터에 대해 112x112 image 를 생성함
    # # 시작 크기에서 4번 *2 해줘서 총 16 배가 되는 건 동일함. 대신 시작 크기를 7x7 로 함.
    # # 디테일한 구조는 좀 다름:
    # # 1. VGG intuition 을 차용함 - 5x5 => 3x3
    # # 2. fc layer 에도 BN 을 넣음 - 굳이...?
    # # 3. 최대 필터 개수를 512 => 256 으로 줄이고 디테일한 부분이 조금씩 다름
    # def _generator(self, z, reuse=False):
    #     with tf.variable_scope('G', reuse=reuse):
    #         net = z
    #         net = slim.fully_connected(net, 4*4*1024, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params)
    #         net = tf.reshape(net, [-1, 4, 4, 1024])

    #         with slim.arg_scope([slim.conv2d_transpose], kernel_size=[3,3], padding='SAME', activation_fn=tf.nn.relu, 
    #                             normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
    #             net = slim.conv2d_transpose(net, 256, stride=2)
    #             net = slim.conv2d_transpose(net, 256, stride=1)
    #             expected_shape(net, [8, 8, 256])
    #             net = slim.conv2d_transpose(net, 256, stride=2)
    #             net = slim.conv2d_transpose(net, 256, stride=1)
    #             expected_shape(net, [16, 16, 256])
    #             net = slim.conv2d_transpose(net, 128, stride=2)
    #             expected_shape(net, [32, 32, 128])
    #             net = slim.conv2d_transpose(net, 64, stride=2)
    #             expected_shape(net, [64, 64, 64])
    #             net = slim.conv2d_transpose(net, 3, stride=1, activation_fn=tf.nn.tanh, normalizer_fn=None)
    #             expected_shape(net, [64, 64, 3])

    #             return net
