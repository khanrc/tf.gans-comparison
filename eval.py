#coding: utf-8
import matplotlib

import tensorflow as tf
from dcgan import DCGAN
import numpy as np
from utils import *
import os, glob
import scipy.misc
slim = tf.contrib.slim

'''
eval 이 좀 애매하다.
방법1) 전체를 다 로드해서 그냥 z를 흘려보내는 방식.
쓸데없이 X를 읽어오긴 하겠지만 되긴 할 듯?
방법2) generator 만 로드하는 방식.
어떻게 하지? => Saver 선언할 때 var_list 를 줄 수 있음. 그걸로 해보자.
'''

def sample_z(shape):
    return np.random.normal(size=shape)


def eval(dir_name='eval'):
    if tf.gfile.Exists(dir_name):
        tf.gfile.DeleteRecursively(dir_name)
    tf.gfile.MkDir(dir_name)

    # training=False => generator 만 생성
    model = DCGAN(training=False, batch_size=None, num_threads=None, num_epochs=None)
    restorer = tf.train.Saver(slim.get_model_variables())
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./checkpoints/')

        z_ = sample_z([16, model.z_dim])

        for v in ckpt.all_model_checkpoint_paths:
            print("Evaluating {} ...".format(v))
            restorer.restore(sess, v)
            global_step = int(v.split('/')[-1].split('-')[-1])
            
            fake_samples = sess.run(model.fake_sample, {model.z: z_})
            

            # inverse transform: [-1, 1] => [0, 1]
            fake_samples = (fake_samples + 1.) / 2.
            merged_samples = merge(fake_samples, size=[4,4])
            fn = "{:0>5d}.png".format(global_step)
            scipy.misc.imsave(os.path.join(dir_name, fn), merged_samples)



'''
하지만 이렇게 말고도 그냥 imagemagick 을 통해 할 수 있다:
$ convert -delay 20 eval/* movie.gif
'''
def to_gif(dir_name='eval'):
    images = []
    for path in glob.glob(os.path.join(dir_name, '*.png')):
        im = scipy.misc.imread(path)
        images.append(im)

    # make_gif(images, dir_name + '/movie.gif', duration=10, true_image=True)
    imageio.mimsave('movie.gif', images, duration=0.2)


if __name__ == "__main__":
    eval()
    # to_gif()
