#coding: utf-8
import tensorflow as tf
import numpy as np
import utils
import config
import os, glob
import scipy.misc
from argparse import ArgumentParser
slim = tf.contrib.slim


def build_parser():
    parser = ArgumentParser()
    models_str = ' / '.join(config.model_zoo)
    parser.add_argument('--model', help=models_str, required=True) 
    parser.add_argument('--name', help='default: name=model')
    parser.add_argument('--dataset', '-D', help='CelebA / LSUN', required=True)
    parser.add_argument('--sample_size', '-N', help='# of samples. It should be a square number. (default: 16)',
        default=16, type=int)

    return parser


def sample_z(shape):
    return np.random.normal(size=shape)


def get_all_checkpoints(ckpt_dir, force=False):
    '''
    When the learning is interrupted and resumed, all checkpoints can not be fetched with get_checkpoint_state 
    (The checkpoint state is rewritten from the point of resume). 
    This function fetch all checkpoints forcely when arguments force=True.
    '''

    if force:
        ckpts = os.listdir(ckpt_dir) # get all fns
        ckpts = map(lambda p: os.path.splitext(p)[0], ckpts) # del ext
        ckpts = set(ckpts) # unique
        ckpts = filter(lambda x: x.split('-')[-1].isdigit(), ckpts) # filter non-ckpt
        ckpts = sorted(ckpts, key=lambda x: int(x.split('-')[-1])) # sort
        ckpts = map(lambda x: os.path.join(ckpt_dir, x), ckpts) # fn => path
    else:
        ckpts = tf.train.get_checkpoint_state(ckpt_dir).all_model_checkpoint_paths
    
    return ckpts


def eval(model, name, dataset, sample_shape=[4,4], load_all_ckpt=True):
    if name == None:
        name = model.name
    dir_name = os.path.join('eval', dataset, name)
    if tf.gfile.Exists(dir_name):
        tf.gfile.DeleteRecursively(dir_name)
    tf.gfile.MakeDirs(dir_name)

    restorer = tf.train.Saver(slim.get_model_variables())

    config = tf.ConfigProto()
    best_gpu = utils.get_best_gpu()
    config.gpu_options.visible_device_list = str(best_gpu)
    with tf.Session(config=config) as sess:
        ckpt_path = os.path.join('checkpoints', dataset, name)
        ckpts = get_all_checkpoints(ckpt_path, force=load_all_ckpt)
        size = sample_shape[0] * sample_shape[1]

        z_ = sample_z([size, model.z_dim])

        for v in ckpts:
            print("Evaluating {} ...".format(v))
            restorer.restore(sess, v)
            global_step = int(v.split('/')[-1].split('-')[-1])
            
            fake_samples = sess.run(model.fake_sample, {model.z: z_})

            # inverse transform: [-1, 1] => [0, 1]
            fake_samples = (fake_samples + 1.) / 2.
            merged_samples = utils.merge(fake_samples, size=sample_shape)
            fn = "{:0>6d}.png".format(global_step)
            scipy.misc.imsave(os.path.join(dir_name, fn), merged_samples)


'''
You can create a gif movie through imagemagick on the commandline:
$ convert -delay 20 eval/* movie.gif
'''
# def to_gif(dir_name='eval'):
#     images = []
#     for path in glob.glob(os.path.join(dir_name, '*.png')):
#         im = scipy.misc.imread(path)
#         images.append(im)

#     # make_gif(images, dir_name + '/movie.gif', duration=10, true_image=True)
#     imageio.mimsave('movie.gif', images, duration=0.2)


if __name__ == "__main__":
    parser = build_parser()
    FLAGS = parser.parse_args()
    FLAGS.model = FLAGS.model.upper()
    FLAGS.dataset = FLAGS.dataset.lower()
    if FLAGS.name is None:
        FLAGS.name = FLAGS.model.lower()
    config.pprint_args(FLAGS)

    N = FLAGS.sample_size**0.5
    assert N == int(N), 'sample size should be a square number'

    # training=False => build generator only
    model = config.get_model(FLAGS.model, FLAGS.name, training=False)
    eval(model, dataset=FLAGS.dataset, name=FLAGS.name, sample_shape=[int(N),int(N)], load_all_ckpt=True)
