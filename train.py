# coding: utf-8

'''
histogram 이나 image 는 heavy-summary 라서 가끔 하고 싶은데, 코드가 지저분해진다. 어떻게 해야 깔끔하게 할 수 있을까?
'''

import tensorflow as tf
from dcgan import DCGAN
from tqdm import tqdm
import numpy as np
import inputpipe as ip
import glob, os
from argparse import ArgumentParser
import utils, config


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--num_epochs', default=20, help='default: 20', type=int)
    parser.add_argument('--batch_size', default=128, help='default: 128', type=int)
    parser.add_argument('--num_threads', default=4, help='# of data read threads (default: 4)', type=int)
    models_str = ' / '.join(config.model_zoo)
    parser.add_argument('--model', help=models_str, required=True) # DRAGAN, CramerGAN
    parser.add_argument('--name', help='default: name=model')
    # more arguments: dataset

    return parser


def input_pipeline(glob_pattern, batch_size, num_threads, num_epochs):
    tfrecords_list = glob.glob(glob_pattern)
    num_examples = utils.num_examples_from_tfrecords(tfrecords_list)
    X = ip.shuffle_batch_join(tfrecords_list, batch_size=batch_size, num_threads=num_threads, num_epochs=num_epochs)
    return X, num_examples


def train(num_epochs, batch_size, n_examples):
    # n_examples = 202599 # same as util.num_examples_from_tfrecords(glob.glob('./data/celebA_tfrecords/*.tfrecord'))
    # 1 epoch = 1583 steps
    print("\n# of examples: {}".format(n_examples))
    print("steps per epoch: {}\n".format(n_examples//batch_size))

    summary_path = os.path.join('./summary/', model.name)
    ckpt_path = os.path.join('./checkpoints', model.name)
    if not os.path.exists(ckpt_path):
        tf.gfile.MakeDirs(ckpt_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer()) # for epochs 

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        summary_writer = tf.summary.FileWriter(summary_path, flush_secs=30, graph=sess.graph)
        total_steps = int(np.ceil(n_examples * num_epochs / float(batch_size))) # total global step
        pbar = tqdm(total=total_steps, desc='global_step')

        saver = tf.train.Saver(max_to_keep=100)
        global_step = 0

        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = sess.run(model.global_step)
            print('\nRestore from {} ... starting global step is {}\n'.format(ckpt.model_checkpoint_path, global_step))
            pbar.update(global_step)
            # 이게 안되면 ckpt.model_checkpoint_path.split('-')[-1] 로 가져와도 됨... 근데뭔가 이렇게 그래프에 박아서 저장한 경우에는 여기서 가져와야 할 것 같다.
            # 그러지 않으면 이렇게 저장한 의미가 없잖아?

        try:
            while not coord.should_stop():
                # 100 step 마다 all_summary_op 를 실행. all_summary_op 에는 heavy op 인 histogram, images 가 포함되어있음.
                summary_op = model.summary_op if global_step % 100 == 0 else model.all_summary_op

                _ = sess.run(model.G_train_op)
                _, global_step, summary = sess.run([model.D_train_op, model.global_step, summary_op])

                summary_writer.add_summary(summary, global_step=global_step)

                if global_step % 10 == 0:
                    pbar.update(10)

                    if global_step % 1000 == 0:
                        saver.save(sess, ckpt_path+'/'+model.name, global_step=global_step)

        except tf.errors.OutOfRangeError:
            print('\nDone -- epoch limit reached\n')
        finally:
            coord.request_stop()

        coord.join(threads)
        summary_writer.close()
        pbar.close()


if __name__ == "__main__":
    parser = build_parser()
    FLAGS = parser.parse_args()
    FLAGS.model = FLAGS.model.upper()
    if FLAGS.name is None:
        FLAGS.name = FLAGS.model.lower()
    config.pprint_args(FLAGS)

    # input pipeline
    X, n_examples = input_pipeline('./data/celebA_tfrecords/*.tfrecord', batch_size=FLAGS.batch_size, num_threads=FLAGS.num_threads, num_epochs=FLAGS.num_epochs)
    model = config.get_model(FLAGS.model, FLAGS.name, input_pipe=X)

    train(num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size, n_examples=n_examples)
