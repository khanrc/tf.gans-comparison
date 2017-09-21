# coding: utf-8

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import inputpipe as ip
import glob, os, sys
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
    parser.add_argument('--dataset', '-D', help='CelebA / LSUN', required=True)
    parser.add_argument('--ckpt_step', default=5000, help='# of steps for saving checkpoint (default: 5000)', type=int)
    parser.add_argument('--renew', action='store_true', help='train model from scratch - \
        clean saved checkpoints and summaries', default=False)

    return parser


def input_pipeline(glob_pattern, batch_size, num_threads, num_epochs):
    tfrecords_list = glob.glob(glob_pattern)
    # num_examples = utils.num_examples_from_tfrecords(tfrecords_list) # takes too long time for lsun
    X = ip.shuffle_batch_join(tfrecords_list, batch_size=batch_size, num_threads=num_threads, num_epochs=num_epochs)
    return X


def sample_z(shape):
    return np.random.normal(size=shape)


def train(model, dataset, input_op, num_epochs, batch_size, n_examples, ckpt_step, renew=False):
    # n_examples = 202599 # same as util.num_examples_from_tfrecords(glob.glob('./data/celebA_tfrecords/*.tfrecord'))
    # 1 epoch = 1583 steps
    print("\n# of examples: {}".format(n_examples))
    print("steps per epoch: {}\n".format(n_examples//batch_size))

    summary_path = os.path.join('./summary/', dataset, model.name)
    ckpt_path = os.path.join('./checkpoints', dataset, model.name)
    if renew:
        if os.path.exists(summary_path):
            tf.gfile.DeleteRecursively(summary_path)
        if os.path.exists(ckpt_path):
            tf.gfile.DeleteRecursively(ckpt_path)
    if not os.path.exists(ckpt_path):
        tf.gfile.MakeDirs(ckpt_path)

    config = tf.ConfigProto()
    best_gpu = utils.get_best_gpu()
    config.gpu_options.visible_device_list = str(best_gpu) # Works same as CUDA_VISIBLE_DEVICES!
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer()) # for epochs 

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # https://github.com/tensorflow/tensorflow/issues/10972        
        # TensorFlow 1.2 has much bugs for text summary
        # make config_summary before define of summary_writer - bypass bug of tensorboard
        
        # It seems that batch_size should have been contained in the model config ... 
        total_steps = int(np.ceil(n_examples * num_epochs / float(batch_size))) # total global step
        config_list = [
            ('num_epochs', num_epochs),
            ('total_iteration', total_steps),
            ('batch_size', batch_size), 
            ('dataset', dataset)
        ]
        model_config_list = [[k, str(w)] for k, w in sorted(model.args.items()) + config_list]
        model_config_summary_op = tf.summary.text(model.name + '/config', tf.convert_to_tensor(model_config_list), 
            collections=[])
        model_config_summary = sess.run(model_config_summary_op)

        # print to console
        print("\n====== Process info =======")
        print("argv: {}".format(' '.join(sys.argv)))
        print("PID: {}".format(os.getpid()))
        print("====== Model configs ======")
        for k, v in model_config_list:
            print("{}: {}".format(k, v))
        print("===========================\n")

        summary_writer = tf.summary.FileWriter(summary_path, flush_secs=30, graph=sess.graph)
        summary_writer.add_summary(model_config_summary)
        pbar = tqdm(total=total_steps, desc='global_step')
        saver = tf.train.Saver(max_to_keep=9999) # save all checkpoints
        global_step = 0

        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = sess.run(model.global_step)
            print('\n[!] Restore from {} ... starting global step is {}\n'.format(ckpt.model_checkpoint_path, global_step))
            pbar.update(global_step)

        try:
            # If training process was resumed from checkpoints, input pipeline cannot detect
            # when training should stop. So we need `global_step < total_step` condition.
            while not coord.should_stop() and global_step < total_steps:
                # model.all_summary_op contains histogram summary and image summary which are heavy op
                summary_op = model.summary_op if global_step % 100 == 0 else model.all_summary_op

                batch_X = sess.run(input_op)
                batch_z = sample_z([batch_size, model.z_dim])

                _, summary = sess.run([model.D_train_op, summary_op], {model.X: batch_X, model.z: batch_z})
                _, global_step = sess.run([model.G_train_op, model.global_step], {model.z: batch_z})

                summary_writer.add_summary(summary, global_step=global_step)

                if global_step % 10 == 0:
                    pbar.update(10)

                    if global_step % ckpt_step == 0:
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
    FLAGS.dataset = FLAGS.dataset.lower()
    if FLAGS.name is None:
        FLAGS.name = FLAGS.model.lower()
    config.pprint_args(FLAGS)

    # get information for dataset
    dataset_pattern, n_examples = config.get_dataset(FLAGS.dataset)
    # input pipeline
    X = input_pipeline(dataset_pattern, batch_size=FLAGS.batch_size, 
        num_threads=FLAGS.num_threads, num_epochs=FLAGS.num_epochs)

    model = config.get_model(FLAGS.model, FLAGS.name, training=True)
    train(model=model, dataset=FLAGS.dataset, input_op=X, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size, 
        n_examples=n_examples, ckpt_step=FLAGS.ckpt_step, renew=FLAGS.renew)
