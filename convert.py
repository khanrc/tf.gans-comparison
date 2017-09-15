# coding: utf-8

import tensorflow as tf
import numpy as np
import scipy.misc
import os
import glob


def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# preproc for celebA
def center_crop(im, output_size):
    output_height, output_width = output_size
    h, w = im.shape[:2]
    if h < output_height and w < output_width:
        raise ValueError("image is small")

    offset_h = int((h - output_height) / 2)
    offset_w = int((w - output_width) / 2)
    return im[offset_h:offset_h+output_height, offset_w:offset_w+output_width, :]


def convert(source_dir, target_dir, crop_size, out_size, exts=[''], num_shards=128, tfrecords_prefix=''):
    if not tf.gfile.Exists(source_dir):
        print('source_dir does not exists')
        return
    
    if tfrecords_prefix and not tfrecords_prefix.endswith('-'):
        tfrecords_prefix += '-'

    if tf.gfile.Exists(target_dir):
        print("{} is Already exists".format(target_dir))
        return
    else:
        tf.gfile.MakeDirs(target_dir)

    # get meta-data
    path_list = []
    for ext in exts:
        pattern = '*.' + ext if ext != '' else '*'
        path = os.path.join(source_dir, pattern)
        path_list.extend(glob.glob(path))

    # shuffle path_list
    np.random.shuffle(path_list)
    num_files = len(path_list)
    num_per_shard = num_files // num_shards # Last shard will have more files

    print('# of files: {}'.format(num_files))
    print('# of shards: {}'.format(num_shards))
    print('# files per shards: {}'.format(num_per_shard))

    # convert to tfrecords
    shard_idx = 0
    writer = None
    for i, path in enumerate(path_list):
        if i % num_per_shard == 0 and shard_idx < num_shards:
            shard_idx += 1
            tfrecord_fn = '{}{:0>4d}-of-{:0>4d}.tfrecord'.format(tfrecords_prefix, shard_idx, num_shards)
            tfrecord_path = os.path.join(target_dir, tfrecord_fn)
            print("Writing {} ...".format(tfrecord_path))
            if shard_idx > 1:
                writer.close()
            writer = tf.python_io.TFRecordWriter(tfrecord_path)

        # mode='RGB' read even grayscale image as RGB shape
        im = scipy.misc.imread(path, mode='RGB')
        # preproc
        try:
            im = center_crop(im, crop_size)
        except Exception as e:
            # print("im_path: {}".format(path))
            # print("im_shape: {}".format(im.shape))
            print("[Exception] {}".format(e))
            continue

        im = scipy.misc.imresize(im, out_size)
        example = tf.train.Example(features=tf.train.Features(feature={
            # "shape": _int64_features(im.shape),
            "image": _bytes_features([im.tostring()])
        }))
        writer.write(example.SerializeToString())

    writer.close()


''' Below function burrowed from https://github.com/fyu/lsun.
Process: LMDB => images => tfrecords
It is more efficient method to skip intermediate images, but that is a little messy job.
The method through images is inefficient but convenient.
'''
def export_images(db_path, out_dir, flat=False, limit=-1):
    print('Exporting {} to {}'.format(db_path, out_dir))
    env = lmdb.open(db_path, map_size=1099511627776, max_readers=100, readonly=True)
    num_images = env.stat()['entries']
    count = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            if not flat:
                image_out_dir = join(out_dir, '/'.join(key[:6]))
            else:
                image_out_dir = out_dir
            if not exists(image_out_dir):
                os.makedirs(image_out_dir)
            image_out_path = join(image_out_dir, key + '.webp')
            with open(image_out_path, 'w') as fp:
                fp.write(val)
            count += 1
            if count == limit:
                break
            if count % 10000 == 0:
                print('{}/{} ...'.format(count, num_images))


if __name__ == "__main__":
    # CelebA
    convert('./data/celebA', './data/celebA_128_tfrecords', crop_size=[128, 128], out_size=[128, 128], 
        exts=['jpg'], num_shards=128, tfrecords_prefix='celebA')

    # LSUN
    # export_images('./tf.gans-comparison/data/lsun/bedroom_val_lmdb/', 
    #     './tf.gans-comparison/data/lsun/bedroom_val_images/', flat=True)
    # convert('./data/lsun/bedroom_train_images', './data/lsun/bedroom_128_tfrecords', crop_size=[128, 128], 
    #     out_size=[128, 128], exts=['webp'], num_shards=128, tfrecords_prefix='lsun_bedroom')
