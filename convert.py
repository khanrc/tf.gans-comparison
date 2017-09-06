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
def center_crop(im, output_height, output_width):
    h, w = im.shape[:2]
    if h < output_height and w < output_width:
        raise ValueError("image is small")

    offset_h = int((h - output_height) / 2)
    offset_w = int((w - output_width) / 2)
    return im[offset_h:offset_h+output_height, offset_w:offset_w+output_width, :]


def convert(source_dir, target_dir, exts=['jpg'], num_shards=128, tfrecords_prefix=''):
	if not tf.gfile.Exists(source_dir):
		print('source_dir does not exists')
		return
	
	if tfrecords_prefix and not tfrecords_prefix.endswith('-'):
		tfrecords_prefix += '-'

	if tf.gfile.Exists(target_dir):
		# print("{} is Already exists".format(target_dir))
		# return
		pass
	else:
		tf.gfile.MakeDirs(target_dir)

	# get meta-data
	path_list = []
	for ext in exts:
		pattern = '*.' + ext if ext != '' else '*'
		path = os.path.join(source_dir, pattern)
		print path
		path_list.extend(glob.glob(path))

	# shuffle path_list
	np.random.shuffle(path_list)
	num_files = len(path_list)
	num_per_shard = num_files // num_shards # 마지막 샤드는 더 많음

	print('# of files: {}'.format(num_files))
	print('# of shards: {}'.format(num_shards))
	print('# files per shards: {}'.format(num_per_shard))

	# convert to tfrecords
	shard_idx = 0
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
			im = center_crop(im)
		except Exception, e:
			# print("im_path: {}".format(path))
			# print("im_shape: {}".format(im.shape))
			print("[Exception] {}".format(e))
			continue

		im = scipy.misc.imresize(im, [64, 64])
		example = tf.train.Example(features=tf.train.Features(feature={
			"shape": _int64_features(im.shape),
			"image": _bytes_features([im.tostring()])
		}))
		writer.write(example.SerializeToString())

	writer.close()

if __name__ == "__main__":
	convert('./data/celebA', './data/celebA_tfrecords', exts=['jpg'], num_shards=128, tfrecords_prefix='celebA')

