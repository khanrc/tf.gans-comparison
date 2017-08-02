# coding: utf-8

import tensorflow as tf
import scipy.misc
import os
import glob


def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# preproc for celebA
# burrowed from https://github.com/nmhkahn/DCGAN-tensorflow-slim/blob/master/dataset/download_and_convert.py
def center_crop(im, output_height, output_width):
    h, w = im.shape[:2]
    if h < output_height and w < output_width:
        raise ValueError("image is small")

    offset_h = int((h - output_height) / 2)
    offset_w = int((w - output_width) / 2)
    return im[offset_h:offset_h+output_height, offset_w:offset_w+output_width, :]


def convert(source_dir, target_dir, exts=['jpg'], num_shards=128, tfrecords_prefix=''):
	if not tf.gfile.Exists(source_dir):
		return

	if tf.gfile.Exists(target_dir):
		# print("{} is Already exists".format(target_dir))
		# return
		pass
	else:
		tf.gfile.MakeDirs(target_dir)

	# get meta-data
	path_list = []
	for ext in exts:
		path = os.path.join(source_dir, '*.' + ext)
		print path
		path_list.extend(glob.glob(path))

	num_files = len(path_list)
	num_per_shard = num_files // num_shards # 마지막 샤드는 더 많음

	print '# of files: {}'.format(num_files)
	print '# of shards: {}'.format(num_shards)
	print '# files per shards: {}'.format(num_per_shard)

	# convert to tfrecords
	i = 0
	for shard_idx in range(num_shards):
		tfrecord_fn = '{}{:0>4d}-of-{:0>4d}.tfrecords'.format(tfrecords_prefix, shard_idx+1, num_shards)
		tfrecord_path = os.path.join(target_dir, tfrecord_fn)
		print("Writing {} ...".format(tfrecord_path))
		with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
			while True:
				im_path = path_list[i]
				im = scipy.misc.imread(im_path)
				# preproc
				try:
					im = center_crop(im, 128, 128)
				except ValueError, e:
					print("[Exception] {}".format(e))
					continue
				im = scipy.misc.imresize(im, [64, 64])

				example = tf.train.Example(features=tf.train.Features(feature={
					"shape": _int64_features(im.shape),
					"image": _bytes_features([im.tostring()])
				}))

				writer.write(example.SerializeToString())

				i += 1
				if shard_idx != num_shards-1:
					# 마지막 샤드가 아니면 num_per_shard 에서 끝냄
					if i % num_per_shard == 0:
						break

				# 끝까지 다 돌면 끝냄 
				if i == num_files:
					break

		# exception 에 많이 걸리면 shard 개수를 다 못채우고 끝날수도 있음. 여기서 그걸 잡아준다.
		if i == num_files:
			break

if __name__ == "__main__":
	convert('./data/celebA', './data/celebA_tfrecords', exts=['jpg'], num_shards=128)

