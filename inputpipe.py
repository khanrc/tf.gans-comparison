# coding: utf-8
import tensorflow as tf


def read_parse_preproc(filename_queue):
    ''' read, parse, and preproc single example. '''
    with tf.variable_scope('read_parse_preproc'):
        reader = tf.TFRecordReader()
        key, records = reader.read(filename_queue)
        
        # parse records
        features = tf.parse_single_example(
            records,
            features={
                "image": tf.FixedLenFeature([], tf.string)
            }
        )

        image = tf.decode_raw(features["image"], tf.uint8)
        image = tf.reshape(image, [128, 128, 3]) # The image_shape must be explicitly specified
        image = tf.image.resize_images(image, [64, 64])
        image = tf.cast(image, tf.float32)
        image = image / 127.5 - 1.0 # preproc - normalize
        
        return [image]


# https://www.tensorflow.org/programmers_guide/reading_data
def get_batch(tfrecords_list, batch_size, shuffle=False, num_threads=1, min_after_dequeue=None, num_epochs=None):
    name = "batch" if not shuffle else "shuffle_batch"
    with tf.variable_scope(name):
        filename_queue = tf.train.string_input_producer(tfrecords_list, shuffle=shuffle, num_epochs=num_epochs)
        data_point = read_parse_preproc(filename_queue)
        
        if min_after_dequeue is None:
            min_after_dequeue = batch_size * 10
        capacity = min_after_dequeue + 3*batch_size
        if shuffle:
            batch = tf.train.shuffle_batch(data_point, batch_size=batch_size, capacity=capacity, 
                min_after_dequeue=min_after_dequeue, num_threads=num_threads, allow_smaller_final_batch=True)
        else:
            batch = tf.train.batch(data_point, batch_size, capacity=capacity, num_threads=num_threads, 
                allow_smaller_final_batch=True)
        
        return batch


def get_batch_join(tfrecords_list, batch_size, shuffle=False, num_threads=1, min_after_dequeue=None, num_epochs=None):
    name = "batch_join" if not shuffle else "shuffle_batch_join"
    with tf.variable_scope(name):
        filename_queue = tf.train.string_input_producer(tfrecords_list, shuffle=shuffle, num_epochs=num_epochs)
        example_list = [read_parse_preproc(filename_queue) for _ in range(num_threads)]
        
        if min_after_dequeue is None:
            min_after_dequeue = batch_size * 10
        capacity = min_after_dequeue + 3*batch_size
        if shuffle:
            batch = tf.train.shuffle_batch_join(tensors_list=example_list, batch_size=batch_size, capacity=capacity, 
                                                min_after_dequeue=min_after_dequeue, allow_smaller_final_batch=True)
        else:
            batch = tf.train.batch_join(example_list, batch_size, capacity=capacity, allow_smaller_final_batch=True)
            
        return batch


# interfaces
def shuffle_batch_join(tfrecords_list, batch_size, num_threads, num_epochs, min_after_dequeue=None):
    return get_batch_join(tfrecords_list, batch_size, shuffle=True, num_threads=num_threads, 
        num_epochs=num_epochs, min_after_dequeue=min_after_dequeue)

def batch_join(tfrecords_list, batch_size, num_threads, num_epochs, min_after_dequeue=None):
    return get_batch_join(tfrecords_list, batch_size, shuffle=False, num_threads=num_threads, 
        num_epochs=num_epochs, min_after_dequeue=min_after_dequeue)

def shuffle_batch(tfrecords_list, batch_size, num_threads, num_epochs, min_after_dequeue=None):
    return get_batch(tfrecords_list, batch_size, shuffle=True, num_threads=num_threads, 
        num_epochs=num_epochs, min_after_dequeue=min_after_dequeue)

def batch(tfrecords_list, batch_size, num_threads, num_epochs, min_after_dequeue=None):
    return get_batch(tfrecords_list, batch_size, shuffle=False, num_threads=num_threads, 
        num_epochs=num_epochs, min_after_dequeue=min_after_dequeue)
