# coding: utf-8
# check python `logging` module
import tensorflow as tf
# import warnings

'''https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
Matplotlib chooses Xwindows backend by default. You need to set matplotlib do not use Xwindows backend.
- `matplotlib.use('Agg')`
- Or add to .config/matplotlib/matplotlibrc line backend : Agg.
- Or when connect to server use ssh -X ... command to use Xwindows.
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.misc
import numpy as np

# warnings.simplefilter('error')

def get_best_gpu():
    '''Dependency: pynvml (for gpu memory informations)
    '''
    from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetMemoryInfo

    print '\n===== Check GPU memory ====='

    # byte to megabyte
    to_mb = lambda x: int(x/1024./1024.) 

    best_idx = -1
    best_free = 0.
    nvmlInit()
    n_gpu = nvmlDeviceGetCount()
    for i in range(n_gpu):
        handle = nvmlDeviceGetHandleByIndex(i)
        name = nvmlDeviceGetName(handle)
        mem = nvmlDeviceGetMemoryInfo(handle)

        total = to_mb(mem.total)
        free = to_mb(mem.free)
        used = to_mb(mem.used)
        free_ratio = mem.free / float(mem.total)

        print("{} - {}/{} MB (free: {} MB - {:.2%})".format(name, used, total, free, free_ratio))

        if free > best_free:
            best_free = free
            best_idx = i

    print '\nSelected GPU is gpu:{}'.format(best_idx)
    print '============================\n'

    return best_idx


# 전체를 다 iterate 하면서 숫자를 셈
# 총 20만개 examples (128개 tfrecord files) iterate 하는데 1.5초 정도 걸림
def num_examples_from_tfrecords(tfrecords_list):
    num_examples = 0 
    for path in tfrecords_list:
        num_examples += sum(1 for _ in tf.python_io.tf_record_iterator(path))
    return num_examples


def expected_shape(tensor, expected):
    """batch size N shouldn't be set. you can use shape instead of tensor.
    
    Usage:
    # batch size N is omitted.
    expected_shape(tensor, [28, 28, 1])
    expected_shape(tensor.shape, [28, 28, 1])
    """
    if isinstance(tensor, tf.Tensor):
        shape = tensor.shape[1:]
    else:
        shape = tensor[1:]
    shape = map(lambda x: x.value, shape)
    err_msg = 'wrong shape {} (expected shape is {})'.format(shape, expected)
    assert shape == expected, err_msg
    # if not shape == expected:
    #     warnings.warn('wrong shape {} (expected shape is {})'.format(shape, expected))


def plot(samples, shape=(4,4), figratio=0.75):
    """only for square-size samples
    wh = sqrt(samples.size)
    figratio: small-size = 0.75 (default) / big-size = 1.0
    """
    if len(samples) != shape[0]*shape[1]:
        print("Error: # of samples = {} but shape is {}".format(len(samples), shape))
        return
    
    h_figsize = shape[0] * figratio
    w_figsize = shape[1] * figratio
    fig = plt.figure(figsize=(w_figsize, h_figsize))
    gs = gridspec.GridSpec(shape[0], shape[1])
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample) # cmap hmm...

    return fig


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def merge(images, size):
    """merge images.

    checklist before/after imsave:
    * are images post-processed? for example - denormalization
    * is np.squeeze required? maybe for grayscale...
    """
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter must have dimensions: HxW or HxWx3 or HxWx4')
