import numpy as np
import os
import scipy.misc
from datetime import datetime
import tensorflow as tf
import random
from PIL import Image

def binarize(images):
    # Converts each pixel value to 0 or 1
    return (np.random.uniform(size=images.shape) < images).astype(np.float32)

def generate_samples(sess, X, h, pred, filename, conf):
    print("Generating Sample Images...")
    n_row, n_col = 10,10
    samples = np.zeros((n_row*n_col, conf.img_height, conf.img_width, conf.channels))
    labels = one_hot(np.array([0,1,2,3,4,5,6,7,8,9]*10), conf.num_classes)  # todo

    for i in range(conf.img_height):
        for j in range(conf.img_width):
            for k in range(conf.channels):
                data_dict = {X:samples}
                if conf.conditional is True:
                    data_dict[h] = labels
                next_sample = sess.run(pred, feed_dict=data_dict)
                # print(next_sample.dtype, next_sample.shape, np.min(next_sample), np.max(next_sample))
                samples[:, i, j, k] = next_sample[:, i, j, k]
                # if conf.debug: save_images(samples, n_row, n_col, conf)
    save_images(samples, n_row, n_col, filename, conf)


def generate_ae(sess, encoder_X, decoder_X, y, data, conf, epoch):
    print("Generating Sample Images...")
    n_row, n_col = 10,10
    samples = np.zeros((n_row*n_col, conf.img_height, conf.img_width, conf.channels), dtype=np.float32)
    if conf.data == 'mnist':
        labels = binarize(data.train.next_batch(n_row*n_col)[0].reshape(n_row*n_col, conf.img_height, conf.img_width, conf.channels))
    else:
        labels = get_batch(data, 0, n_row*n_col) 

    for i in range(conf.img_height):
        for j in range(conf.img_width):
            for k in range(conf.channels):
                next_sample = sess.run(y, {encoder_X: labels, decoder_X: samples})
                if conf.data == 'mnist':
                    next_sample = binarize(next_sample)
                samples[:, i, j, k] = next_sample[:, i, j, k]

    save_images(samples, n_row, n_col, epoch+".png", conf)


def save_images(images, n_row, n_col, filename, conf):
    '''
    Saves images as a single .png
    :param images: numpy array with shape N,H,W or N,H,W,C
    '''
    # Put images in grid
    if conf.channels == 1:
        images = images.reshape((n_row, n_col, conf.img_height, conf.img_width))
        images = images.transpose((0,2,1,3))
        images = images.reshape(n_row*conf.img_height, n_col*conf.img_width)

    images = (images/(conf.bins-1) * 255).astype('uint8') # change number of bins back to 256

    filepath = os.path.join(conf.samples_path, filename)
    # filepath = os.path.join(conf.samples_path, datetime.now().strftime('%Y_%m_%d-%H_%M_%S')+".png")
    if conf.debug: Image.fromarray(images).show()
    Image.fromarray(images).save(filepath)
    print("Saved "+ filepath)


def save_batch_details(batch_X, conf):
    '''
    Save various details about this batch of images. Useful during debugging.
    :param batch_X: numpy array with shape N,H,W or N,H,W,C
    :param conf:
    '''
    flat = batch_X.flatten()
    flat.sort()
    print("Batch details: ", batch_X.dtype, batch_X.shape, np.min(batch_X), np.max(batch_X), flat)
    save_images(batch_X, conf.batch_size, 1, "batch_debug.png", conf)


# def save_images(samples, n_row, n_col, conf, suff):
#     images = samples
#     images = images.reshape((n_row, n_col, conf.img_height, conf.img_width))
#     print(images.shape)
#     images = images.transpose(1, 2, 0, 3) # WRONG ORDER!!!
#     print(images.shape)
#     images = images.reshape((conf.img_height * n_row, conf.img_width * n_col))
#     print(images.shape)
#     # else:
#     #     images = images.reshape((n_row, n_col, conf.img_height, conf.img_width, conf.channels))
#     #     images = images.transpose(1, 2, 0, 3, 4)
#     #     images = images.reshape((conf.img_height * n_row, conf.img_width * n_col, conf.channels))
#     # images = np.random.uniform(0,size=(conf.img_height * n_row, conf.img_width * n_col)).astype(np.int32)
#     filename = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')+suff+".jpg"
#     scipy.misc.toimage(images, cmin=0.0, cmax=1.0).save(os.path.join(conf.samples_path, filename))
#     print("Saved {}".format(os.path.join(conf.samples_path, filename)))


def get_batch(data, pointer, batch_size):
    all_X, all_y = data
    if (batch_size + 1) * pointer >= all_X.shape[0]:
        pointer = 0
    batch_X = all_X[batch_size * pointer : batch_size * (pointer + 1)]
    batch_y = all_y[batch_size * pointer : batch_size * (pointer + 1)]
    pointer += 1
    return [batch_X, batch_y, pointer]


def one_hot(batch_y, num_classes):
    y_ = np.zeros((batch_y.shape[0], num_classes))
    y_[np.arange(batch_y.shape[0]), batch_y] = 1
    return y_


def makepaths(conf):
    ckpt_full_path = os.path.join(conf.ckpt_path, "data=%s_bs=%d_layers=%d_fmap=%d"%(conf.data, conf.batch_size, conf.layers, conf.f_map))
    ckpt_full_path += '_conditional' if conf.conditional else ''
    if not os.path.exists(ckpt_full_path):
        os.makedirs(ckpt_full_path)
    conf.ckpt_file = os.path.join(ckpt_full_path, "model.ckpt")

    prefix = '%d_'%conf.id if conf.id > 0 else ''
    conf.samples_path = os.path.join(conf.samples_path, prefix)
    conf.samples_path += '_%s'%conf.data
    conf.samples_path += '_conditional' if conf.conditional else ''
    conf.samples_path += "_bins=%d_bs=%d_fmap=%d_layers=%d_epoch=%d_loss=%s_%s" % \
                         (conf.bins, conf.batch_size, conf.f_map, conf.layers, conf.epochs, conf.loss, conf.note)
    conf.samples_path += '_debug' if conf.debug else ''
    if not os.path.exists(conf.samples_path):
        os.makedirs(conf.samples_path)
    if tf.gfile.Exists(conf.summary_path):
        tf.gfile.DeleteRecursively(conf.summary_path)
    tf.gfile.MakeDirs(conf.summary_path)

    return conf


def make_deterministic(seed=1234):
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)