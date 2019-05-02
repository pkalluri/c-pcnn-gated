import tensorflow as tf
import numpy as np
import sys
import argparse
import keras
from PIL import Image

from models import PixelCNN
from autoencoder import *
from utils import *
import utils

def train(conf, data):
    X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channels])
    model = PixelCNN(X, conf)

    trainer = tf.train.RMSPropOptimizer(1e-3)
    gradients = trainer.compute_gradients(model.loss)
    clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in gradients]
    optimizer = trainer.apply_gradients(clipped_gradients)

    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(conf.ckpt_file):
            saver.restore(sess, conf.ckpt_file)
            print("Model Restored")
        if conf.epochs > 0:
            print("Started Model Training...")

        pointer = 0
        for i in range(conf.epochs):
            epoch_loss = 0.
            for j in range(conf.num_batches):
                if conf.data == "mnist_bw":
                    batch_X, batch_y = data.train.next_batch(conf.batch_size)  # batch_X is N,HW; batch_y is N
                    batch_X = batch_X.reshape([conf.batch_size,conf.img_height, conf.img_width, conf.channels]) # N,H,W,C
                    batch_X = binarize(batch_X)
                    batch_y = one_hot(batch_y, conf.num_classes)  # N,10
                elif conf.data == "mnist":
                    batch_X, batch_y, pointer = get_batch(data, pointer, conf.batch_size)
                    batch_X = batch_X.reshape([conf.batch_size,conf.img_height, conf.img_width, conf.channels]) # N,H,W,C
                    batch_y = one_hot(batch_y, conf.num_classes)  # N,10
                elif conf.data == "fashion":
                    batch_X, batch_y, pointer = get_batch(data, pointer, conf.batch_size)
                    batch_X = batch_X.reshape([conf.batch_size,conf.img_height, conf.img_width, conf.channels]) # N,H,W,C
                    batch_y = one_hot(batch_y, conf.num_classes)  # N,10
                if i==0 and j==0:
                    save_images(batch_X, conf.batch_size, 1, "batch.png", conf)
                data_dict = {X:batch_X}
                if conf.conditional is True:
                    data_dict[model.h] = batch_y
                # print(sess.run([model.out, model.pred], feed_dict=data_dict))
                _, cost = sess.run([optimizer, model.loss], feed_dict=data_dict)
                epoch_loss += cost
            print("Epoch: %d, Cost: %f"%(i, epoch_loss))
            if (i+1)%1 == 0:
                saver.save(sess, conf.ckpt_file)
                generate_samples(sess, X, model.h, model.pred, i+".png", conf)

        generate_samples(sess, X, model.h, model.pred, "final.png", conf)

# BEGIN
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='mnist_bw')
parser.add_argument('--bins', type=int, default=2)
parser.add_argument('--model', type=str, default='')
parser.add_argument('--layers', type=int, default=12)
parser.add_argument('--f_map', type=int, default=32)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--grad_clip', type=int, default=1)
parser.add_argument('--data_path', type=str, default='data')
parser.add_argument('--ckpt_path', type=str, default='ckpts')
parser.add_argument('--samples_path', type=str, default='out')
parser.add_argument('--summary_path', type=str, default='logs')
parser.add_argument('--loss', type=str, default='original') # original, official_nll, nll, sum, min
parser.add_argument('--id', type=int, default='-1')
parser.add_argument('--note', type=str, default='')
parser.add_argument('--debug', action='store_true') # Use few samples, make model deterministic, run
conf = parser.parse_args()
print("Configs: ", conf)

if conf.debug:
    utils.make_deterministic()
    np.set_printoptions(threshold=sys.maxsize)

# Get data
if conf.debug: conf.batch_size = 3
if conf.data == 'mnist_bw': # pixels in range (0,1)
    from tensorflow.examples.tutorials.mnist import input_data
    if not os.path.exists(conf.data_path):
        os.makedirs(conf.data_path)
    data = input_data.read_data_sets(conf.data_path)
    conf.num_classes = 10
    conf.img_height = 28
    conf.img_width = 28
    conf.channels = 1
    conf.bins = 2
    conf.num_batches = data.train.num_examples // conf.batch_size
    if conf.debug: conf.num_batches = 2
else: # pixels in range (0,255)
    high = 255
    if conf.data == 'mnist':
        from keras.datasets import mnist
        (X,y), _ = mnist.load_data()
        conf.num_classes = 10
        conf.img_height = 28
        conf.img_width = 28
        conf.channels = 1
    elif conf.data == 'fashion':
        from keras.datasets import fashion_mnist
        (X,y), _ = fashion_mnist.load_data()
        conf.num_classes = 10
        conf.img_height = 28
        conf.img_width = 28
        conf.channels = 1
    elif conf.data == 'cifar10':
        from keras.datasets import cifar10
        (X, y), _ = cifar10.load_data()
        conf.num_classes = 10
        conf.img_height = 32
        conf.img_width = 32
        conf.channels = 3
    # elif conf.data == 'cifar':
    #     from keras.datasets import cifar10
    #     data = cifar10.load_data()
    #     labels = data[0][1]
    #     data = data[0][0].astype(np.float32)
    #     # data[:,0,:,:] -= np.mean(data[:,0,:,:])
    #     # data[:,1,:,:] -= np.mean(data[:,1,:,:])
    #     # data[:,2,:,:] -= np.mean(data[:,2,:,:])
    #     data = np.transpose(data, (0, 2, 3, 1))
    #     conf.img_height = 32
    #     conf.img_width = 32
    #     conf.channels = 3
    #     conf.num_classes = 10

    # group pixel values into bins
    bin_size = (high+1) / conf.bins
    X = (X // bin_size).astype('uint8')
    # todo: bin y
    data = (X, y)

    conf.num_batches = X.shape[0] // conf.batch_size
    if conf.debug: conf.num_batches = 2

# Train
if conf.model == '':
    conf.conditional = False
    conf = makepaths(conf)
    train(conf, data)
elif conf.model.lower() == 'conditional':
    conf.conditional = True
    conf = makepaths(conf)
    train(conf, data)
elif conf.model.lower() == 'autoencoder':
    conf.conditional = True
    conf = makepaths(conf)
    trainAE(conf, data)


