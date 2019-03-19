import tensorflow as tf
import numpy as np
import argparse
from models import PixelCNN
from autoencoder import *
from utils import *
# import keras
import utils

def train(conf, data):
    X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])
    model = PixelCNN(X, conf)

    trainer = tf.train.RMSPropOptimizer(1e-3)
    gradients = trainer.compute_gradients(model.loss)

    clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in gradients]
    optimizer = trainer.apply_gradients(clipped_gradients)

    saver = tf.train.Saver(tf.trainable_variables())
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        if os.path.exists(conf.ckpt_file):
            saver.restore(sess, conf.ckpt_file)
            print("Model Restored")
        if conf.epochs > 0:
            print("Started Model Training...")
        pointer = 0
        for i in range(conf.epochs):
            for j in range(conf.num_batches):
                if conf.data == "mnist":
                    batch_X, batch_y = data.train.next_batch(conf.batch_size)
                    batch_X = binarize(batch_X.reshape([conf.batch_size, \
                                                        conf.img_height, conf.img_width, conf.channel]))
                    batch_y = one_hot(batch_y, conf.num_classes)
                else:
                    batch_X, pointer = get_batch(data, pointer, conf.batch_size)
                data_dict = {X:batch_X}
                if conf.conditional is True:
                    data_dict[model.h] = batch_y
                # _, cost = sess.run([optimizer, model.loss], feed_dict=data_dict)
                _, cost, h, logits, labels = sess.run([optimizer, model.loss, model.h, model.fc2, model.X], feed_dict=data_dict)
            # print(h)
            print("Epoch: %d, Cost: %f"%(i, cost))
            if (i+1)%10 == 0:
                saver.save(sess, conf.ckpt_file)
                generate_samples(sess, X, model.h, model.pred, conf, "")

        generate_samples(sess, X, model.h, model.pred, conf, "")

# BEGIN
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='mnist')
parser.add_argument('--layers', type=int, default=12)
parser.add_argument('--f_map', type=int, default=32)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--grad_clip', type=int, default=1)
parser.add_argument('--model', type=str, default='')
parser.add_argument('--data_path', type=str, default='data')
parser.add_argument('--ckpt_path', type=str, default='ckpts')
parser.add_argument('--samples_path', type=str, default='samples')
parser.add_argument('--summary_path', type=str, default='logs')
parser.add_argument('--loss', type=str, default='original') # original, official_nll, nll, sum, min
parser.add_argument('--id', type=int, default='-1')
parser.add_argument('--debug', action='store_true') # Use few samples, make model deterministic
parser.add_argument('--note', type=str, default='')
conf = parser.parse_args()
print(conf)

if conf.debug: utils.make_deterministic()

if conf.debug: conf.batch_size = 2
if conf.data == 'mnist':
    from tensorflow.examples.tutorials.mnist import input_data
    if not os.path.exists(conf.data_path):
        os.makedirs(conf.data_path)
    data = input_data.read_data_sets(conf.data_path)
    conf.num_classes = 10
    conf.img_height = 28
    conf.img_width = 28
    conf.channel = 1
    conf.num_batches = data.train.num_examples // conf.batch_size
    if conf.debug: conf.num_batches = 1
elif conf.data == 'cifar':
    from keras.datasets import cifar10
    data = cifar10.load_data()
    labels = data[0][1]
    data = data[0][0].astype(np.float32)
    data[:,0,:,:] -= np.mean(data[:,0,:,:])
    data[:,1,:,:] -= np.mean(data[:,1,:,:])
    data[:,2,:,:] -= np.mean(data[:,2,:,:])
    data = np.transpose(data, (0, 2, 3, 1))
    conf.img_height = 32
    conf.img_width = 32
    conf.channel = 3
    conf.num_classes = 10
    conf.num_batches = data.shape[0] // conf.batch_size
    if conf.debug: conf.num_batches = 1

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


