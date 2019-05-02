import tensorflow as tf
from layers import *
import unconditional_binary_losses
import conditional_binary_losses
import unconditional_multiclass_losses
import conditional_multiclass_losses

class PixelCNN(object):
    def __init__(self, X, conf, full_horizontal=True, h=None):
        self.X = X
        if conf.data == "original":
            self.X_norm = X
        else:
            '''
                Image normalization for CIFAR-10 was supposed to be done here
            '''
            self.X_norm = X
        v_stack_in, h_stack_in = self.X_norm, self.X_norm

        if conf.conditional is True:
            if h is not None:
                self.h = h
            else:
                self.h = tf.placeholder(tf.float32, shape=[None, conf.num_classes]) 
        else:
            self.h = None

        for i in range(conf.layers):
            filter_size = 3 if i > 0 else 7
            mask = 'b' if i > 0 else 'a'
            residual = True if i > 0 else False
            i = str(i)
            with tf.variable_scope("v_stack"+i):
                v_stack = GatedCNN([filter_size, filter_size, conf.f_map], v_stack_in, False, mask=mask, conditional=self.h).output()
                v_stack_in = v_stack

            with tf.variable_scope("v_stack_1"+i):
                v_stack_1 = GatedCNN([1, 1, conf.f_map], v_stack_in, False, gated=False, mask=None).output()

            with tf.variable_scope("h_stack"+i):
                h_stack = GatedCNN([filter_size if full_horizontal else 1, filter_size, conf.f_map], h_stack_in, True, payload=v_stack_1, mask=mask, conditional=self.h).output()

            with tf.variable_scope("h_stack_1"+i):
                h_stack_1 = GatedCNN([1, 1, conf.f_map], h_stack, True, gated=False, mask=None).output()
                if residual:
                    h_stack_1 += h_stack_in # Residual connection
                h_stack_in = h_stack_1

        with tf.variable_scope("fc_1"):
            fc1 = GatedCNN([1, 1, conf.f_map], h_stack_in, True, gated=False, mask='b').output()

        if conf.data == "mnist_bw":
            with tf.variable_scope("fc_2"):
                self.fc2 = GatedCNN([1, 1, 1], fc1, True, gated=False, mask='b', activation=False).output() # N,H,W,1
            # print(self.fc2)

            losses = conditional_binary_losses if conf.model == 'conditional' else unconditional_binary_losses
            if conf.loss == 'original':
                self.loss = losses.original_loss(logits=self.fc2, labels=self.X, Y=self.h)
            elif conf.loss == 'nll':
                self.loss = losses.nll_loss(logits=self.fc2, labels=self.X, Y=self.h)
            elif conf.loss == 'unaveraged_nll':
                self.loss = losses.unaveraged_nll_loss(logits=self.fc2, labels=self.X, Y=self.h)
            elif conf.loss == 'sum':
                self.loss = losses.sum_loss(logits=self.fc2, labels=self.X, Y=self.h)
            elif conf.loss == 'unaveraged_sum':
                self.loss = losses.unaveraged_sum_loss(logits=self.fc2, labels=self.X, Y=self.h)
            elif conf.loss == 'min':
                self.loss = losses.min_loss(logits=self.fc2, labels=self.X, Y=self.h)

            self.out = tf.nn.sigmoid(self.fc2) # N,H,W,1
            # print(self.out)
            self.pred = tf.random_uniform(tf.shape(self.out))<self.out # sample; N,H,W,1
            # print(self.pred)

        else:
            with tf.variable_scope("fc_2"):
                self.fc2 = GatedCNN([1, 1, conf.channels * conf.bins], fc1, True, gated=False, mask='b', activation=False).output() # N,H,W,B
            # print(self.fc2)
            self.flat = tf.reshape(self.fc2, (-1, conf.bins)) # NHW,B
            # print(self.flat)
            self.labels = tf.cast(tf.reshape(self.X, [-1]), dtype=tf.int32) # NHW
            # print(self.labels)

            losses = conditional_multiclass_losses if conf.model == 'conditional' else unconditional_multiclass_losses
            if conf.loss == 'original':
                self.loss = losses.original_loss(logits=self.flat, labels=self.labels, Y=self.h)
                # self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.flat, labels=self.labels)
                # self.loss = tf.reduce_mean(self.losses)
            elif conf.loss == 'nll':
                self.loss = losses.nll_loss(logits=self.flat, labels=self.labels, Y=self.h)
            elif conf.loss == 'unaveraged_nll':
                self.loss = losses.unaveraged_nll_loss(logits=self.flat, labels=self.labels, Y=self.h)
            elif conf.loss == 'sum':
                self.loss = losses.sum_loss(logits=self.flat, labels=self.labels, Y=self.h)
            elif conf.loss == 'unaveraged_sum':
                self.loss = losses.unaveraged_sum_loss(logits=self.flat, labels=self.labels, Y=self.h)
            elif conf.loss == 'min':
                self.loss = losses.min_loss(logits=self.flat, labels=self.labels, Y=self.h)

            self.out = tf.nn.softmax(self.flat) # NHW,B
            # print(self.out)
            self.pred = tf.multinomial(tf.log(self.out), num_samples = 1, seed = 100) # sample; NHW,1
            # print(self.pred)
            self.pred = tf.reshape(self.pred, (-1, conf.img_height, conf.img_width, 1)) # sample; N,H,W,1
            # print(self.pred)


class ConvolutionalEncoder(object):
    def __init__(self, X, conf):
        '''
            This is the 6-layer architecture for Convolutional Autoencoder
            mentioned in the original paper: 
            Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction

            Note that only the encoder part is implemented as PixelCNN is taken
            as the decoder.
        '''

        W_conv1 = get_weights([5, 5, conf.channels, 100], "W_conv1")
        b_conv1 = get_bias([100], "b_conv1")
        conv1 = tf.nn.relu(conv_op(X, W_conv1) + b_conv1)
        pool1 = max_pool_2x2(conv1)

        W_conv2 = get_weights([5, 5, 100, 150], "W_conv2")
        b_conv2 = get_bias([150], "b_conv2")
        conv2 = tf.nn.relu(conv_op(pool1, W_conv2) + b_conv2)
        pool2 = max_pool_2x2(conv2)

        W_conv3 = get_weights([3, 3, 150, 200], "W_conv3")
        b_conv3 = get_bias([200], "b_conv3")
        conv3 = tf.nn.relu(conv_op(pool2, W_conv3) + b_conv3)
        conv3_reshape = tf.reshape(conv3, (-1, 7*7*200))

        W_fc = get_weights([7*7*200, 10], "W_fc")
        b_fc = get_bias([10], "b_fc")
        self.pred = tf.nn.softmax(tf.add(tf.matmul(conv3_reshape, W_fc), b_fc))


