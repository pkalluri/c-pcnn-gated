import tensorflow as tf

def original_loss(logits=None, labels=None):
    # stabilized nll loss
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def nll_loss(logits=None, labels=None):
    logits = tf.nn.sigmoid(logits)
    loss = labels * tf.log(logits) + (1 - labels) * tf.log((1 - logits))  # log pixel losses
    loss = tf.reduce_mean(loss, axis=[1, 2, 3])  # sum of log --> all pixels
    loss = tf.reduce_mean(loss, axis=0)  # sum of log --> all samples
    return -loss

def unaveraged_nll_loss(logits=None, labels=None):
    logits = tf.nn.sigmoid(logits)
    loss = labels * tf.log(logits) + (1 - labels) * tf.log((1 - logits))  # log pixel losses
    loss = tf.reduce_sum(loss, axis=[1, 2, 3])  # sum of log --> all pixels
    loss = tf.reduce_sum(loss, axis=0)  # sum of log --> all samples
    return -loss

def sum_loss(logits=None, labels=None):
    denominator = tf.reduce_prod(tf.shape(logits))  # for averaging
    logits = tf.nn.sigmoid(logits)
    loss = labels * tf.log(logits) + (1 - labels) * tf.log(1 - logits)  # log pixel losses
    loss = tf.reduce_sum(loss, axis=[1, 2, 3])  # sum in log space --> all pixels
    loss = tf.reduce_logsumexp(loss, axis=0) / tf.cast(denominator,
                                                       tf.float32)  # sum in non-log space --> any sample
    return -loss

# TODO?
# def sum_loss(logits=None, labels=None):
#     denominator = tf.reduce_prod(tf.shape(logits)) # for averaging
#     logits = tf.nn.sigmoid(logits)
#     loss = labels * tf.log(logits) + (1 - labels) * tf.log(1 - logits) # log pixel losses
#     loss = tf.reduce_sum(loss, axis=[1,2,3]) # sum in log space --> all pixels
#     loss = tf.reduce_sum(tf.exp(loss, axis=0))/tf.cast(denominator, tf.float32) # sum in non-log space --> any sample
#     return -loss

def unaveraged_sum_loss(logits=None, labels=None):
    # unaveraged_sum_loss
    logits = tf.nn.sigmoid(logits)
    loss = labels * tf.log(logits) + (1 - labels) * tf.log(1 - logits)  # log pixel losses
    loss = tf.reduce_sum(loss, axis=[1, 2, 3])  # sum in log space --> all pixels
    loss = tf.reduce_logsumexp(loss, axis=0)  # sum in non-log space --> any sample
    return -loss

# TODO
def min_loss(logits=None, labels=None):
    tf.nn.softmax_cross_entropy_with_logits
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

# def test_loss(logits=None, labels=None):
#     x = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
#     return x