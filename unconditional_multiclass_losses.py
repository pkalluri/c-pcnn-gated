import tensorflow as tf

def original_loss(logits=None, labels=None, Y=None):
    # stabilized nll loss
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

def nll_loss(logits=None, labels=None, Y=None):
    logits = tf.nn.softmax(logits)
    loss = tf.reduce_sum(labels * tf.log(logits))  # log pixel losses
    loss = tf.reduce_mean(loss, axis=[1, 2])  # sum of log --> all pixels
    loss = tf.reduce_mean(loss, axis=0)  # sum of log --> all samples
    return -loss

def unaveraged_nll_loss(logits=None, labels=None, Y=None):
    logits = tf.nn.softmax(logits)
    loss = labels * tf.log(logits)  # log pixel losses
    loss = tf.reduce_sum(loss, axis=[1, 2, 3])  # sum of log --> all pixels
    loss = tf.reduce_sum(loss, axis=0)  # sum of log --> all samples
    return -loss

def sum_loss(logits=None, labels=None, Y=None):
    denominator = tf.reduce_prod(tf.shape(logits))  # for averaging
    logits = tf.nn.softmax(logits)
    loss = labels * tf.log(logits)  # log pixel losses
    loss = tf.reduce_sum(loss, axis=[1, 2, 3])  # sum in log space --> all pixels
    loss = tf.reduce_logsumexp(loss, axis=0) / tf.cast(denominator, tf.float32)  # sum in non-log space --> any sample
    # todo: try reduce_sum instead
    return -loss

def unaveraged_sum_loss(logits=None, labels=None, Y=None):
    # unaveraged_sum_loss
    logits = tf.nn.softmax(logits)
    loss = labels * tf.log(logits)  # log pixel losses
    loss = tf.reduce_sum(loss, axis=[1, 2, 3])  # sum in log space --> all pixels
    loss = tf.reduce_logsumexp(loss, axis=0)  # sum in non-log space --> any sample
    # todo: try reduce_sum instead
    return -loss

def min_loss(logits=None, labels=None, Y=None):
    logits = tf.nn.softmax(logits)
    loss = labels * tf.log(logits)  # log pixel losses
    loss = tf.reduce_sum(loss, axis=[1, 2, 3])  # sum in log space --> all pixels TODO: this was reduce_mean! Investigate.
    loss = tf.reduce_max(loss) # max in log space same as max in non-log space --> best sample
    return -loss

# Investigate further
# def soft_min_loss(logits=None, labels=None, Y=None):
#     logits = tf.nn.sigmoid(logits)
#     loss = labels * tf.log(logits) + (1 - labels) * tf.log((1 - logits))  # log pixel losses
#     loss = tf.reduce_mean(loss, axis=[1, 2, 3])  # sum of log --> all pixels
#     loss = tf.reduce_mean(loss*(1-loss))  # -> highest-quality produced samples
#     return -loss

# For debugging
# def test_loss(logits=None, labels=None):
#     x = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
#     return x