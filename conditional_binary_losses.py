import tensorflow as tf

def original_loss(logits=None, labels=None, Y=None):
    # stabilized nll loss
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def nll_loss(logits=None, labels=None, Y=None):
    logits = tf.nn.sigmoid(logits)
    loss = labels * tf.log(logits) + (1 - labels) * tf.log((1 - logits))  # log pixel losses
    loss = tf.reduce_mean(loss, axis=[1, 2, 3])  # sum of log --> all pixels
    loss = tf.reduce_mean(loss, axis=0)  # sum of log --> all samples
    return -loss

def unaveraged_nll_loss(logits=None, labels=None, Y=None):
    logits = tf.nn.sigmoid(logits)
    loss = labels * tf.log(logits) + (1 - labels) * tf.log((1 - logits))  # log pixel losses
    loss = tf.reduce_sum(loss, axis=[1, 2, 3])  # sum of log --> all pixels
    loss = tf.reduce_sum(loss, axis=0)  # sum of log --> all samples
    return -loss

def sum_loss(logits=None, labels=None, Y=None):
    logits = tf.nn.sigmoid(logits)
    loss = labels * tf.log(logits) + (1 - labels) * tf.log(1 - logits)  # log pixel losses
    loss = tf.reduce_mean(loss, axis=[1, 2, 3])  # sum in log space --> all pixels

    num_classes = Y.shape[1]
    Y = tf.argmax(Y, axis=1)
    total_loss = tf.Variable(0.0)
    # sess.run(total_loss.initializer)

    for i in range(num_classes):  # number of classes
        relevant_Y = tf.equal(Y, i)  # tensor with True when y=i
        relevant_Y = tf.reshape(tf.where(relevant_Y), shape=[-1])  # the indices where y=i

        def add_relevant_loss(loss, relevant_Y, final_loss):
            relevant_loss = tf.gather(params=loss, indices=relevant_Y)  # get relevant rows
            relevant_loss = tf.reduce_logsumexp(relevant_loss)  # calculate the any-sample loss of these rows
            final_loss = final_loss + relevant_loss  # add to final loss
            return final_loss

        relevant_losses_exist = tf.math.logical_not(tf.equal(tf.shape(relevant_Y)[0], 0))
        total_loss = tf.cond(relevant_losses_exist, lambda: add_relevant_loss(loss, relevant_Y, total_loss),
                             lambda: total_loss)
    # average
    total_loss = total_loss / tf.cast(tf.shape(logits)[0], tf.float32)
    return -total_loss

# Future work...
# # Should currently be same as nll loss
# def sum_loss(logits=None, labels=None, Y=None):
#     logits = tf.nn.sigmoid(logits)
#     loss = labels * tf.log(logits) + (1 - labels) * tf.log(1 - logits)  # log pixel losses
#     loss = tf.reduce_mean(loss, axis=[1, 2, 3])  # sum in log space --> all pixels
#
#     num_classes = Y.shape[1]
#     Y = tf.argmax(Y, axis=1)
#     final_loss = tf.constant(0.0)
#     for i in range(num_classes): # number of classes
#         relevant_Y = tf.equal(Y,i) # tensor with True when y=i
#         relevant_Y = tf.reshape(tf.where(relevant_Y),shape=[-1]) # the indices where y=i
#         relevant_loss = tf.gather(params=loss, indices=relevant_Y) # get all of this element
#         relevant_loss = tf.reduce_logsumexp(relevant_loss) # calculate the any-sample loss
#         final_loss += relevant_loss # add to all-groups loss
#
#     # average
#     final_loss = final_loss / tf.cast(tf.shape(logits)[0], tf.float32)
#
#     return -final_loss

# def sum_loss(logits=None, labels=None):
#     denominator = tf.reduce_prod(tf.shape(logits)) # for averaging
#     logits = tf.nn.sigmoid(logits)
#     loss = labels * tf.log(logits) + (1 - labels) * tf.log(1 - logits) # log pixel losses
#     loss = tf.reduce_sum(loss, axis=[1,2,3]) # sum in log space --> all pixels
#     loss = tf.reduce_sum(tf.exp(loss, axis=0))/tf.cast(denominator, tf.float32) # sum in non-log space --> any sample
#     return -loss

# def unaveraged_sum_loss(logits=None, labels=None, Y=None):
#     # unaveraged_sum_loss
#     logits = tf.nn.sigmoid(logits)
#     loss = labels * tf.log(logits) + (1 - labels) * tf.log(1 - logits)  # log pixel losses
#     loss = tf.reduce_sum(loss, axis=[1, 2, 3])  # sum in log space --> all pixels
#     loss = tf.reduce_logsumexp(loss, axis=0)  # sum in non-log space --> any sample
#     return -loss

def min_loss(logits=None, labels=None, Y=None):
    logits = tf.nn.sigmoid(logits)
    loss = labels * tf.log(logits) + (1 - labels) * tf.log(1 - logits)  # log pixel losses
    loss = tf.reduce_mean(loss, axis=[1, 2, 3])  # sum in log space --> all pixels

    num_classes = Y.shape[1]
    Y = tf.argmax(Y, axis=1)
    total_loss = tf.Variable(0.0)
    # sess.run(total_loss.initializer)

    for i in range(num_classes):  # number of classes
        relevant_Y = tf.equal(Y, i)  # tensor with True when y=i
        relevant_Y = tf.reshape(tf.where(relevant_Y), shape=[-1])  # the indices where y=i

        def add_relevant_loss(loss, relevant_Y, final_loss):
            relevant_loss = tf.gather(params=loss, indices=relevant_Y)  # get relevant rows
            relevant_loss = tf.reduce_max(relevant_loss)  # calculate the any-sample loss of these rows
            final_loss = final_loss + relevant_loss  # add to final loss
            return final_loss

        relevant_losses_exist = tf.math.logical_not(tf.equal(tf.shape(relevant_Y)[0], 0))
        total_loss = tf.cond(relevant_losses_exist,
                             lambda: add_relevant_loss(loss, relevant_Y, total_loss),
                             lambda: total_loss)

    # no averaging
    # total_loss = total_loss / tf.cast(tf.shape(logits)[0], tf.float32)
    return -total_loss

# Future work...
# def soft_min_loss(logits=None, labels=None, Y=None):
#     logits = tf.nn.sigmoid(logits)
#     loss = labels * tf.log(logits) + (1 - labels) * tf.log((1 - logits))  # log pixel losses
#     loss = tf.reduce_mean(loss, axis=[1, 2, 3])  # sum in log space --> all pixels
#
#     num_classes = Y.shape[1]
#     Y = tf.argmax(Y, axis=1)
#     total_loss = tf.Variable(0.0)
#     # sess.run(total_loss.initializer)
#
#     for i in range(num_classes):  # number of classes
#         relevant_Y = tf.equal(Y, i)  # tensor with True when y=i
#         relevant_Y = tf.reshape(tf.where(relevant_Y), shape=[-1])  # the indices where y=i
#
#         def add_relevant_loss(loss, relevant_Y, final_loss):
#             relevant_loss = tf.gather(params=loss, indices=relevant_Y)  # get relevant rows
#             relevant_loss = tf.reduce_logsumexp(tf.exprelevant_loss)  # calculate the any-sample loss of these rows
#             final_loss = final_loss + relevant_loss  # add to final loss
#             return final_loss
#
#         relevant_losses_exist = tf.math.logical_not(tf.equal(tf.shape(relevant_Y)[0], 0))
#         total_loss = tf.cond(relevant_losses_exist,
#                              lambda: add_relevant_loss(loss, relevant_Y, total_loss),
#                              lambda: total_loss)
#
#     # no averaging
#     total_loss = total_loss / tf.cast(tf.shape(logits)[0], tf.float32)
#     return -total_loss

# def soft_min_loss(logits=None, labels=None, Y=None):
#     logits = tf.nn.sigmoid(logits)
#     loss = labels * tf.log(logits) + (1 - labels) * tf.log((1 - logits))  # log pixel losses
#     loss = tf.reduce_mean(loss, axis=[1, 2, 3])  # sum of log --> all pixels
#     loss = tf.reduce_mean(loss*(1-loss))  # -> highest-quality produced samples
#     return -loss

# def test_loss(logits=None, labels=None, Y=None):
#     logits = tf.nn.sigmoid(logits)
#     loss = labels * tf.log(logits) + (1 - labels) * tf.log(1 - logits)  # log pixel losses
#     loss = tf.reduce_mean(loss, axis=[1, 2, 3])  # sum in log space --> all pixels
#
#     num_classes = Y.shape[1]
#     Y = tf.argmax(Y, axis=1)
#     final_loss = tf.constant(0.0)
#     for i in range(num_classes): # number of classes
#         relevant_Y = tf.equal(Y,i) # tensor with True when y=i
#         relevant_Y = tf.reshape(tf.where(relevant_Y),shape=[-1]) # the indices where y=i
#         relevant_loss = tf.gather(params=loss, indices=relevant_Y) # get all of this element
#         relevant_loss = tf.reduce_sum(relevant_loss) # calculate the any-sample loss
#         final_loss += relevant_loss # add to all-groups loss
#
#     # average
#     final_loss = final_loss / tf.cast(tf.shape(logits)[0], tf.float32)
#
#     return -final_loss