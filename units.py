import tensorflow as tf


def run_in_batch_avg(session, tensors, batch_placeholders, feed_dict={}, batch_size=32):
    res = [0] * (len(tensors))
    batch_tensors = [(placeholder, feed_dict[placeholder]) for placeholder in batch_placeholders]
    total_size = len(batch_tensors[0][1])  # first placeholder's data array length
    batch_count = (total_size + batch_size - 1) / batch_size
    for batch_idx in range(batch_count):
        current_batch_size = None

        for (placeholder, tensor) in batch_tensors:  # two placehoder's value must be change simultaneously
            batch_tensor = tensor[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            current_batch_size = len(batch_tensor)
            feed_dict[placeholder] = tensor[
                                     batch_idx * batch_size: (batch_idx + 1) * batch_size]  # change the xs ys's data

        tmp = session.run(tensors, feed_dict=feed_dict)
        res = [r + t * current_batch_size for (r, t) in zip(res, tmp)]  # weghted average
    return [r / float(total_size) for r in res]


def weight_variable(shape):
    initial = tf.contrib.layers.xavier_initializer_conv2d()
    return tf.Variable(initial(shape=shape))


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
    W = weight_variable([kernel_size, kernel_size, in_features, out_features])
    conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')
    if with_bias:
        return conv + bias_variable([out_features])
    return conv


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
