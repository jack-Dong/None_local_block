'''
this block are not offcial implement of non_local_block in paper
Non-local Neural Networks https://arxiv.org/pdf/1711.07971.pdf

'''

import tensorflow as tf
from  units import conv2d


def non_local_block(input_tensor, computation_compression=2, mode='dot'):
    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
        raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot`,`concatenate`')

    input_shape = input_tensor.get_shape().as_list()
    print(input_shape)
    batchsize, dim1, dim2, channels = input_shape

    if mode == 'gaussian':  # Gaussian instantiation
        x1 = tf.reshape(input_tensor, shape=[-1, dim1 * dim2, channels])
        x2 = tf.reshape(input_tensor, shape=[-1, dim1 * dim2, channels])

        f = tf.matmul(x1, x2, transpose_b=True)

        f = tf.reshape(f, shape=[-1, dim1 * dim2 * dim1 * dim2])

        f = tf.nn.softmax(f, axis=-1)

        f = tf.reshape(f, shape=[-1, dim1 * dim2, dim1 * dim2])

        print("gaussian=", f)
    elif mode == 'dot':
        theta = conv2d(input_tensor, channels, channels // 2, 1)  # add BN relu layer before conv will speed up training
        theta = tf.reshape(theta, shape=[-1, dim1 * dim2, channels // 2])

        phi = conv2d(input_tensor, channels, channels // 2, 1)
        phi = tf.reshape(phi, shape=[-1, dim1 * dim2, channels // 2])

        f = tf.matmul(theta, phi, transpose_b=True)

        # scale the values to make it size invarian t
        f = f / (dim1 * dim2 * channels)

        print("dot f=", f)

    elif mode == 'concatenate':  # this operation cost too much memory, so make sure you input a small resolution  feature map like(14X14 7X7)

        theta = conv2d(input_tensor, channels, channels // 2, 1)
        theta = tf.reshape(theta, shape=[-1, dim1 * dim2, channels // 2])

        phi = conv2d(input_tensor, channels, channels // 2, 1)
        phi = tf.reshape(phi, shape=[-1, dim1 * dim2, channels // 2])

        theta_splits = tf.split(theta, dim1 * dim2, 1)
        phi_splits = tf.split(phi, dim1 * dim2, 1)

        theta_split_shape = tf.shape(theta[0])
        print("theta_split_shape", theta_split_shape)

        initial = tf.constant(1.0 / channels, shape=[channels, 1])

        print('initial', initial)
        W_concat = tf.Variable(initial)

        print("W_concat", W_concat)

        f_matrix = []
        for i in range(dim1 * dim2):
            for j in range(dim1 * dim2):
                print(i, '  ', j)
                tmp = tf.concat([theta_splits[i], phi_splits[j]], 2)
                tmp = tf.reshape(tmp, shape=[-1, channels])
                # print(tmp)
                tmp = tf.matmul(tmp, W_concat)
                print(tmp)
                f_matrix.append(tmp)

        f_matrix_tensor = tf.stack(f_matrix, axis=2)
        print('f_matrix_tensor', f_matrix_tensor)

        f = tf.reshape(f_matrix_tensor, shape=[-1, dim1 * dim2, dim1 * dim2])

        f = f / (dim1 * dim2 * channels)

        print("concatenate f=", f)


    else:  # Embedded Gaussian instantiation
        theta = conv2d(input_tensor, channels, channels // 2, 1)
        theta = tf.reshape(theta, shape=[-1, dim1 * dim2, channels // 2])

        phi = conv2d(input_tensor, channels, channels // 2, 1)
        phi = tf.reshape(phi, shape=[-1, dim1 * dim2, channels // 2])

        if computation_compression > 1:
            phi = tf.layers.max_pooling1d(phi, pool_size=2, strides=computation_compression, padding='SAME')
            print('phi', phi)

        f = tf.matmul(theta, phi, transpose_b=True)

        phi_shape = phi.get_shape().as_list()
        f = tf.reshape(f, shape=[-1, dim1 * dim2 * phi_shape[1]])

        f = tf.nn.softmax(f, axis=-1)

        f = tf.reshape(f, shape=[-1, dim1 * dim2, phi_shape[1]])

        print("Embedded f=", f)

    g = conv2d(input_tensor, channels, channels // 2, 1)
    g = tf.reshape(g, shape=[-1, dim1 * dim2, channels // 2])

    if computation_compression > 1 and mode == 'embedded':
        g = tf.layers.max_pooling1d(g, pool_size=2, strides=computation_compression, padding='SAME')
        print('g', g)

    y = tf.matmul(f, g)

    print('y=', y)

    y = tf.reshape(y, shape=[-1, dim1, dim2, channels // 2])

    y = conv2d(y, channels // 2, channels, kernel_size=3)
    print('y=', y)

    residual = input_tensor + y

    return residual
