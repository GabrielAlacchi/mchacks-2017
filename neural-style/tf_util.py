
import tensorflow as tf

ARTNET_BATCHNORM_COLLECTION = 'batch_norm'
ARTNET_WEIGHT_COLLECTION = 'artnet_collection'

BN_EPSILON = 0.001


def kernel_variable(name, shape, dtype=tf.float32, trainable=True, collection=tf.GraphKeys.GLOBAL_VARIABLES):
    return tf.get_variable(name,
                           initializer=tf.truncated_normal(shape=shape, mean=0.0, stddev=1e-1, dtype=dtype),
                           dtype=dtype,
                           trainable=trainable,
                           collections=[collection])


def bias_variable(name, shape, dtype=tf.float32, trainable=True, collection=tf.GraphKeys.GLOBAL_VARIABLES):
    return tf.get_variable(name,
                           initializer=tf.zeros(shape=shape, dtype=dtype),
                           trainable=trainable,
                           collections=[collection])


def conv_layer(x, filter_size, num_features, strides, trainable=True, relu=True, deconv=False, upscale=2, collection=tf.GraphKeys.GLOBAL_VARIABLES):
    x_shape = x.get_shape().as_list()

    filters_in = x_shape[-1]

    if deconv:
        kernel_shape = list(filter_size) + [num_features, filters_in]
    else:
        kernel_shape = list(filter_size) + [filters_in, num_features]

    weights = kernel_variable(name='weights',
                              shape=kernel_shape,
                              trainable=trainable,
                              collection=collection)
    biases = bias_variable(name='biases',
                           shape=[num_features],
                           trainable=trainable,
                           collection=collection)

    if deconv:
        shape = tf.shape(x)
        num_features_tensor = tf.constant(num_features, dtype=tf.int32)
        try:
            output_size = tf.pack([shape[0], shape[1] * upscale, shape[2] * upscale, num_features_tensor], axis=0)
        except AttributeError:
            output_size = tf.stack([shape[0], shape[1] * upscale, shape[2] * upscale, num_features_tensor], axis=0)

        conv = tf.nn.conv2d_transpose(x, weights,
                                      strides=[1] + list(strides) + [1],
                                      output_shape=output_size,
                                      padding='SAME',
                                      name='deconv')
    else:
        conv = tf.nn.conv2d(x, weights,
                            strides=[1] + list(strides) + [1],
                            padding='SAME',
                            name='conv')

    bias = tf.nn.bias_add(conv, biases, name='bias_add')

    if relu:
        return tf.nn.relu(bias, name='activations')
    else:
        return bias


def res_layer(x, filter_size=None, trainable=True, collection=tf.GraphKeys.GLOBAL_VARIABLES):
    if not filter_size:
        filter_size = (2, 2)

    shape = x.get_shape().as_list()
    with tf.variable_scope('conv_1'):
        conv1 = conv_layer(x, filter_size=filter_size, strides=(1, 1),
                           num_features=shape[-1], trainable=trainable, collection=collection)

    with tf.variable_scope('conv_2'):
        conv2 = conv_layer(conv1, filter_size=filter_size, strides=(1, 1),
                           num_features=shape[-1], trainable=trainable, relu=False, collection=collection)

    return tf.add(x, conv2, name='residual')


def batch_norm(x, trainable=True):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    # [0, 1, 2] for conv layers and [0] for fc layers
    axis = list(range(len(x_shape) - 1))

    with tf.variable_scope('batch_norm'):
        beta = _get_variable('beta',
                             initializer=tf.zeros(dtype=tf.float32, shape=params_shape))

        gamma = _get_variable('gamma',
                              initializer=tf.ones(dtype=tf.float32, shape=params_shape))

        # Op for computing the mean and variance
        mean, variance = tf.nn.moments(x, axes=axis, name='moments')

        if not trainable:
            mean = _get_variable(name='population_mean',
                                 trainable=False,
                                 initializer=tf.zeros(dtype=tf.float32, shape=params_shape))

            variance = _get_variable(name='population_variance',
                                     trainable=False,
                                     initializer=tf.ones(dtype=tf.float32, shape=params_shape))

            tf.add_to_collection(ARTNET_BATCHNORM_COLLECTION, mean)
            tf.add_to_collection(ARTNET_BATCHNORM_COLLECTION, variance)

        return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)


def instance_norm(x):
    epsilon = 1e-9

    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

    return tf.div(tf.sub(x, mean), tf.sqrt(tf.add(var, epsilon)))


def _get_variable(name, initializer, shape=None, dtype=tf.float32, trainable=True, collection=ARTNET_WEIGHT_COLLECTION):
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, ARTNET_WEIGHT_COLLECTION]
    if shape:
        return tf.get_variable(name,
                               shape=shape,
                               initializer=initializer,
                               dtype=dtype,
                               collections=collections,
                               trainable=trainable)
    else:
        return tf.get_variable(name,
                               initializer=initializer,
                               dtype=dtype,
                               collections=collections,
                               trainable=trainable)


def tensor_size(tensor):
    shape = tf.shape(tensor)
    return shape[0] * shape[1] * shape[2] * shape[3]
