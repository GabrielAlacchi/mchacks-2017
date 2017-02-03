
import tensorflow as tf
import tensorflow.contrib.slim as slim

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


def conv_layer(x, filter_size, num_features, strides, trainable=True,
               relu=True, deconv=False, upscale=2,
               mirror_pad=True, collection=tf.GraphKeys.GLOBAL_VARIABLES):
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
        # Implement mirror padding (removes border around images)
        if mirror_pad:
            padding = 'VALID'

            # Mirror padding
            pad_amount = filter_size[0] // 2
            x = tf.pad(
                x, [[0, 0], [pad_amount, pad_amount], [pad_amount, pad_amount], [0, 0]],
                mode='REFLECT')
        else:
            padding = 'SAME'

        conv = tf.nn.conv2d(x, weights,
                            strides=[1] + list(strides) + [1],
                            padding=padding,
                            name='conv')

    bias = tf.nn.bias_add(conv, biases, name='bias_add')

    if relu:
        return tf.nn.relu(bias, name='activations')
    else:
        return bias


def upsample(x, filter_size, num_features, trainable=True,
             relu=True, upscale=2,
             mirror_pad=True, collection=tf.GraphKeys.GLOBAL_VARIABLES):

    if filter_size[0] % 2 == 0 or filter_size[1] % 2 == 0:
        raise ValueError('filter_size must be odd.')

    _, height, width, _ = [s.value for s in x.get_shape()]
    upsampled = tf.image.resize_nearest_neighbor(x, size=(upscale * height, upscale * width), name='upsample')
    return conv_layer(upsampled, filter_size, num_features=num_features,
                      strides=(1, 1), trainable=trainable, relu=relu,
                      mirror_pad=mirror_pad, collection=collection)


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


def instance_norm(x, gamma=None, beta=None, trainable=True, collection=tf.GraphKeys.GLOBAL_VARIABLES):

    params_shape = x.get_shape()[-1:]
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

    if gamma is None:
        gamma = tf.get_variable('gamma',
                                initializer=tf.ones(shape=params_shape, dtype=tf.float32),
                                trainable=trainable,
                                dtype=tf.float32,
                                collections=[collection])

    if beta is None:
        beta = tf.get_variable('beta',
                               initializer=tf.zeros(shape=params_shape, dtype=tf.float32),
                               trainable=trainable,
                               dtype=tf.float32,
                               collections=[collection])

    return tf.nn.batch_normalization(x, mean, var, beta, gamma, BN_EPSILON)


def weighted_instance_norm(x, style_weights, trainable=True, collection=tf.GraphKeys.GLOBAL_VARIABLES):
    """Instance normalization with weighted style_weightings

    Args:
        x: A tensor with (batch, height, width, depth) dimensions.
        style_weights: A 1D tensor of style weights of type tf.float32
        trainable: Whether or not to have the variables to be trainable.
        collection: The tensorflow collection to put the variables in.
    """

    num_styles = style_weights.get_shape()[-1:]
    params_shape = x.get_shape()[-1:]

    # Create a shape that is a matrix with (num_styles, depth) in dimension
    var_shape = num_styles.concatenate(params_shape)

    gamma = tf.get_variable('gamma',
                            initializer=tf.ones(shape=var_shape, dtype=tf.float32),
                            trainable=trainable,
                            dtype=tf.float32,
                            collections=[collection])

    beta = tf.get_variable('beta',
                           initializer=tf.zeros(shape=var_shape, dtype=tf.float32),
                           trainable=trainable,
                           dtype=tf.float32,
                           collections=[collection])

    gamma = tf.reduce_sum(gamma * style_weights, axis=0, name='style_weighted_gamma')
    beta = tf.reduce_sum(beta * style_weights, axis=0, name='style_weighted_beta')

    return instance_norm(x, gamma, beta, trainable=trainable, collection=collection)


def tensor_size(tensor):
    shape = tf.shape(tensor)
    return shape[0] * shape[1] * shape[2] * shape[3]


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

    scope_name = var.name.split(':')[0]
    with tf.name_scope(scope_name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
