
import tensorflow as tf


ALPHA = 1e-3
BETA = 1


def feature_matrix(layer):
    shape = layer.get_shape().as_list()

    # (batch, height, width, depth)
    if not shape[1]:
        matrix = tf.reshape(layer, shape=(shape[0], -1, shape[3]))
    else:
        matrix = tf.reshape(layer, shape=(-1, shape[1] * shape[2], shape[3]))

    return tf.transpose(matrix, perm=(0, 2, 1))


def gram_matrix(feature):
    return tf.matmul(feature, feature, transpose_b=True)


def content_loss(layer, p_l):

    f_l = feature_matrix(layer)

    return tf.reduce_mean(0.5 * tf.reduce_sum((f_l - p_l) ** 2, reduction_indices=[1, 2]), axis=0)


def style_loss(layer, a_l):

    g_l = gram_matrix(feature_matrix(layer))

    shape = tf.shape(layer)

    n_l = shape[3]
    m_l = shape[1] * shape[2]

    multiplier = tf.cast((4 * n_l ** 2 * m_l ** 2), dtype=tf.float32) ** -1

    return tf.reduce_mean(multiplier * tf.reduce_sum((g_l - a_l) ** 2, reduction_indices=[1, 2]), axis=0)


def total_loss(content_layers, style_layers, gram_matrices, feature_matrices, alpha=ALPHA, beta=BETA):

    total_content_loss = tf.constant(0.0, dtype=tf.float32)
    for layer, feature in zip(content_layers, feature_matrices):
        total_content_loss = total_content_loss + content_loss(layer, p_l=feature_matrices)

    total_style_loss = tf.constant(0.0, dtype=tf.float32)
    for layer, gram in zip(style_layers, gram_matrices):
        total_style_loss = total_style_loss + style_loss(layer, a_l=gram)

    total_content_loss /= len(content_layers)
    total_style_loss /= len(style_layers)

    return total_content_loss * alpha + beta * total_style_loss


