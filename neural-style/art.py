
import tensorflow as tf
from vgg import vgg16, download_weights_maybe
import numpy as np
import sys

import cv2
from tf_util import tensor_size

ALPHA = 1e-5
BETA = 1e-2
TV_WEIGHT = 1e-2

LEARNING_RATE = 5.0


def feature_matrix(layer):
    shape = tf.shape(layer)

    matrix_shape = tf.stack([shape[0], shape[1] * shape[2], shape[3]], axis=0)

    # (batch, height, width, depth)
    matrix = tf.reshape(layer, shape=matrix_shape)

    return tf.transpose(matrix, perm=(0, 2, 1))


def gram_matrix(feature):

    shape = tf.shape(feature)
    n_l = shape[1]
    m_l = shape[2]

    multiplier = 1.0 / tf.cast(2 * n_l * m_l, dtype=tf.float32)
    feature_norm = multiplier * feature
    return tf.matmul(feature_norm, feature, transpose_b=True)

    # return tf.matmul(feature, feature, transpose_b=True)


def content_loss(layer, p_l):

    f_l = feature_matrix(layer)

    return tf.reduce_mean(0.5 * tf.reduce_sum((f_l - p_l) ** 2, reduction_indices=[1, 2]), axis=0)


def style_loss(layer, a_l):

    g_l = gram_matrix(feature_matrix(layer))

    return tf.reduce_mean(tf.reduce_sum((g_l - a_l) ** 2, reduction_indices=[1, 2]), axis=0)


def total_loss(image, content_layers, style_layers, feature_matrices, gram_matrices, alpha=ALPHA, beta=BETA):

    shape = tf.shape(image)

    total_content_loss = tf.constant(0.0, dtype=tf.float32)
    for layer, feature in zip(content_layers, feature_matrices):
        total_content_loss = total_content_loss + content_loss(layer, p_l=feature)

    total_style_loss = tf.constant(0.0, dtype=tf.float32)
    for layer, gram in zip(style_layers, gram_matrices):
        total_style_loss = total_style_loss + style_loss(layer, a_l=gram)

    total_content_loss /= len(content_layers)
    total_style_loss /= len(style_layers)

    # Total Variation Denoising (IDK WHAT THIS DOES DON'T ASK ME)
    tv_y_size = tensor_size(image[:, 1:, :, :])
    tv_x_size = tensor_size(image[:, :, 1:, :])
    tv_loss = TV_WEIGHT * 2 * (
        (tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :shape[1] - tf.constant(1, dtype=tf.int32), :, :]) /
         tf.cast(tv_y_size, tf.float32)) +
        (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :shape[2] - tf.constant(1, dtype=tf.int32), :]) /
         tf.cast(tv_x_size, tf.float32)))

    return total_content_loss * alpha + beta * total_style_loss + tv_loss


def precompute(style_layers, content_layers, vgg_scope, sess, user_image, art_image):

    image_shape = art_image.shape
    x = tf.placeholder(dtype=tf.float32, shape=[1] + list(image_shape))

    with tf.variable_scope(vgg_scope, reuse=True):
        vgg = vgg16(x, reuse=True)

    gram_matrices = []
    for layer in style_layers:
        layer_op = vgg.get_layer(layer)
        gram_op = gram_matrix(feature_matrix(layer_op))
        gram = sess.run(gram_op, feed_dict={
            x: art_image.reshape([1] + list(image_shape))
        })

        shape = list(gram.shape)
        gram_matrices += [tf.constant(gram.reshape(shape[1:]))]

    feature_matrices = []
    if len(content_layers) > 0:

        image_shape = user_image.shape
        x = tf.placeholder(dtype=tf.float32, shape=[1] + list(image_shape))

        with tf.variable_scope(vgg_scope, reuse=True):
            vgg = vgg16(x, reuse=True)

        for layer in content_layers:
            layer_op = vgg.get_layer(layer)
            feature_op = feature_matrix(layer_op)
            feature = sess.run(feature_op, feed_dict={
                x: user_image.reshape([1] + list(image_shape))
            })

            shape = list(feature.shape)
            feature_matrices += [tf.constant(feature.reshape(shape[1:]))]

    return feature_matrices, gram_matrices


def main(argv):

    art_image = cv2.imread('images/starry_night.jpg')
    user_image = cv2.imread('images/trump.jpg')

    image_shape = user_image.shape

    image = tf.Variable(initial_value=np.random.rand(1, image_shape[0], image_shape[1], image_shape[2]), dtype=tf.float32, trainable=True, name='output_image')

    sess = tf.Session()

    with tf.variable_scope('vgg'):
        vgg = vgg16(image, reuse=False)
        download_weights_maybe('weights/vgg16_weights.npz')
        vgg.load_weights('weights/vgg16_weights.npz', sess)

    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    content_layers = ['conv3_2', 'conv4_2']

    feature_matrices, gram_matrices = precompute(style_layers, content_layers, vgg_scope='vgg', sess=sess, user_image=user_image, art_image=art_image)

    content_layer_ops = map(lambda layer: vgg.get_layer(layer), content_layers)
    style_layer_ops = map(lambda layer: vgg.get_layer(layer), style_layers)

    loss = total_loss(image, content_layer_ops, style_layer_ops, feature_matrices, gram_matrices)

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    sess.run(tf.variables_initializer(global_vars))

    for step in xrange(550):
        sys.stdout.flush()
        sys.stdout.write('\r Step %i' % step)

        sess.run(optimizer)
        if step % 50 == 0:
            print "\rLoss for step %i: %f" % (step, sess.run(loss))
            cv2.imwrite('images/result.png', sess.run(image).reshape(image_shape))

    print 'Final Loss: %f' % sess.run(loss)
    cv2.imwrite('images/result.png', sess.run(image).reshape(image_shape))

if __name__ == "__main__":
    main(sys.argv[1:])
