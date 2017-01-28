
import tensorflow as tf
from vgg import vgg16, download_weights_maybe
import numpy as np
import sys

from scipy.misc import imread, imresize, imsave

ALPHA = 1e-3
BETA = 1

LEARNING_RATE = 1e-4


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


def total_loss(content_layers, style_layers, feature_matrices, gram_matrices, alpha=ALPHA, beta=BETA):

    total_content_loss = tf.constant(0.0, dtype=tf.float32)
    for layer, feature in zip(content_layers, feature_matrices):
        total_content_loss = total_content_loss + content_loss(layer, p_l=feature_matrices)

    total_style_loss = tf.constant(0.0, dtype=tf.float32)
    for layer, gram in zip(style_layers, gram_matrices):
        total_style_loss = total_style_loss + style_loss(layer, a_l=gram)

    total_content_loss /= len(content_layers)
    total_style_loss /= len(style_layers)

    return total_content_loss * alpha + beta * total_style_loss


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
    image_shape = user_image.shape
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

    art_image = imresize(imread('images/starry_night.jpg'), (210, 280))
    user_image = imresize(imread('images/trump.jpg'), (210, 280))

    image = tf.Variable(initial_value=np.random.rand(1, 210, 280, 3), dtype=tf.float32, trainable=True, name='output_image')

    sess = tf.Session()

    with tf.variable_scope('vgg'):
        vgg = vgg16(image, reuse=False)
        download_weights_maybe('weights/vgg16_weights.npz')
        vgg.load_weights('weights/vgg16_weights.npz', sess)

    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    content_layers = ['conv4_2']

    feature_matrices, gram_matrices = precompute(style_layers, content_layers, vgg_scope='vgg', sess=sess, user_image=user_image, art_image=art_image)

    content_layer_ops = map(lambda layer: vgg.get_layer(layer), content_layers)
    style_layer_ops = map(lambda layer: vgg.get_layer(layer), style_layers)

    loss = total_loss(content_layer_ops, style_layer_ops, feature_matrices, gram_matrices)

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    sess.run(tf.global_variables_initializer())

    for step in xrange(550):
        sys.stdout.flush()
        sys.stdout.write('\r Step %i' % step)

        sess.run(optimizer)
        if step % 50 == 0:
            print "\rLoss for step %i: %f" % (step, sess.run(loss))
            imsave('images/result.png', sess.run(image).reshape((210, 280, 3)))

    imsave('images/result.png', sess.run(image).reshape((210, 280, 3)))

if __name__ == "__main__":
    main(sys.argv[1:])
