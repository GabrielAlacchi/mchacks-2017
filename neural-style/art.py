
import tensorflow as tf
from vgg import vgg16, download_weights_maybe
import numpy as np
import sys

import cv2
from ops import tensor_size

# Default hyper parameters
ALPHA = 1e-1
BETA = 1e-3
TV_WEIGHT = 1.0

LEARNING_RATE = 10
MAX_STEPS = 550

if __name__ == "__main__":
    # Set up flags
    flags = tf.app.flags

    flags.DEFINE_float('alpha', ALPHA, 'Alpha: training parameter (content weighting)')
    flags.DEFINE_float('beta', BETA, 'Beta: training parameter (style weighting)')
    flags.DEFINE_float('tv_weight', TV_WEIGHT, 'TV Weight: Total variation loss weighting')

    flags.DEFINE_float('learning_rate', LEARNING_RATE, 'Learning rate')
    flags.DEFINE_integer('max_steps', MAX_STEPS, 'Number of steps to train for.')

    flags.DEFINE_string('content_layers', 'conv2_2', 'Content layers, comma delimited')
    flags.DEFINE_string('style_layers', 'conv1_2,conv2_2,conv3_3,conv4_3', 'Style layers, comma delimited')

    flags.DEFINE_string('style_image', 'images/mosaic.jpg', 'Image to draw style from')
    flags.DEFINE_string('input_image', 'images/gabriel.jpg', 'Image to transfer style onto')

    flags.DEFINE_string('logdir', 'logdir', 'Place to write summaries too.')

    FLAGS = flags.FLAGS


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
    shape = tf.shape(f_l)
    multiplier = 1.0 / tf.cast(shape[1] * shape[2], dtype=tf.float32)
    return tf.reduce_mean(tf.reduce_sum(multiplier * (f_l - p_l) ** 2, reduction_indices=[1, 2]), axis=0)


def style_loss(layer, a_l):

    g_l = gram_matrix(feature_matrix(layer))

    return tf.reduce_mean(tf.reduce_sum((g_l - a_l) ** 2, reduction_indices=[1, 2]), axis=0)


def total_loss(image, content_layers, style_layers, feature_matrices, gram_matrices,
               alpha=ALPHA, beta=BETA, tv_weight=TV_WEIGHT, total_variation=True,
               summaries=False, summary_scope='summaries'):

    shape = tf.shape(image)

    total_content_loss = tf.constant(0.0, dtype=tf.float32)
    for layer, feature in zip(content_layers, feature_matrices):
        total_content_loss = total_content_loss + content_loss(layer, p_l=feature)

    total_style_loss = tf.constant(0.0, dtype=tf.float32)
    for layer, gram in zip(style_layers, gram_matrices):
        total_style_loss = total_style_loss + style_loss(layer, a_l=gram)

    total_content_loss /= len(content_layers)
    total_style_loss /= len(style_layers)

    if total_variation:
        # Total Variation Denoising (IDK WHAT THIS DOES DON'T ASK ME)
        tv_y_size = tensor_size(image[:, 1:, :, :])
        tv_x_size = tensor_size(image[:, :, 1:, :])
        tv_loss = 2 * (
            (tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :shape[1] - tf.constant(1, dtype=tf.int32), :, :]) /
             tf.cast(tv_y_size, tf.float32)) +
            (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :shape[2] - tf.constant(1, dtype=tf.int32), :]) /
             tf.cast(tv_x_size, tf.float32)))
    else:
        tv_loss = tf.constant(0.0, dtype=tf.float32)

    total_content_loss *= alpha
    total_style_loss *= beta
    tv_loss *= tv_weight

    loss = total_content_loss + total_style_loss + tv_loss

    if summaries:
        with tf.name_scope(summary_scope):
            tf.summary.scalar('Content Loss', total_content_loss)
            tf.summary.scalar('Style Loss', total_style_loss)
            tf.summary.scalar('Total Variation Loss', tv_loss)
            tf.summary.scalar('Total Loss', loss)

    return loss


def precompute(style_layers, content_layers, vgg_scope, sess, user_image, art_image):

    image_shape = art_image.shape
    x = tf.placeholder(dtype=tf.float32, shape=[1] + list(image_shape))

    with tf.variable_scope(vgg_scope, reuse=True):
        vgg = vgg16(x, reuse=True)

    gram_matrices = []
    for layer in style_layers:
        print "Precomputing gram matrix for %s" % layer
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

    art_image = cv2.imread(FLAGS.style_image)
    user_image = cv2.imread(FLAGS.input_image)

    art_image = cv2.cvtColor(art_image, cv2.COLOR_BGR2RGB)
    user_image = cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB)

    image_shape = user_image.shape

    image = tf.Variable(initial_value=np.random.rand(1, image_shape[0], image_shape[1], image_shape[2]), dtype=tf.float32, trainable=True, name='output_image')

    sess = tf.Session()

    with tf.variable_scope('vgg'):
        vgg = vgg16(image, reuse=False)
        download_weights_maybe('weights/vgg16_weights.npz')
        vgg.load_weights('weights/vgg16_weights.npz', sess)

    style_layers = FLAGS.style_layers.split(',')
    content_layers = FLAGS.content_layers.split(',')

    feature_matrices, gram_matrices = precompute(style_layers, content_layers, vgg_scope='vgg', sess=sess, user_image=user_image, art_image=art_image)

    content_layer_ops = map(lambda layer: vgg.get_layer(layer), content_layers)
    style_layer_ops = map(lambda layer: vgg.get_layer(layer), style_layers)

    loss = total_loss(image, content_layer_ops, style_layer_ops, feature_matrices, gram_matrices,
                      alpha=FLAGS.alpha, beta=FLAGS.beta, tv_weight=FLAGS.tv_weight, total_variation=True,
                      summaries=True, summary_scope='summaries')

    global_step = tf.Variable(0, trainable=False, name='global_step', collections=[tf.GraphKeys.GLOBAL_STEP])

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss, global_step=global_step)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.variables_initializer([global_step]))

    merged_summaries = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logdir=FLAGS.logdir)

    step = 0
    while step < FLAGS.max_steps:

        step = sess.run(global_step)
        sys.stdout.flush()
        sys.stdout.write('\r Step %i' % step)

        summary, _ = sess.run([merged_summaries, optimizer])
        summary_writer.add_summary(summary, step)

        if step % 50 == 0:
            print "\rLoss for step %i: %f" % (step, sess.run(loss))
            cv2.imwrite('images/result.png',
                        cv2.cvtColor(sess.run(image).reshape(image_shape),
                                     cv2.COLOR_RGB2BGR))

    print '\rFinal Loss: %f' % sess.run(loss)
    cv2.imwrite('images/result.png',
                cv2.cvtColor(sess.run(image).reshape(image_shape),
                             cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    tf.app.run(main)
