
from the_net import TheNet
import tensorflow as tf
import numpy as np
import art
from vgg import vgg16
from scipy.misc import imread

BATCH_SIZE = 3


def main(argv):

    art_image = imread('image/starry_night.jpg')

    sess = tf.Session()

    thenet = TheNet()

    x = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, 224, 224, 3))
    model = thenet.create_model(x, trainable=True)

    thenet.init_all_variables(sess)

    with tf.variable_scope('vgg'):
        vgg = vgg16(model, reuse=False)
        vgg.load_weights('vgg16_weights.npz', sess)

    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    content_layers = ['conv3_2', 'conv4_2']

    _, gram_matrices = art.precompute(style_layers, [], art_image=art_image, user_image=None, vgg_scope='vgg', sess=sess)

    content_layer_ops = map(lambda layer: vgg.get_layer(layer), content_layers)
    style_layer_ops = map(lambda layer: vgg.get_layer(layer), style_layers)

    feature_matrices = map(lambda layer: art.feature_matrix(layer), content_layer_ops)

    loss = art.total_loss(model, content_layer_ops, style_layer_ops, feature_matrices, gram_matrices)

