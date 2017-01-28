
import tensorflow as tf
from the_net import TheNet
from os import path
import argparse


def init_models(model_files, sess):

    thenet = TheNet()
    x = tf.placeholder(dtype=tf.float32, shape=(1, None, None, 3), name='placeholder')

    for model_file in model_files:
        model_name = path.basename(model_file).replace('.npz', '')

        with tf.variable_scope(model_name):
            thenet.create_model(x, trainable=False)
            thenet.load_model(model_file, sess)


def consume_model(image, sess, model_name):

    batch_shape = [1] + list(image.shape)
    x = tf.placeholder(dtype=tf.float32, shape=batch_shape, name='input_image')
    thenet = TheNet()

    with tf.variable_scope(model_name, reuse=True):
        model = thenet.create_model(x, trainable=False, reuse=True)

    transformed_image = sess.run(model, feed_dict={
        x: image.reshape(batch_shape)
    })

    return transformed_image.reshape(image.shape)

