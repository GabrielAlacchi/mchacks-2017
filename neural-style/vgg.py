
import tensorflow as tf
import numpy as np
import os
import urllib
from scipy.misc import imread, imresize

from tf_util import kernel_variable, bias_variable


def download_weights_maybe(weight_file):
    if not os.path.exists(weight_file):
        print "Downloading weights from https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz"
        urllib.urlretrieve("https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz", weight_file)


class vgg16:
    def __init__(self, imgs, reuse=False):
        self.imgs = imgs
        self.convlayers(reuse)

    def convlayers(self, reuse=False):
        self.parameters = []

        # conv1_1
        with tf.variable_scope('conv1_1', reuse=reuse) as scope:
            kernel = kernel_variable('weights', shape=[3, 3, 3, 64],  trainable=False, collection='VGG_weights')
            conv = tf.nn.conv2d(self.imgs, kernel, [1, 1, 1, 1], padding='SAME')
            biases = bias_variable('biases', shape=[64], trainable=False, collection='VGG_weights')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.variable_scope('conv1_2', reuse=reuse) as scope:
            kernel = kernel_variable('weights', shape=[3, 3, 64, 64],  trainable=False, collection='VGG_weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = bias_variable('biases', shape=[64], trainable=False, collection='VGG_weights')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.avg_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.variable_scope('conv2_1', reuse=reuse) as scope:
            kernel = kernel_variable('weights', shape=[3, 3, 64, 128],  trainable=False, collection='VGG_weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = bias_variable('biases', shape=[128], trainable=False, collection='VGG_weights')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.variable_scope('conv2_2', reuse=reuse) as scope:
            kernel = kernel_variable('weights', shape=[3, 3, 128, 128],  trainable=False, collection='VGG_weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = bias_variable('biases', shape=[128], trainable=False, collection='VGG_weights')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.avg_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.variable_scope('conv3_1', reuse=reuse) as scope:
            kernel = kernel_variable('weights', shape=[3, 3, 128, 256],  trainable=False, collection='VGG_weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = bias_variable('biases', shape=[256], trainable=False, collection='VGG_weights')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.variable_scope('conv3_2', reuse=reuse) as scope:
            kernel = kernel_variable('weights', shape=[3, 3, 256, 256],  trainable=False, collection='VGG_weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = bias_variable('biases', shape=[256], trainable=False, collection='VGG_weights')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.variable_scope('conv3_3', reuse=reuse) as scope:
            kernel = kernel_variable('weights', shape=[3, 3, 256, 256],  trainable=False, collection='VGG_weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = bias_variable('biases', shape=[256], trainable=False, collection='VGG_weights')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.avg_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.variable_scope('conv4_1', reuse=reuse) as scope:
            kernel = kernel_variable('weights', shape=[3, 3, 256, 512],  trainable=False, collection='VGG_weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = bias_variable('biases', shape=[512], trainable=False, collection='VGG_weights')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.variable_scope('conv4_2', reuse=reuse) as scope:
            kernel = kernel_variable('weights', shape=[3, 3, 512, 512],  trainable=False, collection='VGG_weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = bias_variable('biases', shape=[512], trainable=False, collection='VGG_weights')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.variable_scope('conv4_3', reuse=reuse) as scope:
            kernel = kernel_variable('weights', shape=[3, 3, 512, 512],  trainable=False, collection='VGG_weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = bias_variable('biases', shape=[512], trainable=False, collection='VGG_weights')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.avg_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.variable_scope('conv5_1', reuse=reuse) as scope:
            kernel = kernel_variable('weights', shape=[3, 3, 512, 512],  trainable=False, collection='VGG_weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = bias_variable('biases', shape=[512], trainable=False, collection='VGG_weights')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.variable_scope('conv5_2', reuse=reuse) as scope:
            kernel = kernel_variable('weights', shape=[3, 3, 512, 512],  trainable=False, collection='VGG_weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = bias_variable('biases', shape=[512], trainable=False, collection='VGG_weights')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.variable_scope('conv5_3', reuse=reuse) as scope:
            kernel = kernel_variable('weights', shape=[3, 3, 512, 512],  trainable=False, collection='VGG_weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = bias_variable('biases', shape=[512], trainable=False, collection='VGG_weights')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.avg_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def get_layer(self, layer_name):
        return getattr(self, layer_name)

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i < len(self.parameters):
                print i, k, np.shape(weights[k])
                sess.run(self.parameters[i].assign(weights[k]))

if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs)
    vgg.load_weights('weights/vgg16_weights.npz', sess)

