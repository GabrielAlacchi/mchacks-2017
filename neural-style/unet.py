
import tensorflow as tf
import ops
import numpy as np

UNET_COLLECTION = 'unet_collection'


class UNet:

    def create_model(self, x, style_weights, trainable=False, reuse=False):

        with tf.variable_scope('conv1_1', reuse=reuse):
            model = conv1_1 = ops.conv_layer(x, filter_size=(3, 3), num_features=32, strides=(1, 1),
                                             trainable=trainable, collection=UNET_COLLECTION)
            model = ops.weighted_instance_norm(model, style_weights, trainable=trainable, collection=UNET_COLLECTION)

        with tf.variable_scope('conv1_2', reuse=reuse):
            model = ops.conv_layer(model, filter_size=(3, 3), num_features=64, strides=(2, 2),
                                   trainable=trainable, collection=UNET_COLLECTION)
            model = ops.weighted_instance_norm(model, style_weights, trainable=trainable, collection=UNET_COLLECTION)

        with tf.variable_scope('conv2_1', reuse=reuse):
            model = conv2_1 = ops.conv_layer(model, filter_size=(3, 3), num_features=64, strides=(1, 1),
                                             trainable=trainable, collection=UNET_COLLECTION)
            model = ops.weighted_instance_norm(model, style_weights, trainable=trainable, collection=UNET_COLLECTION)

        with tf.variable_scope('conv2_2', reuse=reuse):
            model = ops.conv_layer(model, filter_size=(3, 3), num_features=128, strides=(2, 2),
                                   trainable=trainable, collection=UNET_COLLECTION)
            model = ops.weighted_instance_norm(model, style_weights, trainable=trainable, collection=UNET_COLLECTION)

        with tf.variable_scope('res_1', reuse=reuse):
            model = ops.res_layer(model, filter_size=(3, 3), trainable=trainable, collection=UNET_COLLECTION)
            model = ops.weighted_instance_norm(model, style_weights, trainable=trainable, collection=UNET_COLLECTION)

        with tf.variable_scope('res_2', reuse=reuse):
            model = ops.res_layer(model, filter_size=(3, 3), trainable=trainable, collection=UNET_COLLECTION)
            model = ops.weighted_instance_norm(model, style_weights, trainable=trainable, collection=UNET_COLLECTION)

        with tf.variable_scope('res_3', reuse=reuse):
            model = ops.res_layer(model, filter_size=(3, 3), trainable=trainable, collection=UNET_COLLECTION)
            model = ops.weighted_instance_norm(model, style_weights, trainable=trainable, collection=UNET_COLLECTION)

        with tf.variable_scope('upsample_1', reuse=reuse):
            model = ops.upsample(model, filter_size=(3, 3), num_features=64, trainable=trainable,
                                 upscale=2, collection=UNET_COLLECTION)
            model = tf.concat(values=[conv2_1, model], concat_dim=3)
            model = ops.weighted_instance_norm(model, style_weights, trainable=trainable, collection=UNET_COLLECTION)

        with tf.variable_scope('upsample_2', reuse=reuse):
            model = ops.upsample(model, filter_size=(3, 3), num_features=32, trainable=trainable,
                                 upscale=2, collection=UNET_COLLECTION)
            model = tf.concat(values=[conv1_1, model], concat_dim=3)
            model = ops.weighted_instance_norm(model, style_weights, trainable=trainable, collection=UNET_COLLECTION)

        with tf.variable_scope('conv_output', reuse=reuse):
            model = ops.conv_layer(model, filter_size=(3, 3), num_features=3, strides=(1, 1), relu=False,
                                   trainable=trainable, collection=UNET_COLLECTION)
            model = 255.0 * tf.nn.sigmoid(model, name='activation')

        return model

    def variables_initializer(self):
        return tf.variables_initializer(tf.get_collection(UNET_COLLECTION))
