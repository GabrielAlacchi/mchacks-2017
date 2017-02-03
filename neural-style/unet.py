
import tensorflow as tf
import ops
import numpy as np

UNET_COLLECTION = 'unet_collection'


class UNet:

    def create_model(self, x, style_weights, trainable=False, reuse=False):

        # First big conv layer
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

        with tf.variable_scope('deconv_1', reuse=reuse):
            model = ops.conv_layer(model, filter_size=(3, 3), num_features=64, trainable=trainable, strides=(2, 2),
                                   deconv=True, upscale=2, collection=UNET_COLLECTION)
            model = tf.concat(values=[conv2_1, model], concat_dim=3)
            model = ops.weighted_instance_norm(model, style_weights, trainable=trainable, collection=UNET_COLLECTION)

        with tf.variable_scope('deconv_2', reuse=reuse):
            model = ops.conv_layer(model, filter_size=(3, 3), num_features=32, trainable=trainable, strides=(2, 2),
                                   deconv=True, upscale=2, collection=UNET_COLLECTION)
            model = tf.concat(values=[conv1_1, model], concat_dim=3)
            model = ops.weighted_instance_norm(model, style_weights, trainable=trainable, collection=UNET_COLLECTION)

        with tf.variable_scope('conv_output', reuse=reuse):
            model = ops.conv_layer(model, filter_size=(3, 3), num_features=3, strides=(1, 1),
                                   trainable=trainable, collection=UNET_COLLECTION)
            model = 255.0 * tf.nn.sigmoid(model, name='activation')

        return model

    def save_model(self, weight_file, sess, base_scope=''):

        variables = tf.get_collection(UNET_COLLECTION)
        variable_names = map(lambda var: var.name, variables)

        var_dict = {}
        for name, var in zip(variable_names, variables):

            if base_scope:
                final_name = name.replace('%s/' % base_scope, '')
            else:
                final_name = name

            final_name = final_name.split(':')[0]
            var_dict[final_name] = sess.run(var)

        np.savez(weight_file, **var_dict)

    def load_model(self, weight_file, sess, base_scope=''):

        variables = tf.get_collection(UNET_COLLECTION)

        var_dict = np.load(weight_file)

        for name in var_dict.keys():

            if base_scope:
                target_variable = filter(lambda var: name in var.name and var.name.startswith(base_scope), variables)
            else:
                target_variable = filter(lambda var: name in var.name, variables)

            if len(target_variable) > 0:
                sess.run(target_variable[0].assign(var_dict[name]))

    def init_all_variables(self, sess):

        init_op = tf.variables_initializer(tf.get_collection(UNET_COLLECTION))
        sess.run(init_op)
