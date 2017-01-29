
import tensorflow as tf
import tf_util
import numpy as np

THENET_COLLECTION = 'the_net_collection'


class TheNet:

    def create_model(self, x, trainable=False, reuse=False):

        # First big conv layer
        with tf.variable_scope('conv1_1'):
            model = tf_util.conv_layer(x, filter_size=(5, 5), num_features=32, strides=(1, 1),
                                       trainable=trainable, collection=THENET_COLLECTION)
            model = tf_util.instance_norm(model)

        with tf.variable_scope('conv2_1'):
            model = tf_util.conv_layer(model, filter_size=(3, 3), num_features=64, strides=(2, 2),
                                       trainable=trainable, collection=THENET_COLLECTION)
            model = tf_util.instance_norm(model)

        with tf.variable_scope('conv2_2'):
            model = tf_util.conv_layer(model, filter_size=(3, 3), num_features=128, strides=(2, 2),
                                       trainable=trainable, collection=THENET_COLLECTION)
            model = tf_util.instance_norm(model)

        with tf.variable_scope('res_1'):
            model = tf_util.res_layer(model, filter_size=(3, 3), trainable=trainable, collection=THENET_COLLECTION)

        # with tf.variable_scope('res_2'):
        #     model = tf_util.res_layer(model, filter_size=(3, 3), trainable=trainable, collection=THENET_COLLECTION)

        with tf.variable_scope('deconv_1'):
            model = tf_util.conv_layer(model, filter_size=(3, 3), num_features=64, trainable=trainable, strides=(2, 2),
                                       deconv=True, upscale=2, collection=THENET_COLLECTION)
            model = tf_util.instance_norm(model)

        with tf.variable_scope('deconv_2'):
            model = tf_util.conv_layer(model, filter_size=(3, 3), num_features=3, trainable=trainable, strides=(2, 2),
                                       deconv=True, upscale=2, collection=THENET_COLLECTION)
            model = tf_util.instance_norm(model)
            model = 255.0 * tf.nn.sigmoid(model, name='activation')

        return model

    def save_model(self, weight_file, sess, base_scope=''):

        variables = tf.get_collection(THENET_COLLECTION)
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

        variables = tf.get_collection(THENET_COLLECTION)

        var_dict = np.load(weight_file)

        for name in var_dict.keys():

            if base_scope:
                target_variable = filter(lambda var: name in var.name and var.name.startswith(base_scope), variables)
            else:
                target_variable = filter(lambda var: name in var.name, variables)

            if len(target_variable) > 0:
                sess.run(target_variable[0].assign(var_dict[name]))

    def init_all_variables(self, sess):

        init_op = tf.variables_initializer(tf.get_collection(THENET_COLLECTION))
        sess.run(init_op)
