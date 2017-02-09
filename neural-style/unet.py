
import tensorflow as tf
import ops

UNET_COLLECTION = 'unet_collection'


class UNet:

    def create_model(self, x, num_styles, sliced=False, style_indices=None,
                     style_weights=None, trainable=False, reuse=False):
        """
        Creates the UNet model
        :param x: The input tensor (batch, height, width, depth)
        :param num_styles: The number of styles to train the model for
        :param sliced: Whether to apply sliced instance norm or weighted instance norm
        :param style_indices: A tensor specifying the list of indices of style to use (Necessary if using condiitonal = True). The size of it must match the batch size.
        :param style_weights: A tensor specifying the style weights (Necessary if using conditional = False)
        :param trainable: Whether or not the variables should be trainable
        :param reuse: Whether or not to reuse the scoped variables
        :return: A tensor with the output image batch of the model
        """

        if style_weights is not None:
            style_weights = tf.expand_dims(style_weights, axis=1, name='expanded_style_weights')

        # Decides which normalization op to use (sliced is for a single style) (model is for blending styles)
        def _norm(_input):
            if sliced:
                return ops.conditional_instance_norm(_input, style_indices, num_styles=num_styles,
                                                     trainable=trainable,
                                                     collection=UNET_COLLECTION)
            else:
                return ops.weighted_instance_norm(_input, style_weights,
                                                  trainable=trainable,
                                                  collection=UNET_COLLECTION)

        with tf.variable_scope('conv1_1'):
            model = ops.conv_layer(x, filter_size=(5, 5), num_features=32, strides=(1, 1),
                                   trainable=trainable, collection=UNET_COLLECTION)
            model = _norm(model)

        with tf.variable_scope('conv1_2', reuse=reuse):
            model = conv1_1 = ops.conv_layer(model, filter_size=(3, 3), num_features=32, strides=(1, 1),
                                             trainable=trainable, collection=UNET_COLLECTION)
            model = _norm(model)

        with tf.variable_scope('downsample_1', reuse=reuse):
            model = ops.conv_layer(model, filter_size=(3, 3), num_features=64, strides=(2, 2),
                                   trainable=trainable, collection=UNET_COLLECTION)
            model = _norm(model)

        with tf.variable_scope('conv2_1', reuse=reuse):
            model = conv2_1 = ops.conv_layer(model, filter_size=(3, 3), num_features=64, strides=(1, 1),
                                             trainable=trainable, collection=UNET_COLLECTION)
            model = _norm(model)

        with tf.variable_scope('downsample_2', reuse=reuse):
            model = ops.conv_layer(model, filter_size=(3, 3), num_features=128, strides=(2, 2),
                                   trainable=trainable, collection=UNET_COLLECTION)
            model = _norm(model)

        with tf.variable_scope('res_1', reuse=reuse):
            model = ops.res_layer(model, filter_size=(3, 3), trainable=trainable, collection=UNET_COLLECTION)
            model = _norm(model)

        with tf.variable_scope('res_2', reuse=reuse):
            model = ops.res_layer(model, filter_size=(3, 3), trainable=trainable, collection=UNET_COLLECTION)
            model = _norm(model)

        with tf.variable_scope('res_3', reuse=reuse):
            model = ops.res_layer(model, filter_size=(3, 3), trainable=trainable, collection=UNET_COLLECTION)
            model = _norm(model)

        with tf.variable_scope('res_4', reuse=reuse):
            model = ops.res_layer(model, filter_size=(3, 3), trainable=trainable, collection=UNET_COLLECTION)
            model = _norm(model)

        with tf.variable_scope('res_5', reuse=reuse):
            model = ops.res_layer(model, filter_size=(3, 3), trainable=trainable, collection=UNET_COLLECTION)
            model = _norm(model)

        with tf.variable_scope('upsample_1', reuse=reuse):
            model = ops.upsample(model, filter_size=(3, 3), num_features=64, trainable=trainable,
                                 upscale=2, collection=UNET_COLLECTION)
            model = tf.concat(values=[conv2_1, model], concat_dim=3)
            model = _norm(model)

        with tf.variable_scope('upsample_2', reuse=reuse):
            model = ops.upsample(model, filter_size=(3, 3), num_features=32, trainable=trainable,
                                 upscale=2, collection=UNET_COLLECTION)
            model = tf.concat(values=[conv1_1, model], concat_dim=3)
            model = _norm(model)

        with tf.variable_scope('conv_output', reuse=reuse):
            model = ops.conv_layer(model, filter_size=(3, 3), num_features=3, strides=(1, 1), relu=False,
                                   trainable=trainable, collection=UNET_COLLECTION)
            model = 255.0 * tf.nn.sigmoid(model, name='activation')

        return model

    def restore_partial(self, checkpoint_file, sess):
        variables_to_restore = [var for var in tf.get_collection(UNET_COLLECTION)
                                if 'gamma' not in var.name and 'beta' not in var.name]
        saver = tf.train.Saver(variables_to_restore)
        print "Restoring weights from %s" % checkpoint_file
        saver.restore(sess, checkpoint_file)

    def variables_initializer(self):
        return tf.variables_initializer(tf.get_collection(UNET_COLLECTION))
