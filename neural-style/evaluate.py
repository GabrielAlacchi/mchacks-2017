
import tensorflow as tf
import numpy as np
import unet
import cv2

flags = tf.app.flags

flags.DEFINE_string('logdir', 'logdir', 'Logdir where checpoints can be found')
flags.DEFINE_string('input_image', 'images/montreal.jpg', 'Image to stylize')
flags.DEFINE_string('outfile', 'images/result.jpg', 'Output file')
flags.DEFINE_string('style_weights', '1.0,0.0', 'Style weights to use')

FLAGS = flags.FLAGS


def main(argv):

    image = cv2.imread(FLAGS.input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    im_shape = list(image.shape)
    resize = False
    if im_shape[0] % 4 != 0:
        im_shape[0] -= im_shape[0] % 4
        resize = True
    if im_shape[1] % 4 != 0:
        im_shape[1] -= im_shape[1] % 4
        resize = True

    if resize:
        image = cv2.resize(image, dsize=(im_shape[1], im_shape[0]))

    image_tens = tf.expand_dims(tf.constant(image, dtype=tf.float32), axis=0)

    style_weight_list = map(float, FLAGS.style_weights.split(','))

    style_weights = tf.constant(style_weight_list, dtype=tf.float32)

    with tf.variable_scope('unet'):
        net = unet.UNet()
        model = net.create_model(image_tens, len(style_weight_list), style_weights=style_weights, trainable=False)

    with tf.Session() as sess:

        checkpoints = tf.train.get_checkpoint_state(FLAGS.logdir)
        unet_variables = tf.get_collection(unet.UNET_COLLECTION)
        saver = tf.train.Saver(unet_variables)
        saver.restore(sess, checkpoints.model_checkpoint_path)

        output = sess.run(model)
        output = np.squeeze(output)

        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite(FLAGS.outfile, output)


if __name__ == "__main__":
    tf.app.run(main)
