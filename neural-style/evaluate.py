
import tensorflow as tf
import numpy as np
import unet
import cv2
from scipy.misc import imsave

flags = tf.app.flags

flags.DEFINE_string('logdir', 'logdir', 'Logdir where checpoints can be found')
flags.DEFINE_string('input_image', 'images/gabriel.jpg', 'Image to stylize')

FLAGS = flags.FLAGS


def main(argv):

    image = cv2.imread(FLAGS.input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(348, 348))
    image_tens = tf.expand_dims(tf.constant(image, dtype=tf.float32), axis=0)
    style_weights = tf.constant([1.0], dtype=tf.float32)

    with tf.variable_scope('unet'):
        net = unet.UNet()
        model = net.create_model(image_tens, style_weights, trainable=False)

    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        checkpoints = tf.train.get_checkpoint_state(FLAGS.logdir)
        unet_variables = tf.get_collection(unet.UNET_COLLECTION)
        saver = tf.train.Saver(unet_variables)
        saver.restore(sess, checkpoints.model_checkpoint_path)

        output = sess.run(model)
        output = np.squeeze(output)

        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite('images/result.jpg', output)

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    tf.app.run(main)
