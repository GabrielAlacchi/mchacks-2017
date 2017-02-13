
import tensorflow as tf
import numpy as np
import model
import cv2
import simplejson as json
from os import path

flags = tf.app.flags

flags.DEFINE_string('logdir', 'logdir', 'Logdir where checpoints can be found')
flags.DEFINE_string('input_image', 'images/montreal.jpg', 'Image to stylize')
flags.DEFINE_string('outfile', 'images/result.jpg', 'Output file')
flags.DEFINE_string('style_name', 'mosaic', 'Style name to use.')

FLAGS = flags.FLAGS


def main(argv):

    image = cv2.imread(FLAGS.input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(path.join(FLAGS.logdir, 'art_index.json')) as f:
        art_index = json.load(f)

    num_styles = len(art_index.keys())
    style_name = FLAGS.style_name

    if style_name not in art_index:
        print "Style is not in the model %s" % style_name
        exit(-1)
    else:
        print "Using style %s" % style_name

    style_index = art_index[style_name]

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

    style_weight_list = [0.0 for _ in xrange(num_styles)]
    style_weight_list[style_index] = 1.0

    style_weights = tf.constant(style_weight_list, dtype=tf.float32)

    with tf.variable_scope('unet'):
        transform = model.transform(image_tens, num_styles, style_weights=style_weights, trainable=False)

    with tf.Session() as sess:

        checkpoints = tf.train.get_checkpoint_state(FLAGS.logdir)
        unet_variables = tf.get_collection(model.MODEL_COLLECTION)
        saver = tf.train.Saver(unet_variables)
        saver.restore(sess, checkpoints.model_checkpoint_path)

        output = sess.run(transform)
        output = np.squeeze(output)

        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite(FLAGS.outfile, output)


if __name__ == "__main__":
    tf.app.run(main)
