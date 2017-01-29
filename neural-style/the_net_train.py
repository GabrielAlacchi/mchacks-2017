
from the_net import TheNet
import tensorflow as tf
import numpy as np
import sys
from data_set import create_data_sets
import art
from vgg import vgg16
from scipy.misc import imread

BATCH_SIZE = 3
EPOCHS = 1
LEARNING_RATE = 1e-3


def main(argv):

    art_image = imread('images/starry_night.jpg')

    sess = tf.Session()

    thenet = TheNet()

    image_pl = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, 224, 224, 3))

    with tf.variable_scope('starry_night'):
        model = thenet.create_model(image_pl, trainable=True)

    thenet.init_all_variables(sess)

    with tf.variable_scope('vgg'):
        vgg = vgg16(model, reuse=False)
        vgg.load_weights('weights/vgg16_weights.npz', sess)

    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    content_layers = ['conv3_2', 'conv4_2']

    _, gram_matrices = art.precompute(style_layers, [], art_image=art_image, user_image=None, vgg_scope='vgg', sess=sess)

    content_layer_ops = map(lambda layer: vgg.get_layer(layer), content_layers)
    style_layer_ops = map(lambda layer: vgg.get_layer(layer), style_layers)

    feature_matrices = map(lambda layer: art.feature_matrix(layer), content_layer_ops)

    loss = art.total_loss(model, content_layer_ops, style_layer_ops, feature_matrices, gram_matrices)

    train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    training, testing = create_data_sets('data')

    num_steps_per_epoch = training.get_epoch_steps()
    epochs_to_train = EPOCHS
    reporting_step = 100

    print "Running %d epochs with %d training steps per epoch" % (epochs_to_train, num_steps_per_epoch)
    for epoch in xrange(epochs_to_train):

        print "Epoch: %d --- Epochs left: %d" % (epoch, epochs_to_train - epoch)
        print "-----------------------------------------------"

        for step in xrange(num_steps_per_epoch):
            sys.stdout.flush()
            sys.stdout.write('\rStep: %d' % (step + 1))

            batch_images = training.next_batch()
            sess.run(train_step, feed_dict={
                image_pl: batch_images
            })

            if step % reporting_step == 0:
                loss_val = sess.run(loss, feed_dict={
                    image_pl: batch_images
                })

                print "\r    Step %d --- loss: %f" % (step, loss_val)

        test_images = testing.next_batch()
        loss_val = sess.run(loss, feed_dict={
            image_pl: test_images,
        })

        print "\nFinal results for epoch %d --- loss: %f --- accuracy: %f" % (epoch, loss_val, accuracy)

    print "-----------------------------------------------"

    print "Loading validation set for final accuracy..."

    final_loss = 0.0
    final_accuracy = 0.0

    for step in xrange(testing.get_epoch_steps()):
        validation_images, validation_labels = testing.next_batch()
        loss_val, accuracy = sess.run(loss, feed_dict={
            image_pl: validation_images
        })

        final_loss += loss_val
        final_accuracy += accuracy

    final_loss /= testing.get_epoch_steps()

    print "Final Loss: %f" % final_loss

    thenet.save_model('weights/starry_night.npz', base_scope='starry_night', sess=sess)

if __name__ == "__main__":
    main(sys.argv[1:])
