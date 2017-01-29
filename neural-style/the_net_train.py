
from the_net import TheNet
from unet import UNet
import tensorflow as tf
import numpy as np
import sys
from data_set import create_data_sets
import art
from vgg import vgg16
import cv2

import argparse

BATCH_SIZE = 25
EPOCHS = 1
LEARNING_RATE = 1e-3


def main(argv):

    argparser = argparse.ArgumentParser(prog='Training script')
    argparser.add_argument('-m', '--model', default='thenet',
                           dest='model',
                           help='Model to use for training, options: thenet, unet')
    argparser.add_argument('-a', '--art-image', default='images/starry_night.jpg',
                           dest='art_image',
                           help='Art image to train on')
    argparser.add_argument('-s', '--save-dest', default='weights/starry_night.npz',
                           dest='save_dest',
                           help='Save destination')
    argparser.add_argument('-e', '--epochs', default=EPOCHS, type=int,
                           dest='epochs',
                           help='Number of epochs to train')
    argparser.add_argument('-l', '--learning-rate', default=LEARNING_RATE, type=float,
                           dest='learning_rate',
                           help='Learning rate to train with')
    argparser.add_argument('-b', '--batch-size', default=BATCH_SIZE, type=int,
                           dest='batch_size',
                           help='Batch Size')

    args = argparser.parse_args(argv)

    art_image = cv2.imread(args.art_image)

    sess = tf.Session()

    if args.model == 'thenet':
        thenet = TheNet()
    else:
        thenet = UNet()

    image_pl = tf.placeholder(dtype=tf.float32, shape=(args.batch_size, 224, 224, 3))

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

    train_step = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

    sess.run(tf.global_variables_initializer())

    training, testing = create_data_sets('data', training_reserve=0.3, testing_reserve=0.7)
    training.set_batch_size(args.batch_size)
    testing.set_batch_size(args.batch_size)

    num_steps_per_epoch = training.get_epoch_steps()
    epochs_to_train = args.epochs
    reporting_step = 100

    print "Running %d epochs with %d training steps per epoch" % (epochs_to_train, num_steps_per_epoch)
    for epoch in xrange(epochs_to_train):

        training.shuffle()

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

        print "\nFinal results for epoch %d --- loss: %f" % (epoch, loss_val)

    print "-----------------------------------------------"

    thenet.save_model(args.save_dest, base_scope='starry_night', sess=sess)

if __name__ == "__main__":
    main(sys.argv[1:])
