#!/usr/bin/python

import unet
import tensorflow as tf
import ops
from style_input import style_input
import art
from vgg import vgg16
import cv2
from os import path

import sys

flags = tf.app.flags

# Training parameters
flags.DEFINE_integer('batch_size', 5, 'Batch Size.')
flags.DEFINE_integer('epochs', 2, 'Training Epochs.')
flags.DEFINE_integer('max_steps', 40000, 'Max training steps. -1 for epoch based training')
flags.DEFINE_integer('report_step', 100, 'Step to report loss at')
flags.DEFINE_integer('save_step', 10000, 'Step to save a checkpoint at')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning Rate.')
flags.DEFINE_string('image_dir', 'data/img', 'Root where the test and train sub directories can be found')

# Tensorflow output
flags.DEFINE_string('logdir', 'logdir', 'Model and logs directory')
flags.DEFINE_string('model_name', 'vangogh', 'Model Name')
flags.DEFINE_string('restore', None, 'Checkpoint file to restore from')

flags.DEFINE_float('adam_beta1', 0.9, 'Beta 1 for Adam Optimizer.')
flags.DEFINE_float('adam_beta2', 0.999, 'Beta 2 for Adam Optimizer.')
flags.DEFINE_float('adam_epsilon', 1e-8, 'Epsilon for Adam Optimizer.')

# Style parameters
flags.DEFINE_string('art_image', 'images/the_starry_night.jpg', 'The Image to Train Style for.')
flags.DEFINE_float('content_alpha', art.ALPHA, 'The weight for the content loss in the total loss.')
flags.DEFINE_float('style_beta', art.BETA, 'The weight for the style loss in the total loss.')
flags.DEFINE_float('tv_weight', art.TV_WEIGHT, 'The weight for the total variation in the total loss.')

FLAGS = flags.FLAGS


def main(argv):

    art_image = cv2.imread(FLAGS.art_image)

    # Open CV works with BGR but we can't have it... We can't have it...... Can't have it **shrugs**
    art_image = cv2.cvtColor(art_image, code=cv2.COLOR_BGR2RGB)

    style_weights = tf.constant([1.0], dtype=tf.float32, name='style_weights')

    with tf.Session() as sess:

        u_net = unet.UNet()

        train_image_dir = path.join(FLAGS.image_dir, 'train2014')

        # Get the symbolic COCO input tensors
        train_batch = style_input(train_image_dir, read_threads=2,
                                  num_epochs=FLAGS.epochs, batch_size=FLAGS.batch_size)

        # Ignore labels
        train_example_batch, _ = train_batch

        with tf.variable_scope('unet'):
            model = u_net.create_model(train_example_batch, style_weights, trainable=True)

        print "Initializing unet variables"
        sess.run(u_net.variables_initializer())

        print "Initializing vgg"
        with tf.variable_scope('vgg'):
            vgg_model = vgg16(model, reuse=False)
            vgg_model.load_weights('weights/vgg16_weights.npz', sess)

        # Re use the vgg model to get the content feature matrices
        with tf.variable_scope('vgg'):
            vgg_batch = vgg16(train_example_batch, reuse=True)

        style_layers = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']
        content_layers = ['conv2_2']

        # Get the gram matrices for the image (they are tf constants)
        _, gram_matrices = art.precompute(style_layers, [], art_image=art_image, user_image=None, vgg_scope='vgg', sess=sess)

        # Get the style and content ops from the vgg instance running on top of the UNet model.
        content_layer_ops = map(lambda layer: vgg_model.get_layer(layer), content_layers)
        style_layer_ops = map(lambda layer: vgg_model.get_layer(layer), style_layers)

        # Get the feature matrices of the training batch operation
        feature_matrices = map(lambda layer: art.feature_matrix(vgg_batch.get_layer(layer)), content_layers)

        # Compute the total loss, including summaries
        loss = art.total_loss(model, content_layer_ops, style_layer_ops, feature_matrices, gram_matrices,
                              alpha=FLAGS.content_alpha, beta=FLAGS.style_beta,
                              tv_weight=FLAGS.tv_weight, total_variation=True,
                              summaries=True, summary_scope='summaries')

        unet_variables = tf.get_collection(unet.UNET_COLLECTION)
        with tf.name_scope('weight_summaries'):
            print "Creating Variable Summaries..."
            map(ops.variable_summaries, unet_variables)

        global_step = tf.Variable(0, trainable=False, name='global_step')

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=FLAGS.adam_beta1,
                                           beta2=FLAGS.adam_beta2, epsilon=FLAGS.adam_epsilon)
        train_step = optimizer.minimize(loss, global_step=global_step)

        saver = tf.train.Saver(unet_variables + [global_step])

        print "Merging summaries..."
        merged_summary = tf.summary.merge_all()

        print "Summaries will be written to directory: %s" % FLAGS.logdir
        summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

        print "Starting queue runners..."
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        local_vars = tf.local_variables()

        print "Initializing variables..."
        sess.run(tf.global_variables_initializer())
        sess.run(tf.variables_initializer(local_vars))

        if FLAGS.restore:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.restore)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Restored from: %s" % FLAGS.restore

        print "Starting training"
        try:
            while not coord.should_stop():
                # Run training step
                summary, _ = sess.run([merged_summary, train_step])
                step = tf.train.global_step(sess, global_step)

                summary_writer.add_summary(summary, step)

                sys.stdout.flush()
                sys.stdout.write('\r Step: %d' % step)
                if (step - 1) % FLAGS.report_step == 0:
                    loss_val = sess.run(loss)
                    print "\rLoss for step %d: %f" % (step, loss_val)
                if (step - 1) % FLAGS.save_step == 0:
                    saver.save(sess, path.join(FLAGS.logdir, FLAGS.model_name + '.ckpt'), global_step=global_step,
                               write_meta_graph=False)
                if step == FLAGS.max_steps:
                    break
        except tf.errors.OutOfRangeError:
            print 'Epoch limit reached.'
        finally:
            coord.request_stop()
            coord.join(threads)

        print "\rDone training"
        print "Exporting meta graph and final checkpoint..."

        saver.save(sess, path.join(FLAGS.logdir, FLAGS.model_name + '.ckpt'), global_step=global_step, write_meta_graph=True)

if __name__ == "__main__":
    tf.app.run(main)
