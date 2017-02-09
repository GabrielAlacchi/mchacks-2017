#!/usr/bin/python

import unet
import tensorflow as tf
import ops
from style_input import style_input
import art
from vgg import vgg16
import cv2
from os import path
import simplejson as json

import sys

flags = tf.app.flags

# Training parameters
flags.DEFINE_integer('batch_size', 3, 'Batch Size.')
flags.DEFINE_integer('epochs', 2, 'Training Epochs.')
flags.DEFINE_integer('max_steps', 40000, 'Max training steps. -1 for epoch based training')
flags.DEFINE_integer('report_step', 100, 'Step to report loss at')
flags.DEFINE_integer('save_step', 1000, 'Step to save a checkpoint at')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning Rate.')
flags.DEFINE_string('image_dir', 'data/img', 'Root where the test and train sub directories can be found')

# Tensorflow output
flags.DEFINE_string('logdir', 'logdir', 'Model and logs directory')
flags.DEFINE_string('model_name', 'vangogh', 'Model Name')
flags.DEFINE_string('restore', None, 'Checkpoint folder to restore from')
flags.DEFINE_string('restore_partial', None, 'Checkpoint folder to restore weights/biases (not gamma/beta) from')

flags.DEFINE_float('adam_beta1', 0.9, 'Beta 1 for Adam Optimizer.')
flags.DEFINE_float('adam_beta2', 0.999, 'Beta 2 for Adam Optimizer.')
flags.DEFINE_float('adam_epsilon', 1e-1, 'Epsilon for Adam Optimizer.')

# Style parameters
flags.DEFINE_string('art_list', 'images/art_list.json', 'The Images to Train Style for in a json file.')
flags.DEFINE_float('content_alpha', art.ALPHA, 'The weight for the content loss in the total loss.')
flags.DEFINE_float('style_beta', art.BETA, 'The weight for the style loss in the total loss.')
flags.DEFINE_float('tv_weight', art.TV_WEIGHT, 'The weight for the total variation in the total loss.')

FLAGS = flags.FLAGS


def gram_stack(layer_tuple, style_indices):
    layer_name, layer_matrices = layer_tuple
    with tf.name_scope(layer_name):
        stack = tf.stack(layer_matrices, axis=0, name='gram_matrices')
        return tf.gather(stack, style_indices, name='indexed_gram_matrices')


def stack_gram_matrices(art_images, art_list, style_layers, style_indices, vgg_scope, sess):

    with tf.name_scope('gram_matrices'):
        # Create a list of empty lists the size of style_layers
        gram_layer_list = [[] for _ in style_layers]
        for art_image, art_name in zip(art_images, art_list):
            # Precompute the gram matrices for this art_image
            print "Computing gram matrices for %s" % art_name
            _, gram_matrices = art.precompute(style_layers, [],
                                              art_image=art_image, user_image=None,
                                              vgg_scope=vgg_scope, sess=sess)
            for i, matrix in enumerate(gram_matrices):
                gram_layer_list[i].append(matrix)

        # Create stacks out of each layer, and gather the corresponding indices by label
        return map(lambda layer_tuple: gram_stack(layer_tuple, style_indices),
                   zip(style_layers, gram_layer_list))


def main(argv):

    with open(FLAGS.art_list, 'r') as f:
        art_list = json.load(f)

    art_dir = path.dirname(FLAGS.art_list)

    # Read all the art image files
    print "Loading artworks..."
    art_images = map(lambda name: cv2.imread(path.join(art_dir, name)), art_list)

    # Open CV works with BGR but we can't have it... We can't have it...... Can't have it **shrugs**
    art_images = map(lambda im: cv2.cvtColor(im, code=cv2.COLOR_BGR2RGB), art_images)

    num_styles = len(art_images)

    print "Modeling %d artworks..." % num_styles

    art_index = {}
    for index, filename in enumerate(art_list):
        art_name = path.basename(filename).split('.')[0]
        art_index[art_name] = index

    # Write the art_index to a file so that it can be used later
    with open(path.join(FLAGS.logdir, 'art_index.json'), 'w') as f:
        json.dump(art_index, f)

    with tf.Session() as sess:

        u_net = unet.UNet()

        train_image_dir = path.join(FLAGS.image_dir, 'train2014')

        # Get the symbolic COCO input tensors
        train_batch = style_input(train_image_dir, batch_size=FLAGS.batch_size, num_styles=num_styles, read_threads=2)

        # Get the input batches
        image_batch, style_indices = train_batch

        global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)

        with tf.variable_scope('unet'):
            # Create the model with sliced instance norm
            model = u_net.create_model(image_batch, num_styles,
                                       style_indices=style_indices, sliced=True, trainable=True)

        print "Initializing unet variables"
        sess.run(u_net.variables_initializer())

        print "Initializing vgg"
        with tf.variable_scope('vgg'):
            vgg_model = vgg16(model, reuse=False)
            vgg_model.load_weights('weights/vgg16_weights.npz', sess)

        # Re use the vgg model to get the content feature matrices
        with tf.variable_scope('vgg'):
            vgg_batch = vgg16(image_batch, reuse=True)

        style_layers = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']
        content_layers = ['conv2_2']

        # Get the gram matrix stacks
        gram_matrix_stacks = stack_gram_matrices(art_images, art_list, style_indices=style_indices,
                                                 vgg_scope='vgg', style_layers=style_layers, sess=sess)

        # Index the stack by style index
        gram_matrices = map(lambda stack: stack, gram_matrix_stacks)

        # Get the style and content ops from the vgg instance running on top of the UNet model.
        content_layer_ops = map(lambda layer: vgg_model.get_layer(layer), content_layers)
        style_layer_ops = map(lambda layer: vgg_model.get_layer(layer), style_layers)

        # Get the feature matrices of the training batch operation
        feature_matrices = map(lambda layer: art.feature_matrix(vgg_batch.get_layer(layer)), content_layers)

        # Compute the total loss, including summaries
        loss = art.total_loss(model, content_layer_ops, style_layer_ops, feature_matrices, gram_matrices,
                              alpha=FLAGS.content_alpha, beta=FLAGS.style_beta,
                              tv_weight=FLAGS.tv_weight, total_variation=False,
                              summaries=True, summary_scope='summaries')

        unet_variables = tf.get_collection(unet.UNET_COLLECTION)

        # Chilling out with the weight summaries
        # with tf.name_scope('weight_summaries'):
        #    print "Creating Variable Summaries..."
        #    map(ops.variable_summaries, unet_variables)

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
            # If we want to restore everything.
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.restore)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Restored from: %s" % FLAGS.restore
        elif FLAGS.restore_partial:
            # If we just want to restore the weights and biases (not instance norm parameters)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.restore_partial)
            u_net.restore_partial(ckpt.model_checkpoint_path, sess)

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
