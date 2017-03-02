
import tensorflow as tf
from os import path, listdir


def read_jpg(filename_queue, image_size):
    reader = tf.WholeFileReader(name='file_reader')
    key, record_string = reader.read(filename_queue)
    example = tf.image.decode_jpeg(record_string, name='decode_jpeg', channels=3)

    # Cast and reshape (This assumes the images are 224 * 224 * 3).
    example = tf.image.resize_image_with_crop_or_pad(example, *image_size)
    return tf.cast(example, dtype=tf.float32, name='training_image')


def style_input(train_image_dir, batch_size, image_size=None, min_after_dequeue=5, read_threads=1, num_styles=1):

    if not image_size:
        image_size = (256, 256)

    train_filenames = [path.join(train_image_dir, f) for f in listdir(train_image_dir)]
    train_filenames = tf.constant(train_filenames, name='filenames')

    train_queue = tf.train.string_input_producer(train_filenames,
                                                 shuffle=True)

    # Get the range 0 ... num_styles as the possible style_indices
    style_index_queue = tf.train.range_input_producer(num_styles, shuffle=True)
    style_example = style_index_queue.dequeue()

    image_example = read_jpg(train_queue, image_size=image_size)

    capacity = min_after_dequeue + 3 * batch_size

    image_batch, style_indices = tf.train.shuffle_batch([image_example, style_example], batch_size=batch_size,
                                                        capacity=capacity, num_threads=read_threads, min_after_dequeue=min_after_dequeue)

    return image_batch, style_indices