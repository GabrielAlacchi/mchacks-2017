from os import path, listdir
import numpy as np
import random
import argparse
import sys, glob, cv2


def _load_list(image_names, image_dir, net_type):

    # TODO: This is where you would perform image resizing
    batch_images = map(lambda image_name: load_image(path.join(image_dir, image_name + '.jpg'), net_type=net_type), image_names)
    # Concatenate images and labels over batch dimension
    return np.concatenate(batch_images, axis=0)


class DataSet:

    def __init__(self, image_names, data_dir,  net_type ='VGG', batch_size = 1,image_size=None):
        # TODO: Implement image_size option so that if provided the images are resized after loaded.

        self.image_names = image_names
        self.data_dir = data_dir
        self.num_images = len(image_names)
        self.index = 0
        self.batch_size = batch_size
        self.image_size = image_size

        self.image_dir = path.join(self.data_dir, 'img')
        #self.label_dir = path.join(self.data_dir, 'label')
        self.net_type = net_type

        # Filter out images without labels (for whatever reason)
        #self.image_names = filter(lambda name: path.exists(path.join(self.label_dir, name + '.txt')), image_names)

        if self.num_images % self.batch_size != 0:
            print "Warning: The number of images %d is not divisible by the batch size %d, some images will be ignored" \
                  % (self.num_images, self.batch_size)
            self.index = 0

    def shuffle(self):
        if self.index != 0:
            print "Warning: Shuffling data set in the middle of an epoch"
        random.shuffle(self.image_names)

    def get_epoch_steps(self):
        return int(self.num_images / self.batch_size)

    def set_batch_size(self, batch_size):
        # TODO: Make sure this isn't being done after the first batch is loaded
        self.batch_size = batch_size

        if self.num_images % self.batch_size != 0:
            print "Warning: The number of images %d is not divisible by the batch size %d, some images will be ignored" \
                  % (self.num_images, self.batch_size)
            self.index = 0

    # This method simply returns a images as an array of (BATCH, DEPTH, WIDTH, HEIGHT) and labels as an array (BATCH, LABEL)
    def next_batch(self):
        # TODO: Implement some kind of cache for images so we can optimize the training process.
        # If possible we should offer an option to cache images directly into a GPU tensorflow variable (if we do tensorflow gpu training)

        batch_names = self.image_names[self.index:self.index+self.batch_size]

        # Increment the index
        self.index += self.batch_size

        if self.index + self.batch_size >= self.num_images:
            self.index = 0

        return _load_list(batch_names, image_dir=self.image_dir, net_type = self.net_type)

    def load_all(self):

        return _load_list(self.image_names, image_dir=self.image_dir, net_type = self.net_type)


# Returns training sets, validation sets, testing sets in that order as a tuple
def create_data_sets(data_dir, training_reserve=0.7, testing_reserve=0.3, net_type = "VGG"):

    # Make sure the proportions add to 1.0, if not warn the user and switch back to defaults
    if training_reserve + testing_reserve!= 1.0:
        print "Warning: Provided training, testing reserves do not add to one, switching back to defaults."
        training_reserve = 0.7
        testing_reserve = 0.3

    image_dir = path.join(data_dir, 'img')
    label_dir = path.join(data_dir, 'label')

    # Get the file name (without extension) of all the jpeg files in the image directory
    image_names = [path.basename(f).replace('.jpg', '') for f in listdir(image_dir) if f.endswith('.jpg')]

    # Shuffle up the images
    random.shuffle(image_names)

    num_training = int(training_reserve * len(image_names))
    num_testing = int(testing_reserve * len(image_names))

    i = 0
    training_names = image_names[i:i+num_training]
    i += num_training
    testing_names = image_names[i:i+num_testing]
    i += num_testing

    return (DataSet(training_names, data_dir, net_type),
            DataSet(testing_names, data_dir, net_type))


def resize_bulk(data_dir, img_size):
    image_dir = path.join(data_dir, 'img')
    files = listdir(image_dir)

    for image_file in files:
        print "Resizing %s" % path.basename(image_file)
        image_file = path.join(data_dir, 'img', image_file)
        im = cv2.imread(image_file)

        shape = im.shape

        left_point = int(shape[0] / 2 - 224 / 2)
        right_point = int(shape[0] / 2 + 224 / 2)
        top_point = int(shape[1] / 2 - 224 / 2)
        bottom_point = int(shape[1] / 2 + 224 / 2)
        im = im[left_point:right_point, top_point:bottom_point]

        cv2.imwrite(image_file, im)


def load(path_to_data):
    #takes a string pointing to a location that contains an img folder with raw images, 
    #and a label folder that contains the labels and bounding boxes as text files for each frame
    files = glob.glob(path_to_data + '/img/*')[:1]
    images = []

    for f in files:
        root = f[-9:-4]
        img = load_image(f, False)
        images.append([img])
    return images


def load_image(f, flat=False, net_type="VGG"):
    img = cv2.imread(f)

    data = np.asarray(img, dtype="float32")

    if flat:
        img_size = data.shape[0] * data.shape[1] * data.shape[2]
        data = img.reshape((1,img_size))
    else:
        # Add the batch dimension to the image
        if net_type is "VGG":
            data = data.reshape([1] + list(data.shape))
        elif net_type is "Custom":
            data = data.reshape([1] + list(data.shape))
            data = np.swapaxes(data,2,3)
            data = np.swapaxes(data,1,2)
        # Set the dimension order to (BATCH, DEPTH, WIDTH, HEIGHT)
    
    return data


def main(argv):
    
    argparser = argparse.ArgumentParser(description='Running this script will resize all the images to the provided size')
    argparser.add_argument('-d', '--data-dir',
        dest='data_dir',
        help='Data root directory',
        required=True)
    argparser.add_argument('-s', '--size',
        dest='image_size',
        help='A comma delimited tuple representing size in the order (height, width). Example: --size 210,280',
        required=True)

    args = argparser.parse_args(argv)

    data_dir = args.data_dir
    image_size = args.image_size
    image_size = map(lambda size: int(size), image_size.split(','))
    if len(image_size) != 2:
        print "Invalid image size input: %s" % args.image_size
    else:
        resize_bulk(data_dir, img_size=image_size)

if __name__ == "__main__":
    main(sys.argv[1:])
