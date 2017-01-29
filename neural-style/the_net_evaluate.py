
import tensorflow as tf
from the_net import TheNet
from os import path
import argparse
import sys
import json
import cv2

from BaseHTTPServer import BaseHTTPRequestHandler
import SocketServer

PORT = 8000

sess = tf.Session()
decoder = json.JSONDecoder()

def init_models(model_files, sess):

    thenet = TheNet()
    x = tf.placeholder(dtype=tf.float32, shape=(1, None, None, 3), name='placeholder')

    for model_file in model_files:
        model_name = path.basename(model_file).replace('.npz', '')
        print "Initializing %s model" % model_name

        with tf.variable_scope(model_name):
            thenet.create_model(x, trainable=False)
            thenet.load_model(model_file, sess)


def consume_model(image, sess, model_name):

    batch_shape = [1] + list(image.shape)
    x = tf.placeholder(dtype=tf.float32, shape=batch_shape, name='input_image')
    thenet = TheNet()

    with tf.variable_scope(model_name, reuse=True):
        model = thenet.create_model(x, trainable=False, reuse=True)

    transformed_image = sess.run(model, feed_dict={
        x: image.reshape(batch_shape)
    })

    return transformed_image.reshape(image.shape)


# Create custom HTTPRequestHandler class
class TheNetHTTPRequestHandler(BaseHTTPRequestHandler):

    # handle GET command
    def do_POST(self):

        try:
            content_len = int(self.headers.getheader('content-length', 0))
            post_body = self.rfile.read(content_len)
            body_obj = decoder.decode(post_body)

            image_path = body_obj['image_file_path']
            models = body_obj['models']

            image = cv2.imread(image_path)

            result_images = []
            for model_name in models:
                transformed = consume_model(image, sess, model_name)
                image_words = image_path.split('.')
                transform_path = '%s_%s.%s' % (image_words[0], model_name, image_words[1])
                result_images += [transform_path]
                cv2.imwrite(transform_path, transformed)

            # send code 200 response
            self.send_response(200)

            # send header first
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            # send file content to client
            json_obj = {
                "result_images": result_images
            }

            self.wfile.write(json.dumps(json_obj))
        except BaseException as e:
            print e.message
            self.send_response(500)
        return


def main(argv):

    argparser = argparse.ArgumentParser(prog="The Net model Evaluation Server")
    argparser.add_argument('-p', '--port', default=PORT, type=int,
                           dest='port',
                           help='Port to bind the server on.')
    argparser.add_argument('-m', '--model-files', required=True,
                           dest='model_files',
                           help='Comma delimited list of model files. Example: --model-files starry_night.npz,picasso.npz')

    args = argparser.parse_args(argv)
    model_files = args.model_files.split(',')

    init_models(model_files, sess)

    Handler = TheNetHTTPRequestHandler

    httpd = SocketServer.TCPServer(("", args.port), Handler)

    print "serving at port", PORT
    httpd.serve_forever()

if __name__ == "__main__":
    main(sys.argv[1:])
