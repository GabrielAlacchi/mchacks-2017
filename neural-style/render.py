import tensorflow as tf
import numpy as np
from scipy.misc import imread, imsave, imresize
import argparse
import time
from os import path, listdir
import sys
import simplejson as json

from BaseHTTPServer import BaseHTTPRequestHandler
import SocketServer

decoder = json.JSONDecoder()
sess = tf.Session()
saver = None
model_dir = None

PORT = 8000


def parse_args():
    parser = argparse.ArgumentParser(description="Render image using pretrained model.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="./output.png")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--arch", type=str, default="./models/model.meta")
    args = parser.parse_args()

    args.image = imread(args.input, mode="RGB").astype(np.float32)
    args.image = np.expand_dims(args.image, axis=0)
    return args


def init_models(meta_graph):

    print "Loading meta graph..."
    return tf.train.import_meta_graph(meta_graph, clear_devices=True)


def consume_model(image, model_name):

    image = np.expand_dims(image, axis=0)

    with sess.as_default():
        saver.restore(sess, path.join(model_dir, model_name + '.model'))
        inputs = tf.get_collection("inputs")
        outputs = tf.get_collection("output")

        if len(inputs) == 0 and len(outputs) == 0:
            raise ValueError("Invalid model_name %s" % model_name)
        else:
            input_pl = inputs[0]
            output = outputs[0]
            result = output.eval({input_pl: image})
            result = np.squeeze(result, axis=0)
            return result


# Create custom HTTPRequestHandler class
class TheNetHTTPRequestHandler(BaseHTTPRequestHandler):

    # handle GET command
    def do_POST(self):

        try:
            content_len = int(self.headers.getheader('content-length', 0))
            post_body = self.rfile.read(content_len)
            body_obj = decoder.decode(post_body)

            image_path = body_obj['image_path']
            model_name = body_obj['model']

            image = imread(image_path, mode='RGB')

            if image.shape[0] > 828 or image.shape[1] > 828:
                ratio = 828.0 / float(max(image.shape[0], image.shape[1]))
                im_shape = map(lambda x: int(ratio * x), list(image.shape))
                image = imresize(image, (im_shape[0], im_shape[1]))
                imsave(image_path, image)

            transformed = consume_model(image, model_name)
            image_words = image_path.split('.')
            transform_path = '%s_%s.%s' % (image_words[0], model_name, image_words[1])
            result_image = transform_path
            imsave(transform_path, transformed)

            # send code 200 response
            self.send_response(200)

            # send header first
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            # send file content to client
            json_obj = {
                "result_image": result_image
            }

            self.wfile.write(json.dumps(json_obj))
        except BaseException as e:
            print e.message
            self.send_response(500)
        return


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(prog="Model evaluation script")
    argparser.add_argument('-p', '--port', default=PORT, type=int,
                           dest='port',
                           help='Port to bind the server on.')
    argparser.add_argument('-m', '--model-dir', required=True,
                           dest='model_dir',
                           help='directory with all the .models in it')

    args = argparser.parse_args(sys.argv[1:])

    saver = init_models(path.join(args.model_dir, 'model.meta'))

    model_dir = args.model_dir
    Handler = TheNetHTTPRequestHandler

    httpd = SocketServer.TCPServer(("", args.port), Handler)

    print "serving at port", PORT
    httpd.serve_forever()
